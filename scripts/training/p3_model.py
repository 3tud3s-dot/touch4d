"""Shared one-step P3 dynamics; P2 remains a separate, unchanged predictor."""
import torch
from .model import PositionPredictor, initial_knn, masked
from .gs_decoder import build_initial_anchor_field, pool_anchor_features, interpolate_anchor_displacements
from .gs_temporal import TemporalAnchorFusion
from .gs_template_refiner import TemplateRefiner


class P3Predictor(PositionPredictor):
    def __init__(self, history=2, template_refinement=False, architecture='p3-one-step-v0',
                 scale_bound=1.3, opacity_bound=.5, **kwargs):
        if type(history) is not int or history not in (1, 2) or type(template_refinement) is not bool:
            raise ValueError("P3 history must be 1/2; template_refinement must be bool")
        if architecture != 'p3-one-step-v0' or scale_bound != 1.3 or opacity_bound != .5:
            raise ValueError("Unsupported P3 architecture or attribute bounds")
        defaults = dict(horizon=10, prediction_mode='autoregressive', output_mode='gs_geometry',
                        gs_local_attention='knn', gs_decoder='anchors', gs_structure='anchor_attention')
        for key, value in defaults.items():
            if key != 'horizon' and key in kwargs and kwargs[key] != value:
                raise ValueError(f"P3 requires {key}={value}")
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.config.update(architecture=architecture, history=history, template_refinement=template_refinement,
                           scale_bound=scale_bound, opacity_bound=opacity_bound)
        # Append all extra modules after the unchanged P2 constructor/RNG stream.
        self.temporal = TemporalAnchorFusion(defaults.get('width', 128), defaults.get('heads', 4)) if history == 2 else None
        self.refiner = TemplateRefiner(defaults.get('width', 128)) if template_refinement else None

    def _geometry(self, xyz, keep):
        return (initial_knn(xyz, keep, self.config['gs_local_k']),
                build_initial_anchor_field(xyz, keep, self.config['anchor_count'],
                    self.config['anchor_pool_k'], self.config['anchor_bandwidth']))

    def _encode(self, xyz, attr, keep, neighbors, field):
        features = self.gs_encoder(masked(torch.cat((xyz, attr), -1), keep))
        features = self.gs_local(features, xyz, keep, neighbors)
        anchors, centers = self.gs_structure.encode_anchors(features, xyz, keep, field)
        update = masked(self.gs_structure.output(anchors), field.anchor_mask)
        features = masked(features + interpolate_anchor_displacements(update, field, keep), keep)
        return features, anchors, centers, field.anchor_mask

    def forward(self, inputs, return_states=False):
        x = inputs; keep = x['gs_mask']; current = masked(x['gs_xyz'], keep)
        initial_attr = masked(x['gs_attr'], keep); attr = initial_attr
        if not all(torch.isfinite(v).all() for v in (current, attr)):
            raise FloatingPointError('Nonfinite valid current GS')
        if keep.any() and not torch.allclose(attr[..., 3:7][keep].norm(dim=-1),
                                             attr.new_ones(int(keep.sum())), atol=1e-5, rtol=0):
            raise ValueError('Expected normalized inherited quaternion')
        neighbors, field = self._geometry(current, keep)
        previous = None
        if self.temporal is not None:
            if not all(k in x for k in ('gs_prev_xyz', 'gs_prev_attr', 'gs_prev_mask')):
                raise ValueError('Two-frame P3 needs the previous GS, with an independent mask')
            pkeep = x['gs_prev_mask']; pxyz = masked(x['gs_prev_xyz'], pkeep)
            pattr = masked(x['gs_prev_attr'], pkeep)
            pneighbors, pfield = self._geometry(pxyz, pkeep)
            previous = self._encode(pxyz, pattr, pkeep, pneighbors, pfield)
        tokens, token_valid = self.encode_conditions(x)
        outputs, states = [], []
        b = len(current)
        for step in range(self.config['horizon']):
            encoded = self._encode(current, attr, keep, neighbors, field)
            features, anchors, centers, anchor_mask = encoded
            if self.temporal is not None:
                residual = self.temporal(anchors, previous[1], centers, previous[2], anchor_mask, previous[3])
                features = masked(features + interpolate_anchor_displacements(residual, field, keep), keep)
            # Shared dynamics and a common causal prefix for M1/M2/M3.
            memory = torch.cat((self.null_token.expand(b, -1, -1), tokens[:, :step+1].flatten(1, 2)), 1)
            valid = torch.cat((torch.ones((b, 1), device=keep.device, dtype=torch.bool),
                               token_valid[:, :step+1].flatten(1, 2)), 1)
            query = features + self.prediction_time.weight[step]
            for block in self.blocks:
                query = block(query, memory, valid)
            anchor_query = pool_anchor_features(query, field)
            delta = interpolate_anchor_displacements(self.position_head(anchor_query), field, keep)
            current = masked(current + delta, keep)
            attr = initial_attr if self.refiner is None else self.refiner(anchor_query, field, initial_attr, keep)
            if not torch.isfinite(current).all() or not torch.isfinite(attr).all():
                raise FloatingPointError('Nonfinite P3 state')
            outputs.append((current, attr))
            if return_states: states.append(dict(xyz=current, attr=attr))
            # Cache the previous frame's independently encoded state without detach.
            # Future predicted frames retain the t-template's fixed FPS/pools/RBF.
            previous = encoded
        prediction = dict(xyz=torch.stack([v[0] for v in outputs], 1),
            log_scales=torch.stack([v[1][..., :3] for v in outputs], 1),
            quaternions=torch.stack([v[1][..., 3:7] for v in outputs], 1),
            opacity_logits=torch.stack([v[1][..., 7:8] for v in outputs], 1),
            sh_dc=torch.stack([v[1][..., 8:] for v in outputs], 1))
        return (prediction, states) if return_states else prediction
