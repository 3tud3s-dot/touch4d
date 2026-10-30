"""A/B share one causal G5 rollout; B adds only an Anchor-state feature readout.

S is the object Anchor feature BEFORE adding the prediction-time embedding.
Z is a per-window local variable; no state crosses shuffled windows. All
Scale/Rotation/Opacity/SH attributes remain initial. No GEO or A2A is present.
"""
import random
import numpy as np
import torch
from scripts.training.model import initial_knn, masked
from scripts.training.v4_model import fixed_field
from scripts.training.gs_decoder import pool_anchor_features, interpolate_anchor_displacements
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from .basis import observed_signed_basis
from .state import AnchorStateMemory

VARIANTS = ('g5_struct', 'g6_state_struct')
ARCHITECTURES = dict(g5_struct='v4_g5_struct_v1', g6_state_struct='v4_g6_state_struct_v1')


class StructStatePredictor(G5DirectPredictor):
    def __init__(self, variant='g5_struct', horizon=15, prediction_interval=1):
        if variant not in VARIANTS:
            raise ValueError('Expected g5_struct or g6_state_struct')
        super().__init__(horizon=horizon, prediction_interval=prediction_interval)
        self.variant = variant
        self.has_state = variant == 'g6_state_struct'
        self.config = dict(self.config, architecture=ARCHITECTURES[variant], variant=variant,
            checkpoint_kind=variant+'_v1', anchor_state=self.has_state,
            state_width=128 if self.has_state else 0, state_input_width=301 if self.has_state else 0,
            state_initialization='pre_time_object_anchor_features' if self.has_state else 'none',
            state_readout='additive_feature_zero_projection' if self.has_state else 'none',
            state_cross_window=False, geometry_attributes='initial')
        # Append after the complete original G5 initialization. Shared weights
        # are thus exactly identical, not merely seeded with the same number.
        if self.has_state:
            # Extra state parameters must not advance the external/model-data
            # CPU generator. Both variants leave exactly the same RNG state.
            with torch.random.fork_rng(devices=[]):
                self.state_memory = AnchorStateMemory()

    def forward(self, x, *, return_trace=False, state_readout_off=False):
        if state_readout_off and not self.has_state:
            raise ValueError('state_readout_off is a B-only frozen diagnostic')
        h = self.config['horizon']
        frame_ids = x.get('condition_frame_ids')
        if (frame_ids is None or frame_ids.dtype != torch.int64 or frame_ids.shape != (len(x['gs_xyz']), h)
                or not torch.all(frame_ids[:, 1:]-frame_ids[:, :-1] == self.config['prediction_interval'])):
            raise ValueError('H/interval condition frames mismatch')
        if any(x[key].shape[1] != h for key in ('action_state','controller_points','tactile')):
            raise ValueError('Exactly H observed conditions required')
        keep = x['gs_mask']
        if keep.dtype != torch.bool or not keep.any(1).all():
            raise ValueError('Each sample requires valid GS')
        initial, attr = masked(x['gs_xyz'], keep), masked(x['gs_attr'], keep)
        if not torch.isfinite(initial).all() or not torch.isfinite(attr).all():
            raise ValueError('Nonfinite initial GS')
        field = fixed_field(initial, keep)
        neighbors = initial_knn(initial, keep, 16)
        batches = torch.arange(len(initial), device=initial.device)[:, None]
        initial_anchors = masked(initial[batches, field.anchor_indices], field.anchor_mask)
        current = initial
        previous_anchor = initial_anchors
        previous_d = torch.zeros_like(initial_anchors)
        previous_cp = previous_valid = previous_z = None
        states, steps = [], []
        for step in range(h):
            features = masked(self.gs_encoder(torch.cat((current, attr), -1)), keep)
            features = self.local(features, current, keep, neighbors)
            # Preserve the original addition/masking order for bitwise parity.
            object_features = pool_anchor_features(features, field)
            query = masked(object_features+self.pred_time.weight[step], field.anchor_mask)
            object_features = masked(object_features, field.anchor_mask)
            anchors = masked(current[batches, field.anchor_indices], field.anchor_mask)
            tools, valid, cp, condition_trace = self.conditions(x, step)
            for block in self.cross:
                query, alpha = block(query, tools, anchors, cp, field.bandwidth, field.anchor_mask, valid)
            adjacent = torch.zeros_like(valid) if step == 0 else valid & previous_valid
            cp_delta = torch.zeros_like(cp) if step == 0 else masked(cp-previous_cp, adjacent)
            weights = alpha[..., :60]
            tool_motion = torch.einsum('bahj,bjd->bahd', weights, cp_delta).flatten(-2)
            history = torch.zeros_like(anchors) if step == 0 else anchors-previous_anchor
            head_mass = (weights*adjacent[:, None, None]).sum(-1)
            anchor_history_valid = field.anchor_mask[..., None].to(query.dtype)*(step > 0)
            base_query = query
            state_input = state_z = state_readout = state_basis = state_hand12 = state_hand_valid = None
            if self.has_state:
                state_hand12, state_hand_valid, state_basis, _ = observed_signed_basis(x, step,
                    anchor_xyz=anchors, anchor_valid=field.anchor_mask, cp=cp, cp_valid=valid,
                    previous_cp=previous_cp, previous_valid=previous_valid, sigma=field.bandwidth)
                hand_flags = state_hand_valid[:, None].expand(-1, anchors.shape[1], -1).to(query.dtype)
                state_input = torch.cat((object_features, base_query, state_basis.flatten(-2),
                    history, previous_d, hand_flags, anchor_history_valid), -1)
                state_input = masked(state_input, field.anchor_mask)
                if step == 0:
                    previous_z = object_features  # Z0 = S0, reset for EVERY window.
                state_z, state_readout = self.state_memory(state_input, previous_z, field.anchor_mask)
                query = base_query if state_readout_off else masked(base_query+state_readout, field.anchor_mask)
                previous_z = state_z  # Full H-step state graph, no detach.
            motion_input = torch.cat((query, tool_motion, history, previous_d, head_mass, anchor_history_valid), -1)
            base_d = masked(self.motion(motion_input), field.anchor_mask)
            global_delta, global_trace = self.global_motion(x, step, query=query, anchor_valid=field.anchor_mask,
                hand_tokens=condition_trace['tokens62'][:, :2], cp=cp, cp_valid=valid,
                previous_cp=previous_cp, previous_valid=previous_valid)
            seed, direct_trace = self.direct_motion(x, step, query=query, anchor_xyz=anchors,
                anchor_valid=field.anchor_mask, hand_tokens=condition_trace['tokens62'][:, :2],
                hand_basis12=global_trace['hand_basis'], hand_valid=global_trace['hand_valid'], cp=cp, cp_valid=valid,
                previous_cp=previous_cp, previous_valid=previous_valid, anchor_history=history, previous_d=previous_d,
                global_hand_contribution=global_trace['hand_contribution'], sigma=field.bandwidth)
            local = masked(seed.sum(2), field.anchor_mask)
            g4_d = masked(base_d+global_delta[:, None], field.anchor_mask)
            d = masked(g4_d+local, field.anchor_mask)
            next_xyz = masked(initial+interpolate_anchor_displacements(d, field, keep), keep)
            if not torch.isfinite(next_xyz).all():
                raise FloatingPointError('Nonfinite Struct/State prediction')
            states.append(next_xyz)
            if return_trace:
                local_gs = interpolate_anchor_displacements(local, field, keep)
                count = keep.sum(1, keepdim=True).clamp_min(1).to(local_gs.dtype)
                mean = masked(local_gs, keep).sum(1)/count
                residual = masked(local_gs-mean[:, None], keep)
                rms = torch.linalg.vector_norm(residual, dim=(1,2))/count.squeeze(-1).sqrt()
                trace = dict(alpha=alpha, anchor_xyz=anchors, anchor_history=history, tool_motion=tool_motion,
                    adjacent_valid=adjacent, head_history_mass=head_mass, base_cumulative_d=base_d,
                    global_delta=global_delta, cumulative_d=d, previous_d=previous_d, state=next_xyz,
                    g4_global_delta=global_delta, hand_basis12=global_trace['hand_basis'],
                    direct_seed_per_hand=seed, local_correction=local, final_cumulative_anchor_displacement=d,
                    local_gs_mean=mean, local_gs_demeaned_rms=rms,
                    **global_trace, **condition_trace, **direct_trace)
                trace.update(object_anchor_features=object_features, base_query=base_query, enhanced_query=query,
                    state_input=state_input, state_z=state_z, state_readout=state_readout,
                    state_basis18=state_basis, state_hand_basis12=state_hand12, state_hand_valid=state_hand_valid)
                steps.append(trace)
            previous_anchor, previous_d = anchors, d
            previous_cp, previous_valid = cp, valid
            current = next_xyz
        prediction = {'xyz':torch.stack(states, 1)}
        for key, begin, end in (('log_scales',0,3),('quaternions',3,7),('opacity_logits',7,8),('sh_dc',8,11)):
            prediction[key] = attr[:, None, :, begin:end].expand(-1,h,-1,-1)
        return (prediction, dict(steps=steps, field=field)) if return_trace else prediction


def build_model(variant, *, horizon=15, prediction_interval=1):
    return StructStatePredictor(variant, horizon=horizon, prediction_interval=prediction_interval)


def fresh_model(variant, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return build_model(variant)


def paired_initialization(verify_outputs=True):
    torch.manual_seed(42)
    reference = G5DirectPredictor(horizon=15, prediction_interval=1)
    a = fresh_model(VARIANTS[0]); rng_a = torch.get_rng_state().clone()
    b = fresh_model(VARIANTS[1]); rng_b = torch.get_rng_state().clone()
    if not torch.equal(rng_a,rng_b):
        raise AssertionError('State initialization advanced external RNG')
    shared = reference.state_dict()
    for name, value in shared.items():
        if not torch.equal(value, a.state_dict()[name]) or not torch.equal(value, b.state_dict()[name]):
            raise AssertionError('Shared G5 initialization mismatch: '+name)
    if set(a.state_dict()) != set(shared):
        raise AssertionError('A must add no parameters or buffers')
    if any(p.count_nonzero() for p in b.state_memory.readout.parameters()):
        raise AssertionError('State readout must be exactly zero')
    exact = None
    if verify_outputs:
        from scripts.experiments.v4_stage1.fixtures import synthetic_batch
        from scripts.training.v4_runtime import prepare_batch
        from scripts.data_process import DataProcessor
        x = prepare_batch(synthetic_batch(60,15,1), DataProcessor('001-rope'), reference)['inputs']
        with torch.no_grad():
            original = reference(x)
            exact = {v:{k:torch.equal(pred[k], value) for k,value in original.items()}
                     for v,pred in ((VARIANTS[0],a(x)),(VARIANTS[1],b(x)))}
        if not all(all(row.values()) for row in exact.values()):
            raise AssertionError('Initial full H15 output mismatch')
    return dict(seed=42,shared_exact=True,zero_state_projection=True,external_cpu_rng_equal=True,
        shared_g5_state_hash=state_hash(shared),
        shared_g5_parameter_hash=state_hash(dict(reference.named_parameters())),
        state_module_hash=state_hash(b.state_memory.state_dict()),
        variant_model_hashes={v:state_hash(m.state_dict()) for v,m in zip(VARIANTS,(a,b))},
        parameter_counts={v:sum(p.numel() for p in m.parameters()) for v,m in zip(VARIANTS,(a,b))},
        added_state_parameters=sum(p.numel() for p in b.state_memory.parameters()),h15_all_attributes_exact=exact,
        state_definition='S before time embedding; Z0=S0; per-window reset; no detach; no future inputs')
