"""Read-only G5 probes. Hooks observe raw Mapping inputs without changing outputs.

Regions reuse G4's fixed initial-GS distance quartiles (not Anchor re-ranking).
The Uniform comparison is a same-state operator probe, never a trained model.
"""
from contextlib import contextmanager
import csv
import dataclasses
import hashlib
import math
from pathlib import Path
import struct
import zlib
import numpy as np
import torch
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions
from scripts.training.gs_decoder import interpolate_anchor_displacements
from .motion import geometry_weights

SELECTED = (1, 5, 10, 15)
REGION_NOTE = 'G4 fixed_regions: initial valid GS distance rank, bottom/top floor(N/4); Anchor inherits its initial GS region; no new distance threshold'


def write_png(path, rgb):
    """Lossless RGB PNG using stdlib; no Pillow/matplotlib server dependency."""
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('PNG expects nonempty uint8 H,W,3')
    def chunk(name,payload):
        return struct.pack('>I',len(payload))+name+payload+struct.pack('>I',zlib.crc32(name+payload)&0xffffffff)
    height,width=rgb.shape[:2]
    scan=b''.join(b'\x00'+np.ascontiguousarray(row).tobytes() for row in rgb)
    encoded=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
             +chunk(b'IDAT',zlib.compress(scan,6))+chunk(b'IEND',b''))
    Path(path).write_bytes(encoded)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite diagnostic/model/optimizer tensor')
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite_tree(v)


def cpu_tree(value):
    """Analysis copies only; never replace a live training/AR tensor."""
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name:cpu_tree(getattr(value,f.name)) for f in dataclasses.fields(value)})
    return value


def stats(values, prefix='', mask=None):
    """Filter BEFORE finite/quantile checks, excluding padding and invalid data."""
    values = values.detach()
    if mask is not None: values = values[mask]
    values = values.flatten().double()
    finite_tree(values)
    names = ('min', 'p50', 'p95', 'p99', 'max', 'mean', 'rms')
    if not values.numel():
        return {prefix + 'count': 0, **{prefix + k: None for k in names}}
    quant = torch.quantile(values, values.new_tensor([.5, .95, .99])).tolist()
    result = dict(count=values.numel(), min=float(values.min()), p50=quant[0], p95=quant[1],
                  p99=quant[2], max=float(values.max()), mean=float(values.mean()),
                  rms=float(values.square().mean().sqrt()))
    return {prefix + k: v for k, v in result.items()}


def vectors(values, prefix='', factor=1.):
    values = values.detach().reshape(-1, 3).double() * factor
    finite_tree(values)
    if not len(values):
        return {prefix + k: None for k in ('rms', 'mean_x', 'mean_y', 'mean_z', 'nonzero_fraction')}
    result = dict(rms=float(values.square().sum(-1).mean().sqrt()),
                  nonzero_fraction=float((values.norm(dim=-1) > 0).double().mean()))
    for axis, i in zip('xyz', range(3)):
        result['mean_' + axis] = float(values[:, i].mean())
        result['positive_' + axis] = float((values[:, i] > 0).double().mean())
        result['negative_' + axis] = float((values[:, i] < 0).double().mean())
    return {prefix + k: v for k, v in result.items()}


def direction_comparison(a, b):
    a, b = a.double(), b.double()
    product = a.norm(dim=-1) * b.norm(dim=-1)
    valid = product > torch.finfo(a.dtype).eps
    cosine = ((a * b).sum(-1) / product.clamp_min(torch.finfo(a.dtype).eps))[valid]
    # Near-zero directions are undefined, never labelled as wrong directions.
    return dict(direction_defined_count=int(valid.sum()),
                direction_cosine_mean=float(cosine.mean()) if len(cosine) else None,
                direction_negative_fraction=float((cosine < 0).double().mean()) if len(cosine) else None)


def entropy(weights):
    return -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)


@contextmanager
def mapping_context(model):
    seen = []
    handle = model.direct_motion.mapping.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach()))
    try: yield seen
    finally: handle.remove()


def anchor_regions(x, field):
    groups = fixed_regions(x)
    return groups, {key: torch.isin(field.anchor_indices[0], ids) & field.anchor_mask[0]
                    for key, ids in groups.items()}


def uniform_propagation(seed, indices, graph_mask, anchor_valid, hand_valid):
    """Frozen same-state vector operator: no model hook/replacement or training."""
    allowed = graph_mask[:, :, None] & anchor_valid[:, :, None, None] & hand_valid[:, None, :, None]
    weights = allowed.to(seed.dtype) / allowed.sum(-1, keepdim=True).clamp_min(1)
    batch = torch.arange(len(seed), device=seed.device)[:, None, None]
    neighbor = seed[batch, indices].permute(0, 1, 3, 2, 4)
    return seed + .5 * (weights[..., None] * (neighbor - seed[:, :, :, None])).sum(-2)


@torch.no_grad()
def controlled_neighbor_probe(module, *, query, hand_tokens, basis18, seed, initial_xyz,
                              current_xyz, anchor_valid, hand_valid, indices, neighbor_mask,
                              reference_weights=None):
    args = dict(query=query, hand_tokens=hand_tokens, basis18=basis18, direct_seed_per_hand=seed,
                initial_xyz=initial_xyz, current_xyz=current_xyz, anchor_valid=anchor_valid,
                hand_valid=hand_valid, indices=indices, neighbor_mask=neighbor_mask)
    before, weights = module(**args)
    if reference_weights is not None and not torch.equal(weights,reference_weights):
        raise AssertionError('Controlled probe did not reproduce actual rollout weights')
    active = (anchor_valid[0] & neighbor_mask[0, :, 1:].any(-1)).nonzero().flatten()
    hands = hand_valid[0].nonzero().flatten()
    if not len(active) or not len(hands):
        return dict(verified=False, reason='No active non-Self neighbor/hand')
    a, hand = int(active[0]), int(hands[0])
    slot = int(neighbor_mask[0, a, 1:].nonzero()[0]) + 1
    b = int(indices[0, a, slot])
    changed = seed.clone(); delta = seed.new_tensor([.01, -.02, .03])
    changed[0, b, hand] += delta
    after, altered_weights = module(**dict(args, direct_seed_per_hand=changed))
    response = after[0, a, hand] - before[0, a, hand]
    expected = .5 * weights[0, a, hand, slot] * delta
    torch.testing.assert_close(response, expected, rtol=2e-4, atol=2e-7)
    if not torch.equal(weights, altered_weights) or not response.abs().sum() > 0:
        raise AssertionError('Neighbor-seed probe did not match frozen Score semantics')
    return dict(verified=True, target_anchor=a, source_anchor=b, hand=hand,
                perturbation_normalized_xyz=delta.cpu().tolist(), response_normalized_xyz=response.cpu().tolist(),
                expected_response_normalized_xyz=expected.cpu().tolist(), attention_weights_bit_exact=True,
                interpretation='Neighbor signed vector changes propagated output; Score has no direct neighbor-seed input. Query/geometry/basis held fixed. Not a full-rollout causal attribution.')


@torch.no_grad()
def graph_diagnostics(x, trace, factor):
    field = trace['field']; indices = trace['anchor_neighbor_indices'][0]
    valid = trace['anchor_neighbor_mask'][0]
    initial = x['gs_xyz'][0, field.anchor_indices[0]]
    distance = (initial[:, None] - initial[indices]).norm(dim=-1)
    nonself = valid & (indices != torch.arange(len(indices), device=indices.device)[:, None])
    return dict(**stats(distance * factor, 'edge_mm_', nonself),
                **stats(distance / field.bandwidth[0], 'edge_over_bandwidth_', nonself),
                valid_anchors=int(field.anchor_mask.sum()), graph_slots=indices.shape[-1],
                note='Fixed initial spatial neighbors, not material topology. Self edges excluded from distance summary.')


@torch.no_grad()
def collect(model, x, trace, contexts, *, variant, window, stage, factor):
    finite_tree(trace['steps']); finite_tree(contexts)
    if len(contexts) != 15: raise AssertionError('Expected actual H15 raw Mapping inputs')
    field = trace['field']; groups, regions = anchor_regions(x, field)
    geometry, propagation = [], []
    for step, s in enumerate(trace['steps']):
        base = dict(variant=variant, window=window, stage=stage, horizon=step + 1)
        valid_a = field.anchor_mask[0]; hands = s['hand_valid'][0]
        seed = s['direct_seed_per_hand'][0]
        propagated = s.get('propagated_seed_per_hand', s['direct_seed_per_hand'])[0]
        local_gs = interpolate_anchor_displacements(s['local_correction'], field, x['gs_mask'])[0]
        if variant == 'propagate':
            weights = s['anchor_attention_weights'][0]
            active = valid_a[:, None] & hands[None]
            if not torch.allclose(weights.sum(-1)[active], torch.ones_like(weights.sum(-1)[active]), atol=2e-6, rtol=0):
                raise AssertionError('A2A weights not normalized on valid rows')
            if (weights.sum(-1)[~active] != 0).any(): raise AssertionError('A2A mask leakage')
            uniform = uniform_propagation(s['direct_seed_per_hand'], trace['anchor_neighbor_indices'],
                trace['anchor_neighbor_mask'], field.anchor_mask, s['hand_valid'])[0]
            for hand in hands.nonzero().flatten().tolist():
                for region, mask in regions.items():
                    if not mask.any(): continue
                    d, p = seed[mask, hand], propagated[mask, hand]
                    w = weights[mask, hand]
                    k = trace['anchor_neighbor_mask'][0, mask].sum(-1)
                    propagation.append(dict(base, hand=('left', 'right')[hand], region=region,
                        anchor_count=int(mask.sum()), self_weight_mean=float(w[:, 0].mean()),
                        nonself_weight_mean=float((1-w[:, 0]).mean()), entropy_mean=float(entropy(w).mean()),
                        entropy_over_logk_mean=float((entropy(w) / k.float().log().clamp_min(1e-20)).mean()),
                        row_sum_error_max=float((w.sum(-1)-1).abs().max()),
                        **vectors(d, 'direct_mm_', factor), **vectors(p, 'propagated_mm_', factor),
                        **vectors(p-d, 'mix_change_mm_', factor),
                        **vectors(p-uniform[mask, hand], 'learned_minus_uniform_mm_', factor),
                        **vectors(d-d.mean(0), 'direct_spatial_residual_mm_', factor),
                        **vectors(p-p.mean(0), 'propagated_spatial_residual_mm_', factor),
                        **direction_comparison(d, p)))
        if step + 1 not in SELECTED: continue
        cp = x['controller_points'][0, step]
        current_hands = x['action_valid'][0, step] & x['operation_hands'][0] & torch.isfinite(x['action_state'][0, step]).all(-1)
        valid_cp = x['controller_mask'][0, step] & current_hands[:, None] & torch.isfinite(cp).all(-1)
        safe_cp = torch.where(valid_cp[..., None], cp, 0.)
        beta, relative = geometry_weights(s['anchor_xyz'], safe_cp[None], valid_cp[None], field.anchor_mask, field.bandwidth)
        for hand in hands.nonzero().flatten().tolist():
            if not valid_cp[hand].any(): continue
            distance = relative[0, :, hand, valid_cp[hand]].norm(dim=-1)
            for region, mask in regions.items():
                if not mask.any(): continue
                ctx = contexts[step][0, mask, hand]
                basis = s['anchor_hand_basis18'][0, mask, hand]
                w = beta[0, mask, hand]
                if not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=2e-6, rtol=0):
                    raise AssertionError('Geometric current Controller Softmax row sum')
                row = dict(base, hand=('left','right')[hand], region=region, anchor_count=int(mask.sum()),
                           valid_controllers=int(valid_cp[hand].sum()), bandwidth_mm=float(field.bandwidth[0])*factor,
                           **stats(distance[mask] * field.bandwidth[0] * factor, 'pair_distance_mm_'),
                           **stats(distance[mask], 'pair_distance_over_bandwidth_'),
                           **stats(distance[mask].min(-1).values * field.bandwidth[0] * factor, 'nearest_mm_'),
                           **stats(distance[mask].min(-1).values, 'nearest_over_bandwidth_'),
                           **stats(w.max(-1).values, 'softmax_max_weight_'),
                           **stats(entropy(w), 'softmax_entropy_'),
                           **stats(w.sum(-1), 'softmax_row_sum_'),
                           **stats(basis.norm(dim=-1), 'basis18_norm_'),
                           **stats(basis.norm(dim=-1)*factor, 'basis18_norm_mm_'),
                           normalized_displacement_to_mm=factor)
                for name, start, end in [('query',0,128),('hand_token',128,256),('geometry',256,260),
                                          ('signed_basis',260,278),('anchor_history',278,281),
                                          ('previous_cumulative',281,284),('g4_hand',284,287)]:
                    row.update(stats(ctx[:,start:end].abs(), name+'_abs_'))
                for j in range(18):
                    row[f'basis18_{j}_mean'] = float(basis[:,j].mean())
                    row[f'basis18_{j}_abs_max'] = float(basis[:,j].abs().max())
                    row[f'basis18_{j}_mean_mm'] = float(basis[:,j].mean())*factor
                row.update(vectors(propagated[mask,hand], 'per_hand_local_mm_', factor))
                row.update(vectors(s['hand_contribution'][0,hand][None], 'per_hand_g4_mm_', factor))
                total_local = s['local_correction'][0,mask]
                global_vector = s['g4_global_delta'][0]
                row.update(vectors(total_local,'total_local_anchor_mm_',factor))
                row.update(vectors(global_vector[None],'g4_global_mm_',factor))
                row.update(vectors(total_local+global_vector,'combined_local_global_mm_',factor))
                loc = local_gs[groups[region]]
                row.update(vectors(loc,'local_gs_mm_',factor))
                row.update(vectors(loc-loc.mean(0),'local_gs_demeaned_mm_',factor))
                local_norm=total_local.norm(dim=-1);g_norm=global_vector.norm()
                defined=(local_norm+g_norm)>0
                ratio=(total_local+global_vector).norm(dim=-1)/(local_norm+g_norm).clamp_min(1e-20)
                row['cancellation_ratio_mean'] = float(ratio[defined].mean()) if defined.any() else None
                row['cancellation_defined_count'] = int(defined.sum())
                for prefix, endpoint_weights in [('cumulative',s['controller_cumulative_weights']),('adjacent',s['controller_adjacent_weights'])]:
                    ew=endpoint_weights[0,mask,hand]; observed=ew.sum(-1)>0
                    row.update(stats(ew.max(-1).values, prefix+'_softmax_max_',observed))
                    row.update(stats(entropy(ew),prefix+'_softmax_entropy_',observed))
                geometry.append(row)
    return geometry, propagation


def parameter_evidence(model, before, grads, *, lr=1e-4, decay=1e-4):
    groups = dict(direct_output='direct_motion.mapping.3.', direct_hidden='direct_motion.mapping.',
                  propagation_score='propagation.', g4_global='global_motion.',
                  motion_decoder='motion.', controller_encoder='conditions.controller.',
                  gs_encoder='gs_encoder.', cross_attention='cross.')
    rows={}
    for group,prefix in groups.items():
        names=[n for n in before if n.startswith(prefix) and not (group=='direct_hidden' and n.startswith('direct_motion.mapping.3.'))]
        if not names: continue
        params=dict(model.named_parameters())
        def norm(values): return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))
        diffs=[params[n].detach()-before[n] for n in names]
        nondecay=[params[n].detach()-before[n]+lr*decay*before[n] if grads[n] is not None else params[n].detach()-before[n] for n in names]
        rows[group]=dict(parameter_norm_before=norm([before[n] for n in names]),
            raw_gradient_norm=norm([grads[n] for n in names if grads[n] is not None]),
            clipped_gradient_norm=norm([params[n].grad for n in names if params[n].grad is not None]),
            actual_update_norm=norm(diffs),update_minus_adamw_decay_norm=norm(nondecay),
            grad_none_parameters=sum(grads[n] is None for n in names),parameter_tensors=len(names))
    return rows


def write_csv(path, rows):
    keys=sorted(set().union(*(r.keys() for r in rows))) if rows else ['no_rows']
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
