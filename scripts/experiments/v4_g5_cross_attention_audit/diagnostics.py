"""Read-only summaries. Feature RMS is sqrt(mean(feature**2)), not mm."""
import torch
from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.training.losses import nearest_distances
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions
from scripts.experiments.v4_g5_motion_attribution.metrics import rms


def anchor_regions(inputs, field):
    result = {"all": field.anchor_mask[0].nonzero().flatten()}
    for name, ids in fixed_regions(inputs).items():
        belongs = torch.zeros_like(inputs['gs_mask'][0])
        belongs[ids] = True
        result[name] = (belongs[field.anchor_indices[0]] & field.anchor_mask[0]).nonzero().flatten()
    return result


def feature_rms(value):
    return float(value.square().mean().sqrt()) if value.numel() else None


def attention_rows(inputs, trace, captures, scale, scope):
    rows = []
    regions = anchor_regions(inputs, trace['field'])
    for k, (step, raw) in enumerate(zip(trace['steps'], captures), 1):
        for layer, call in enumerate(raw['cross_calls']):
            query, alpha = call['args'][0], call['output'][1]
            update = raw['cross_projection_raw'][layer]
            valid = call['args'][6][0]
            allowed = valid[None, None, :].expand(alpha.shape[1], alpha.shape[2], -1)
            controller = alpha[0, ..., :60]
            if torch.count_nonzero(controller.masked_select(~allowed)):
                raise AssertionError('Invalid Controller has nonzero Attention mass')
            entropy = -(alpha[0] * alpha[0].clamp_min(torch.finfo(alpha.dtype).tiny).log()).sum(-1)
            mass = controller.sum(-1)
            # Conditional-on-Controller entropy, undefined for zero Controller mass.
            safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
            conditional = controller / safe_mass[..., None]
            centropy = -(conditional * conditional.clamp_min(torch.finfo(alpha.dtype).tiny).log()).sum(-1)
            for region, ids in regions.items():
                if not len(ids):
                    continue
                qr, ur = feature_rms(query[0, ids]), feature_rms(update[0, ids])
                for head in range(4):
                    motion = step['tool_motion'][0, ids, head*3:(head+1)*3] * scale * 1000
                    available = mass[ids, head] > 0
                    row = dict(**scope, horizon=k, layer=layer+1, head=head, region=region,
                        anchors=len(ids), query_feature_rms=qr, value_update_feature_rms=ur,
                        update_to_query_ratio=(ur/qr if qr and qr > 0 else None),
                        controller_mass=float(mass[ids, head].mean()),
                        null_mass=float(alpha[0, ids, head, -1].mean()),
                        attention_entropy_nats=float(entropy[ids, head].mean()),
                        controller_conditional_entropy_nats=(float(centropy[ids, head][available].mean())
                                                            if available.any() else None),
                        final_layer_history_mass=(float(step['head_history_mass'][0, ids, head].mean())
                                                  if layer == 1 else None),
                        final_layer_tool_vector_rms_mm=rms(motion) if layer == 1 else None)
                    for axis, name in enumerate('xyz'):
                        row[f'final_layer_tool_{name}_rms_mm'] = (float(motion[:,axis].square().mean().sqrt())
                                                                if layer == 1 else None)
                    rows.append(row)
    return rows


def same_history_regions(inputs, trace, normal_step, changed, gt, scale, scope):
    """GS regions are fixed initially; GT is an unlabelled independent set."""
    factor, rows = scale * 1000, []
    field = trace['field']
    aregions = anchor_regions(inputs, field)
    gsregions = fixed_regions(inputs)
    original = normal_step['state'][0]
    fields = {
        'base': changed['base_cumulative_d']-normal_step['base_cumulative_d'],
        'global': (changed['global_delta']-normal_step['global_delta'])[:,None].expand_as(normal_step['base_cumulative_d']),
        'local': changed['local_correction']-normal_step['local_correction']}
    gsfields = {name:interpolate_anchor_displacements(value,field,inputs['gs_mask'])[0]*factor
                for name,value in fields.items()}
    for region, ids in gsregions.items():
        if not len(ids):
            continue
        a = aregions[region]
        diff = (changed['xyz'][0,ids]-original[ids])*factor
        row = dict(**scope, region=region, gs_count=len(ids), anchor_count=len(a),
            change_rms_mm=rms(diff),
            query_change_feature_rms=(feature_rms(changed['query'][0,a]-normal_step['_query'][0,a]) if len(a) else None),
            pred_to_gt_mm=float(nearest_distances(changed['xyz'][0,ids],gt,512).mean())*factor,
            normal_pred_to_gt_mm=float(nearest_distances(original[ids],gt,512).mean())*factor)
        for axis,name in enumerate('xyz'):
            row[f'change_mean_{name}_mm'] = float(diff[:,axis].mean())
        for branch,value in gsfields.items():
            row[branch+'_change_gs_rms_mm'] = rms(value[ids])
        rows.append(row)
    return rows
