"""Bounded backward, signal and local-output perturbation measurements."""
import numpy as np
import torch
from scripts.experiments.fixed_window.fixed_fit import digest, TargetCache
from .core import (LOSSES, components, observe, altered_inputs, same_state, stats,
                   cosine, cosines, point_angles, fixed_regions, rms_mm,
                   finite_gradient, single_horizon, normalized_direction, parameter_summary)


def values(losses):
    out = {k: float(v.detach()) for k, v in losses.items()}
    out['alpha_raw'] = out['alpha_weighted']/.1
    return out


def signal_rows(batch, pred, trace, regions, scale_mm):
    x = batch['inputs']; cp = x['controller_points'].reshape(1, 10, 60, 3)
    rows = []
    for h, step in enumerate(trace['steps']):
        cm = step['adjacent_valid'][0]
        raw = torch.zeros_like(cp[0, h]) if h == 0 else cp[0, h]-cp[0, h-1]
        for region, ids in regions.items():
            a = step['alpha'][0, ids]; tool = step['tool_motion'][0, ids].reshape(-1, 4, 3)
            rows.append(dict(horizon=h+1, region=region, anchors=len(ids),
                valid_controllers=int(step['token_mask62'][0, 2:].sum()),
                adjacent_valid_controllers=int(cm.sum()),
                controller_adjacent_rms_mm=rms_mm(raw[cm], scale_mm) if cm.any() else 0.,
                controller_mass_by_head=a[..., :60].sum(-1).mean(0).detach().cpu().tolist(),
                anchor_mass_by_head=a[..., 60:-1].sum(-1).mean(0).detach().cpu().tolist(),
                null_mass_by_head=a[..., -1].mean(0).detach().cpu().tolist(),
                tool_motion_12d_rms_mm=rms_mm(tool.flatten(-2), scale_mm),
                tool_motion_head_vector_rms_mm=(tool.detach().double().square().sum(-1).mean(0).sqrt()*scale_mm).cpu().tolist(),
                tool_motion_signed_mean_by_head_mm=(tool.detach().mean(0)*scale_mm).cpu().tolist(),
                anchor_control_displacement_rms_mm=rms_mm(step['cumulative_d'][0, ids], scale_mm),
                actual_fps_motion_rms_mm=rms_mm(pred['xyz'][0, h, trace['field'].anchor_indices[0, ids]]-
                                               x['gs_xyz'][0, trace['field'].anchor_indices[0, ids]], scale_mm)))
    return rows


def audit(model, batch, processor, renderer, *, full_components=True):
    model.eval(); before = digest(model.state_dict()); scale_mm = processor.state_dict()['scale']*1000
    with observe(model) as captured:
        pred, trace = model(batch['inputs'], return_trace=True)
    gs, anchors = fixed_regions(batch, trace['field'])
    losses, time_losses = components(pred, batch, processor, renderer, with_time=True)
    named = list(model.named_parameters())
    records = [('output', 0, pred['xyz'])]
    for h, step in enumerate(trace['steps']):
        for name, value in [('state', step['state']), ('alpha', step['alpha']),
                            ('tool_motion', step['tool_motion']),
                            ('anchor_features', captured['anchor_features'][h]),
                            ('motion_input', captured['motion_input'][h]), ('anchor_d', step['cumulative_d'])]:
            records.append((name, h, value))
    targets = [v for _, _, v in records]+[v for _, v in named]
    if not all(v.requires_grad for v in targets):
        raise RuntimeError('Expected graph tensor lacks requires_grad')
    tensors = {}; param_grads = {}
    terms = LOSSES if full_components else ('joint',)
    for loss in terms:
        gg = torch.autograd.grad(losses[loss], targets, retain_graph=loss != terms[-1], allow_unused=True)
        gg = [None if g is None else finite_gradient(g).detach().cpu() for g in gg]
        tensors[loss] = {(name, h): g for (name, h, _), g in zip(records, gg)}
        param_grads[loss] = gg[len(records):]
    if full_components:
        for key in tensors['joint']:
            parts = [tensors[n][key] for n in LOSSES[:-1]]; joint = tensors['joint'][key]
            if joint is not None:
                total = sum(g for g in parts if g is not None)
                torch.testing.assert_close(total, joint, atol=3e-6, rtol=3e-5)
    position = []; intermediate = []; raw = {}
    gs_cpu = {k: v.cpu() for k, v in gs.items()}
    anchor_cpu = {k: v.cpu() for k, v in anchors.items()}
    delta = (pred['xyz'][0]-batch['inputs']['gs_xyz'][0]).detach().cpu()
    for loss in terms:
        out = tensors[loss][('output', 0)]
        if out is None:
            raise RuntimeError('Loss disconnected from prediction output')
        raw[loss+'_direct_Hmean'] = out.numpy()
        raw[loss+'_total_AR_Hmean'] = np.stack([tensors[loss][('state', h)].numpy() for h in range(10)], 1)
    raw.update(gs_indices=gs_cpu['all'].numpy(), near_indices=gs_cpu['near'].numpy(), far_indices=gs_cpu['far'].numpy())
    for h in range(10):
        for scope in ('direct_Hmean', 'direct_single_horizon', 'total_AR_Hmean', 'future_AR_Hmean'):
            gg = {}
            for loss in terms:
                direct = tensors[loss][('output', 0)][0, h]
                total = tensors[loss][('state', h)]
                if total is None:
                    raise RuntimeError('Internal AR state disconnected')
                gg[loss] = {'direct_Hmean': direct, 'direct_single_horizon': direct*10,
                            'total_AR_Hmean': total[0], 'future_AR_Hmean': total[0]-direct}[scope]
            for region, ids in gs_cpu.items():
                sub = {k: v[ids] for k, v in gg.items()}
                cs = cosines(sub)
                for loss, g in sub.items():
                    position.append(dict(horizon=h+1, scope=scope, region=region, loss=loss,
                        valid_gs=len(ids), motion_rms_mm=rms_mm(delta[h, ids], scale_mm),
                        standalone_current_loss=time_losses[loss][h],
                        **stats(g), **cs))
        for key in ('alpha', 'tool_motion', 'anchor_features', 'motion_input', 'anchor_d'):
            for region, ids in anchor_cpu.items():
                sub = {loss: (None if tensors[loss][(key, h)] is None else tensors[loss][(key, h)][0, ids]) for loss in terms}
                for loss, g in sub.items():
                    row = dict(horizon=h+1, tensor=key, region=region, loss=loss, **stats(g), **cosines(sub))
                    if g is not None and key == 'alpha':
                        row.update(controller_gradient_l2=float(g[..., :60].double().norm()),
                                   null_gradient_l2=float(g[..., -1].double().norm()),
                                   anchor_key_gradient_l2=float(g[..., 60:-1].double().norm()),
                                   controller_gradient_l2_by_head=g[..., :60].double().square().sum((0,2)).sqrt().tolist(),
                                   null_gradient_l2_by_head=g[..., -1].double().square().sum(0).sqrt().tolist())
                    if g is not None and key == 'motion_input':
                        row.update(feature_gradient_l2=float(g[..., :128].double().norm()),
                                   tool_12d_gradient_l2=float(g[..., 128:140].double().norm()),
                                   history_gradient_l2=float(g[..., 140:143].double().norm()),
                                   previous_d_gradient_l2=float(g[..., 143:146].double().norm()))
                    intermediate.append(row)
    angles = []
    if full_components:
        for h in (0, 4, 9):
            for a, b in [('cd', 'rgb'), ('cd', 'alpha_weighted'), ('rgb', 'alpha_weighted')]:
                for region, ids in gs_cpu.items():
                    angles.append(dict(horizon=h+1, region=region, pair=a+'__'+b,
                        **point_angles(tensors[a][('output', 0)][0, h, ids], tensors[b][('output', 0)][0, h, ids])))
    signals = signal_rows(batch, pred, trace, anchors, scale_mm)
    parameter = parameter_summary(named, param_grads)
    # Local analytic derivative of frozen Motion MLP, evaluated at saved inputs.
    # It measures sensitivity in this network input basis, not actual optimizer effects.
    jacobians = []
    for h in (0, 4, 9):
        u = captured['motion_input'][h].detach().requires_grad_()
        d = model.motion(u)
        jac = torch.stack([torch.autograd.grad(d[..., j].sum(), u, retain_graph=j < 2)[0] for j in range(3)], -2)
        for region, ids in anchors.items():
            j = jac[0, ids].detach().double()
            jacobians.append(dict(horizon=h+1, region=region,
                tool_jacobian_frobenius_rms=float(j[..., 128:140].square().sum((-1,-2)).mean().sqrt()),
                feature_jacobian_frobenius_rms=float(j[..., :128].square().sum((-1,-2)).mean().sqrt()),
                previous_d_jacobian_frobenius_rms=float(j[..., 143:146].square().sum((-1,-2)).mean().sqrt()),
                tool_path_local_linear_delta_rms_mm=rms_mm(torch.einsum('aij,aj->ai',j[...,128:140],u[0,ids,128:140].detach().double()),scale_mm)))
    detached = {k: v.detach() for k, v in pred.items()}
    result = dict(losses=values(losses), per_horizon_losses=time_losses, loss_gradients=position, parameter_gradients=parameter,
                  intermediate_gradients=intermediate, angle_distributions=angles,
                  attention_signal=signals, motion_decoder_jacobian=jacobians,
                  state_sha256_before=before, state_sha256_after=same_state(model, before),
                  sigma_geo_mm=float(trace['field'].bandwidth[0])*scale_mm,
                  gradient_unit='derivative per original normalized tensor coordinate; alpha weighted by 0.1',
                  direct_single_horizon_note='10 * direct H-mean, valid because all 10 target times have observations',
                  future_AR_note='total internal-state gradient minus direct stacked-output gradient, both H-mean')
    return result, detached, raw


@torch.no_grad()
def ablations(model, batch, processor, renderer, normal):
    before = digest(model.state_dict()); scale = processor.state_dict()['scale']*1000
    out = {}
    normal_loss = values(components(normal,batch,processor,renderer))
    for mode in ('normal', 'freeze_action', 'zero_tactile_pressure', 'zero_explicit_tool_motion'):
        if mode == 'normal':
            p = normal
        elif mode == 'zero_explicit_tool_motion':
            with observe(model, zero_tool=True):
                p = model(batch['inputs'])
        else:
            p = model(altered_inputs(batch['inputs'], mode))
        keep = batch['inputs']['gs_mask'][0]
        xyz = p['xyz'][0, :, keep]
        diff = xyz-normal['xyz'][0, :, keep]
        loss = values(components(p,batch,processor,renderer))
        out[mode] = dict(losses=loss, delta_loss={k:loss[k]-normal_loss[k] for k in loss},
            output_change_rms_mm=[rms_mm(z,scale) for z in diff],
            output_motion_rms_mm=[rms_mm(z-batch['inputs']['gs_xyz'][0,keep],scale) for z in xyz],
            signed_mean_displacement_mm=((xyz-batch['inputs']['gs_xyz'][0,keep]).mean(1)*scale).cpu().tolist())
    same_state(model, before)
    return out


def finite_differences(pred, batch, processor, renderer, *, pilot=False):
    """H10 standalone loss; full H-mean loss delta is standalone delta / 10."""
    from .core import components
    p, b = single_horizon(pred, batch, 9)
    p = {k: v.detach().clone() for k,v in p.items()}; p['xyz'].requires_grad_()
    renderer = TargetCache(renderer.renderer if hasattr(renderer,'renderer') else renderer)
    baseline = components(p,b,processor,renderer)
    grads = {k: torch.autograd.grad(baseline[k],p['xyz'],retain_graph=k!=LOSSES[-1])[0].detach() for k in LOSSES}
    base = values(baseline); scale_mm=processor.state_dict()['scale']*1000; mask=b['inputs']['gs_mask'][:,None]
    directions = {'negative_joint_gradient': normalized_direction(-grads['joint'],mask)}
    if not pilot:
        x=batch['inputs'];cp=x['controller_points'].reshape(1,10,60,3)
        valid=x['controller_mask'].reshape(1,10,60)&x['operation_hands'][:,None].repeat_interleave(30,-1)
        valid=valid[:,0]&valid[:,-1]
        if not valid.any():raise ValueError('No legal controller cumulative direction')
        movement=(cp[:,-1]-cp[:,0])[valid].mean(0)
        movement=movement.expand_as(p['xyz'])
        directions['positive_controller_cumulative']=normalized_direction(movement,mask)
        directions['negative_controller_cumulative']=normalized_direction(-movement,mask)
        gen=torch.Generator(device='cpu').manual_seed(1729)
        random=torch.randn(p['xyz'].shape,generator=gen,dtype=p['xyz'].dtype).to(p['xyz'].device)
        directions['fixed_random']=normalized_direction(random,mask)
    rows=[]
    amplitudes=(.001,.01) if pilot else (.001,.01,.1,.5,1.)
    for name,direction in directions.items():
        if direction is None:
            rows.append(dict(direction=name,status='UNAVAILABLE_ZERO_DIRECTION'));continue
        derivatives={k:float((grads[k].double()*direction.double()/scale_mm).sum()) for k in LOSSES}
        for amplitude in amplitudes:
            signs=(1,-1) if amplitude in (.001,.01) else (1,)
            evals={};actual={}
            for sign in signs:
                with torch.no_grad():
                    changed=dict(p);changed['xyz']=p['xyz'].detach()+direction*(sign*amplitude/scale_mm)
                    actual[sign]=rms_mm((changed['xyz']-p['xyz'].detach())[mask],scale_mm)
                    evals[sign]=values(components(changed,b,processor,renderer))
            for loss in LOSSES:
                fd=(evals[1][loss]-evals[-1][loss])/(2*amplitude) if -1 in evals else None
                rows.append(dict(direction=name,status='OK',amplitude_mm=amplitude,actual_rms_mm=actual[1],
                    loss=loss,baseline=base[loss],perturbed=evals[1][loss],delta=evals[1][loss]-base[loss],
                    delta_full_Hmean=(evals[1][loss]-base[loss])/10,
                    autograd_directional_derivative_per_mm=derivatives[loss],central_difference_per_mm=fd,
                    central_absolute_error=None if fd is None else abs(fd-derivatives[loss]),
                    central_relative_error=None if fd is None else abs(fd-derivatives[loss])/max(abs(fd),abs(derivatives[loss]),1e-10),
                    negative_perturbed=None if -1 not in evals else evals[-1][loss]))
    return dict(baseline=base,rows=rows,
                controller_direction='mean active valid current-minus-initial controller XYZ at condition t+9, no GT',
                amplitude_definition='RMS Euclidean displacement over valid predicted GS in millimetres',
                no_physical_realizability_claim=True,random_seed=1729)
