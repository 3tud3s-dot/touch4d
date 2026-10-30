"""Routing diagnostics on valid Anchors; no contact interpretation or new gates."""
import torch
from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.experiments.v4_g5_cross_attention_audit.diagnostics import anchor_regions
from .routing import TOP_K
from scripts.experiments.v4_g5_motion_attribution.interventions import branch_parts,combine
from scripts.experiments.v4_g5_motion_attribution.metrics import vector_stats


def entropy(p):
    safe=torch.where(p>0,p,torch.ones_like(p))
    return -(p*safe.log()).sum(-1)


def js_divergence(a,b):
    mean=(a+b)*.5
    logm=torch.where(mean>0,mean,torch.ones_like(mean)).log()
    loga=torch.where(a>0,a,torch.ones_like(a)).log()
    logb=torch.where(b>0,b,torch.ones_like(b)).log()
    return .5*((a*(loga-logm)).sum(-1)+(b*(logb-logm)).sum(-1))


def route_rows(inputs,trace,captures,normal_trace,normal_captures,scale,scope,mode,history):
    """Full-AR: all H globally, spatial quartiles H1/5/10/15; A: all supplied.

    G5 Local and projected W_O metrics are joint-Head quantities repeated beside
    each Head's routing stats, NOT per-Head causal contributions. Top-k uses
    stable node-ID tie breaking; zero Controller mass is undefined.
    """
    if len(inputs['gs_xyz'])!=1:raise ValueError('Diagnostic statistics require fixed Batch1')
    regions=anchor_regions(inputs,trace['field']);rows=[];checks=[]
    from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions
    gsregions=fixed_regions(inputs)
    for k,raw in enumerate(captures):
        step=raw['absolute_step'] if 'absolute_step' in raw else k
        s=trace['steps'][step];ns=normal_trace['steps'][step]
        local=interpolate_anchor_displacements(s['local_correction']-ns['local_correction'],trace['field'],inputs['gs_mask'])[0]*scale*1000
        # Actual GS masks/regions are not material correspondence in future GT.
        for call in raw['routing_calls']:
            layer=call['layer'];normal=normal_captures[step]['routing_calls'][layer]
            a=call['alpha_learned'][0];b=call['alpha_route'][0]
            valid=call['inputs']['tool_valid'][0]
            distances=call['geometry'][0,...,3].sqrt()*call['inputs']['sigma'][0]*scale*1000
            mass_a=a[...,:60].sum(-1);mass_b=b[...,:60].sum(-1)
            distance_a=(a[...,:60]*distances[:,None]).sum(-1)
            distance_b=(b[...,:60]*distances[:,None]).sum(-1)
            conditional_a=distance_a/torch.where(mass_a>0,mass_a,torch.ones_like(mass_a))
            conditional_b=distance_b/torch.where(mass_b>0,mass_b,torch.ones_like(mass_b))
            eff_k=min(TOP_K,int(valid.sum()))
            defined=(mass_a>0)&(mass_b>0)
            if eff_k:
                ia=a[...,:60].masked_fill(~valid[None,None],-torch.inf).argsort(dim=-1,descending=True,stable=True)[...,:eff_k]
                ib=b[...,:60].masked_fill(~valid[None,None],-torch.inf).argsort(dim=-1,descending=True,stable=True)[...,:eff_k]
                overlap=(ia[...,None]==ib[...,None,:]).any(-1).sum(-1).to(a.dtype)/eff_k
            else:overlap=torch.zeros_like(mass_a)
            base_fields={
                'null_learned':a[...,-1],'null_route':b[...,-1],
                'left_mass_learned':a[...,:30].sum(-1),'left_mass_route':b[...,:30].sum(-1),
                'right_mass_learned':a[...,30:60].sum(-1),'right_mass_route':b[...,30:60].sum(-1),
                'entropy_learned_nats':entropy(a),'entropy_route_nats':entropy(b),
                'total_variation':.5*(a-b).abs().sum(-1),'js_divergence_nats':js_divergence(a,b),
                'route_vs_Normal_total_variation':.5*(b-normal['alpha_learned'][0]).abs().sum(-1),
                'learned_vs_Normal_total_variation':.5*(a-normal['alpha_learned'][0]).abs().sum(-1),
                'weighted_controller_distance_learned_mm':distance_a,'weighted_controller_distance_route_mm':distance_b}
            diff=(call['route_update']-call['learned_update'])[0].reshape(len(a),4,-1)
            projected=call['route_projected_update'][0]-call['learned_projected_update'][0]
            checks.append(dict(**scope,mode=mode,history=history,horizon=step+1,layer=layer+1,
                **{key:float(value) for key,value in call['checks'].items()},
                learned_alpha_returned_bit_exact=True,learned_value_aggregate_reconstructed_bit_exact=True))
            for region,ids in regions.items():
                if not len(ids):continue
                if history=='full_AR' and region!='all' and step+1 not in (1,5,10,15):continue
                field_means=torch.stack([value[ids].mean(0) for value in base_fields.values()]).cpu().tolist()
                per_head=(diff[ids].square().mean((0,2)).sqrt()).cpu().tolist()
                projected_rms=float(projected[ids].square().mean().sqrt())
                gids=gsregions[region];local_rms=float(local[gids].square().sum(-1).mean().sqrt()) if len(gids) else None
                for head in range(4):
                    select=defined[ids,head]
                    overlap_value=float(overlap[ids,head][select].mean()) if select.any() else None
                    ca=mass_a[ids,head]>0;cb=mass_b[ids,head]>0
                    row=dict(**scope,mode=mode,history=history,horizon=step+1,layer=layer+1,head=head,region=region,
                        anchors=len(ids),valid_controllers=int(valid.sum()),top_k_effective=eff_k,
                        top_k_agreement=overlap_value,top_k_defined_anchors=int(select.sum()),
                        conditional_distance_learned_mm=float(conditional_a[ids,head][ca].mean()) if ca.any() else None,
                        conditional_distance_route_mm=float(conditional_b[ids,head][cb].mean()) if cb.any() else None,
                        value_aggregate_change_feature_rms=per_head[head],
                        projected_value_change_feature_rms_joint_heads=projected_rms,
                        local_branch_change_gs_rms_mm_joint_heads=local_rms)
                    row.update({name:field_means[i][head] for i,name in enumerate(base_fields)})
                    rows.append(row)
    return rows,checks


def capture_routing_arrays(arrays,mode,captures):
    for step,raw in enumerate(captures,1):
        if step not in (1,5,10,15):continue
        for call in raw['routing_calls']:
            prefix=f"{mode}_h{step:02}_l{call['layer']+1}"
            arrays[prefix+'_learned']=call['alpha_learned'][0].cpu().numpy()
            arrays[prefix+'_route']=call['alpha_route'][0].cpu().numpy()
            arrays[prefix+'_distance_normalized']=call['geometry'][0,...,3].sqrt().cpu().numpy()
            arrays[prefix+'_controller_valid']=call['inputs']['tool_valid'][0].cpu().numpy()


def branch_rows(inputs,trace,normal_trace,scale,scope,mode):
    """Branch magnitude AND change under each mode's own AR history, not causality.

    Distinguish Anchor control variables from their actual fixed-RBF GS fields.
    No attempt to add branch RMS or allocate a nonlinear CD improvement to heads.
    """
    from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions
    field=trace['field'];factor=scale*1000;rows=[]
    for name in ('anchor_indices','anchor_mask','weights','bandwidth'):
        torch.testing.assert_close(getattr(field,name),getattr(normal_trace['field'],name),rtol=0,atol=0)
    groups={'anchor_control':anchor_regions(inputs,field),'gs_actual':fixed_regions(inputs)}
    for h,(s,ns) in enumerate(zip(trace['steps'],normal_trace['steps']),1):
        p=branch_parts(s,field);n=branch_parts(ns,field)
        p['final']=combine(p,field);n['final']=combine(n,field)
        torch.testing.assert_close(p['final'],s['cumulative_d'],rtol=0,atol=0)
        for space in groups:
            current=p if space=='anchor_control' else {k:interpolate_anchor_displacements(v,field,inputs['gs_mask']) for k,v in p.items()}
            normal=n if space=='anchor_control' else {k:interpolate_anchor_displacements(v,field,inputs['gs_mask']) for k,v in n.items()}
            for name,value in current.items():
                for region,ids in groups[space].items():
                    if not len(ids):continue
                    row=dict(**scope,mode=mode,history='full_AR',horizon=h,space=space,region=region,branch=name,
                        **vector_stats(value[0,ids]*factor))
                    row.update({'change_'+k:v for k,v in vector_stats((value-normal[name])[0,ids]*factor).items()})
                    rows.append(row)
    return rows
