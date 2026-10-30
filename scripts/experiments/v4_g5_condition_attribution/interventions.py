"""Temporary, storage-safe input/K/V hooks on the original frozen G5 graph.

Normal runs the original arithmetic. C2 freezes each Normal layer's Alpha both
for Value aggregation and the final Motion Head readout, preventing residual
Key-to-tool12 leakage in a claimed Value-only intervention. C3 changes only
projected Key (including its original pairwise scoring), never Normal Value.
"""
from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F
from scripts.training.model import masked
from scripts.experiments.v4_g5_cross_attention_audit.interventions import (
    intervention as capture_original, replay_step as replay_original, SELECTED, one_frame)

CONTENT_MODES=('T3PressureZero','T3All6Off','CrossA0GeometryOff',
               'CrossHandFusionOff','CrossA0T3Off')
DIRECT_MODES=('DirectHandTokenOff','LocalController6Off','SignedHand12Off')
MODES=('Normal',*CONTENT_MODES,*DIRECT_MODES)
KV_MODES=CONTENT_MODES[:4]
CHANNELS=('EndToEnd','FixedRoutingChangedValue','ChangedKeyNormalValue')


def check_mode(mode):
    if mode not in MODES:raise ValueError('Unknown frozen condition mode: '+str(mode))


def features13(value,mode):
    check_mode(mode)
    if value.shape[-1]!=13:raise ValueError('Expected 7D A0 +6D T3')
    if mode not in CONTENT_MODES or mode=='CrossHandFusionOff':return value
    result=value.clone()
    if mode=='T3PressureZero':result[...,7:9]=0
    elif mode=='T3All6Off':result[...,7:13]=0
    elif mode=='CrossA0GeometryOff':result[...,:7]=0
    else:result.zero_()
    return result


@contextmanager
def content_hooks(encoder,mode,only_step=None):
    """Never modify x, the geometry returned by features(), or Hand Tokens."""
    check_mode(mode);rows=[];handles=[];active={}
    def begin(module,args):
        step=int(args[1]);active['mode']=mode if only_step is None or step==only_step else 'Normal'
        row=dict(step=step,mode=active['mode']);rows.append(row);active['row']=row
    def controller(module,args):
        original=args[0];effective=features13(original,active['mode'])
        active['row'].update(original_features13=original,effective_features13=effective)
        return (effective,*args[1:])
    def hand(module,args,output):
        effective=torch.zeros_like(output) if active['mode']=='CrossHandFusionOff' else output
        active['row'].update(original_hand_projection=output,effective_hand_projection=effective)
        return effective
    def finish(module,args,output):
        active['row'].update(tools=output[0],valid=output[1],cp=output[2],hand_tokens=output[3]['tokens62'][:,:2])
    handles.extend((encoder.register_forward_pre_hook(begin),
        encoder.controller.register_forward_pre_hook(controller),
        encoder.hand_projection.register_forward_hook(hand),encoder.register_forward_hook(finish)))
    try:yield rows
    finally:
        for handle in reversed(handles):handle.remove()


@contextmanager
def direct_hooks(model,mode,only_step=None):
    """Basis changes affect Mapping context AND its signed matrix product."""
    check_mode(mode);rows=[];handles=[];active={}
    def global_pre(module,args,kwargs):
        step=int(args[1]);effective=mode if only_step is None or step==only_step else 'Normal'
        row=dict(step=step,mode=effective,global_kwargs_original=dict(kwargs))
        rows.append(row);active['row']=row
        actual=dict(kwargs)
        if effective=='DirectHandTokenOff':actual['hand_tokens']=torch.zeros_like(kwargs['hand_tokens'])
        row['global_kwargs_effective']=actual
        return args,actual
    def global_mapping(module,args):
        row=active['row'];context=args[0];effective=context
        if row['mode']=='SignedHand12Off':
            effective=context.clone();effective[...,-12:]=0
        row.update(global_context_original=context,global_context_effective=effective)
        return (effective,*args[1:])
    def global_matrix(module,args,output):active['row']['global_matrix']=output
    def global_post(module,args,kwargs,output):
        delta,trace=output;row=active['row']
        row.update(global_output_before=delta,global_basis_before=trace['hand_basis'])
        if row['mode']=='SignedHand12Off':
            basis=torch.zeros_like(trace['hand_basis'])
            matrix=row['global_matrix'].reshape(len(basis),2,3,12)
            contribution=masked(torch.einsum('bhij,bhj->bhi',matrix,basis),trace['hand_valid'])
            trace=dict(trace,hand_basis=basis,hand_contribution=contribution)
            delta=contribution.sum(1)
        row.update(global_output_effective=delta,global_trace_effective=trace)
        return delta,trace
    def local_pre(module,args,kwargs):
        row=active['row'];actual=dict(kwargs);row['local_kwargs_original']=dict(kwargs)
        if row['mode']=='DirectHandTokenOff':actual['hand_tokens']=torch.zeros_like(kwargs['hand_tokens'])
        row['local_kwargs_effective']=actual
        return args,actual
    def local_mapping(module,args):
        context=args[0];row=active['row'];effective=context
        if context.shape[-1]!=287:raise ValueError('Expected original G5 Direct context287')
        if row['mode']=='LocalController6Off':
            effective=context.clone();effective[...,272:278]=0
        row.update(local_context_original=context,local_context_effective=effective)
        return (effective,*args[1:])
    def local_matrix(module,args,output):active['row']['local_matrix']=output
    def local_post(module,args,kwargs,output):
        seed,trace=output;row=active['row'];row['local_basis_before']=trace['anchor_hand_basis18']
        if row['mode']=='LocalController6Off':
            basis=trace['anchor_hand_basis18'].clone();basis[...,12:18]=0
            matrix=row['local_matrix'].reshape(*basis.shape[:-1],3,18)
            allowed=kwargs['anchor_valid'][:,:,None]&kwargs['hand_valid'][:,None]
            seed=masked(torch.einsum('bahij,bahj->bahi',matrix,basis),allowed)
            trace=dict(trace,anchor_hand_basis18=basis)
        row.update(local_basis_effective=trace['anchor_hand_basis18'],local_seed_effective=seed)
        return seed,trace
    handles.extend((model.global_motion.register_forward_pre_hook(global_pre,with_kwargs=True),
        model.global_motion.mapping.register_forward_pre_hook(global_mapping),
        model.global_motion.mapping.register_forward_hook(global_matrix),
        model.global_motion.register_forward_hook(global_post,with_kwargs=True),
        model.direct_motion.register_forward_pre_hook(local_pre,with_kwargs=True),
        model.direct_motion.mapping.register_forward_pre_hook(local_mapping),
        model.direct_motion.mapping.register_forward_hook(local_matrix),
        model.direct_motion.register_forward_hook(local_post,with_kwargs=True)))
    try:yield rows
    finally:
        for handle in reversed(handles):handle.remove()


@contextmanager
def cross_hooks(model,channel='EndToEnd',normal_calls=None,changed_tools=None):
    """Observe REAL Q/K/V and scores; replace only declared projected tensors."""
    if channel not in CHANNELS:raise ValueError('Unknown K/V channel')
    if channel!='EndToEnd' and (normal_calls is None or changed_tools is None):
        raise ValueError('K/V separation requires same-step Normal layer captures and changed Tools')
    rows=[];active={};handles=[]
    try:
        for layer,block in enumerate(model.cross):
            def begin(module,args,kwargs,layer=layer):
                if len(rows)%2!=layer:raise AssertionError('Cross layer order')
                args=list(args)
                if channel=='EndToEnd' and changed_tools is not None:args[1]=changed_tools
                names=('query','tools','anchor_xyz','controller_xyz','sigma','anchor_valid','tool_valid')
                inputs={n:args[i] if i<len(args) else kwargs[n] for i,n in enumerate(names)}
                row=dict(layer=layer,step=len(rows)//2,inputs=inputs,pairwise={})
                active[layer]=row;rows.append(row)
                return tuple(args),kwargs
            handles.append(block.register_forward_pre_hook(begin,with_kwargs=True))
            for name,module in [('q',block.q),('k',block.k),('v',block.v)]:
                def projected(module,args,output,name=name,layer=layer,block=block):
                    row=active[layer];effective=output
                    if ((channel=='FixedRoutingChangedValue' and name=='v') or
                        (channel=='ChangedKeyNormalValue' and name=='k')):
                        tools=masked(changed_tools,row['inputs']['tool_valid'])
                        effective=F.linear(block.memory_norm(tools),module.weight,module.bias)
                    row[name]=effective;row[name+'_before']=output
                    return effective
                handles.append(module.register_forward_hook(projected))
            for head,module in enumerate(block.pairwise):
                def pair(module,args,output,head=head,layer=layer):active[layer]['pairwise'][head]=output
                handles.append(module.register_forward_hook(pair))
            def projection(module,args,layer=layer,block=block):
                row=active[layer];x=row['inputs'];b,a,w=x['query'].shape;n=x['tools'].shape[1]
                q=row['q'].reshape(b,a,4,32);k=row['k'].reshape(b,n,4,32);v=row['v'].reshape(b,n,4,32)
                rel=(masked(x['anchor_xyz'],x['anchor_valid'])[:,:,None]-masked(x['controller_xyz'],x['tool_valid'])[:,None])/x['sigma'][:,None,None,None]
                geometry=torch.cat((rel,rel.square().sum(-1,keepdim=True)),-1);scores=[]
                for head in range(4):
                    qh=q[:,:,head,None].expand(-1,-1,n,-1);kh=k[:,None,:,head].expand(-1,a,-1,-1)
                    score=(qh*kh).sum(-1)/math.sqrt(32)
                    score=score+(geometry*block.geometry_score[head]).sum(-1)
                    scores.append(score+row['pairwise'][head].squeeze(-1))
                allowed=x['anchor_valid'][:,:,None,None]&x['tool_valid'][:,None,None]
                scores=torch.stack(scores,2).masked_fill(~allowed,-torch.inf)
                alpha=torch.cat((scores,scores.new_zeros(b,a,4,1)),-1).softmax(-1)
                original=torch.einsum('bahj,bjhd->bahd',alpha[...,:n],v).reshape(b,a,w)
                torch.testing.assert_close(original,args[0],rtol=0,atol=0)
                used=normal_calls[layer]['alpha_used'] if channel=='FixedRoutingChangedValue' else alpha
                if not torch.isfinite(used).all():raise AssertionError('Nonfinite Attention')
                torch.testing.assert_close(used.sum(-1),torch.ones_like(used[...,0]),rtol=0,atol=2e-6)
                if torch.count_nonzero(used[...,:n].masked_select(~allowed.expand_as(used[...,:n]))):
                    raise AssertionError('Attention Mask leakage')
                update=args[0] if channel!='FixedRoutingChangedValue' else torch.einsum('bahj,bjhd->bahd',used[...,:n],v).reshape(b,a,w)
                row.update(alpha_computed=alpha,alpha_used=used,geometry=geometry,values=v,
                           aggregate=update,original_aggregate=args[0])
                return (update,*args[1:])
            handles.append(block.projection.register_forward_pre_hook(projection))
            def projected_update(module,args,output,layer=layer):active[layer]['projected_update']=output
            handles.append(block.projection.register_forward_hook(projected_update))
            def finish(module,args,kwargs,output,layer=layer):
                row=active[layer];torch.testing.assert_close(row['alpha_computed'],output[1],rtol=0,atol=0)
                row['query_after']=output[0]
                return output[0],row['alpha_used']
            handles.append(block.register_forward_hook(finish,with_kwargs=True))
        yield rows
    finally:
        for handle in reversed(handles):handle.remove()


@contextmanager
def intervention(model,mode='Normal',only_step=None):
    check_mode(mode)
    if only_step is not None and (type(only_step) is not int or not 0<=only_step<model.config['horizon']):
        raise ValueError('Invalid only_step')
    if model.use_propagation or len(model.cross)!=2:raise ValueError('Requires original G5-Direct')
    with capture_original(model,'Normal') as rows:
        with content_hooks(model.conditions,mode,only_step) as contents:
            with direct_hooks(model,mode,only_step) as direct:
                with cross_hooks(model) as calls:
                    try:yield rows
                    finally:
                        for k,row in enumerate(rows):
                            row['capture_is_normal']=mode=='Normal'
                            # Preserve the original exception if a model call
                            # fails before all later hooks have executed.
                            if k<len(contents):row['condition_content']=contents[k]
                            if k<len(direct):row['direct_intervention']=direct[k]
                            row['condition_cross_calls']=calls[2*k:2*k+2]


def replay_step(model,inputs,normal_trace,captures,step,mode='Normal',channel='EndToEnd'):
    check_mode(mode)
    if channel not in CHANNELS or (channel!='EndToEnd' and mode not in KV_MODES):
        raise ValueError('K/V separation restricted to four predefined Cross modes')
    if not all(r.get('capture_is_normal') for r in captures):raise ValueError('Requires Normal history')
    with content_hooks(model.conditions,mode) as content:
        tools,valid,cp,condition_trace=model.conditions(inputs,step)
    reference=captures[step]['condition_cross_calls']
    with direct_hooks(model,mode) as direct:
        with cross_hooks(model,channel,reference,tools if mode in CONTENT_MODES else None) as calls:
            result=replay_original(model,inputs,normal_trace,captures,step,'Normal')
    result.update(condition_cross_calls=calls,condition_content=content[0],direct_intervention=direct[0],
                  channel=channel,mode=mode)
    # All separations preserve Normal physical positions and Boolean masks.
    for call,ref in zip(calls,reference):
        for key in ('anchor_xyz','controller_xyz','sigma','anchor_valid','tool_valid'):
            torch.testing.assert_close(call['inputs'][key],ref['inputs'][key],rtol=0,atol=0)
    return result
