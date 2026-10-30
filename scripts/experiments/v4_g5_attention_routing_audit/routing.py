"""Seven fixed Value-only routes. No trainable parameter or global RNG use."""
from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F
from scripts.training.model import masked

MODES=('Normal','UniformRaw','UniformGlobalMass','UniformPerHandMass',
       'RBFPerHandMass','LearnedGeometryPerHandMass','PermuteWithinHand')
MASS_MATCHED=MODES[3:]
WEIGHT_ATOL=2e-6  # FP32 reductions; never renormalize away the required masses.
TOP_K=5


def conditional(logits,allowed):
    """Intra-hand softmax; empty sets return exact zero, without a new Null."""
    allowed=allowed.expand_as(logits)
    has=allowed.any(-1,keepdim=True)
    scores=logits.masked_fill(~allowed,-torch.inf)
    scores=torch.where(has,scores,torch.zeros_like(scores))
    return torch.where(allowed,scores.softmax(-1),torch.zeros_like(scores))


def permutation(count,layer,head,hand,device):
    """Frozen integer-hash shuffle of valid ranks; no data/result-dependent seed.

    Use the same bijection for all Anchors of this layer/head/hand. n<=1 is the
    unavoidable identity. If a hash sort happens to be identity, rotate once.
    """
    ranks=torch.arange(count,device=device,dtype=torch.int64)
    if count<=1:return ranks
    salt=(layer+1)*1000003+(head+1)*2654435761+(hand+1)*2246822519
    keys=((ranks+1)*1103515245+salt)%2147483647
    order=keys.argsort(stable=True)
    return order.roll(1) if torch.equal(order,ranks) else order


def replace_weights(alpha,valid,anchor_valid,geometry,geometry_logits,mode,layer):
    """alpha [B,A,heads,61], geometry [B,A,60,4], last slot is original Null."""
    if mode not in MODES:raise ValueError('Unknown routing mode: '+str(mode))
    if alpha.shape[-1]!=61 or valid.shape[-1]!=60:raise ValueError('Expected original60 Controllers +Null')
    allowed=anchor_valid[:,:,None,None]&valid[:,None,None,:]
    count=allowed.sum(-1,keepdim=True).to(alpha.dtype)
    if mode=='Normal':return alpha
    if mode=='UniformRaw':
        return torch.cat((allowed.expand_as(alpha[...,:60]).to(alpha.dtype)/(count+1),
                          (count+1).reciprocal().expand_as(alpha[...,-1:])), -1)
    routed=torch.zeros_like(alpha)
    routed[...,-1]=alpha[...,-1]  # Preserve the actual learned Null, not 1-sum approximations.
    if mode=='UniformGlobalMass':
        mass=alpha[...,:60].sum(-1,keepdim=True)
        routed[...,:60]=mass*allowed.to(alpha.dtype)/count.clamp_min(1)
    elif mode=='PermuteWithinHand':
        for b in range(len(alpha)):
            for hand in range(2):
                ids=valid[b,hand*30:(hand+1)*30].nonzero().flatten()+hand*30
                for head in range(alpha.shape[2]):
                    order=permutation(len(ids),layer,head,hand,alpha.device)
                    values=alpha[b,:,head,:].index_select(-1,ids[order])
                    routed[b,:,head,:].index_copy_(-1,ids,values)
    else:
        for hand in range(2):
            sl=slice(hand*30,(hand+1)*30)
            mass=alpha[...,sl].sum(-1,keepdim=True)
            if mode=='UniformPerHandMass':logits=torch.zeros_like(alpha[...,sl])
            elif mode=='RBFPerHandMass':logits=(-.5*geometry[...,sl,3])[:,:,None,:].expand_as(alpha[...,sl])
            else:logits=geometry_logits[...,sl]
            routed[...,sl]=mass*conditional(logits,allowed[...,sl])
    return routed


def validate(alpha,route,valid,anchor_valid,mode):
    """Check every Anchor/head, including null-only padding; return error tensors."""
    allowed=(anchor_valid[:,:,None,None]&valid[:,None,None,:]).expand_as(alpha[...,:60])
    if not torch.isfinite(route).all() or (route<0).any():raise FloatingPointError('Invalid routing weights')
    if torch.count_nonzero(route[...,:60].masked_select(~allowed)):raise AssertionError('Masked Controller routing leak')
    total_error=(route.sum(-1)-1).abs().max()
    if float(total_error)>WEIGHT_ATOL:raise AssertionError('Routing rows do not sum to one')
    empty=~allowed.any(-1)
    if not torch.equal(route[...,-1][empty],torch.ones_like(route[...,-1][empty])):
        raise AssertionError('Empty/padded row must select only Null')
    null_error=(route[...,-1]-alpha[...,-1]).abs().max()
    hand_error=torch.stack([(route[...,h*30:(h+1)*30].sum(-1)-alpha[...,h*30:(h+1)*30].sum(-1)).abs().max()
                            for h in range(2)]).max()
    if mode!='UniformRaw' and not torch.equal(route[...,-1],alpha[...,-1]):raise AssertionError('Learned Null mass changed')
    if mode in MASS_MATCHED and float(hand_error)>WEIGHT_ATOL:raise AssertionError('Per-hand mass changed')
    if mode=='UniformGlobalMass' and float((route[...,:60].sum(-1)-alpha[...,:60].sum(-1)).abs().max())>WEIGHT_ATOL:
        raise AssertionError('Global Controller mass changed')
    return dict(row_sum_error=total_error,null_mass_error=null_error,hand_mass_error=hand_error)


@contextmanager
def routing_blocks(model,mode='Normal',only_step=None):
    """Inject only the input of the original W_O projection, via temporary hooks.

    Q/K/V and pairwise outputs are captured from the REAL original block call.
    Reconstruct its scores in the same arithmetic order and assert equality to
    the alpha returned by that block. The returned alpha is NEVER replaced, so
    tool_motion12/head_mass4 still consume alpha_learned under the current Query.
    No original forward or parameter is assigned/modified.
    """
    if mode not in MODES:raise ValueError('Unknown routing mode')
    if getattr(model,'use_propagation',None) is not False or len(model.cross)!=2:
        raise ValueError('Requires original two-layer G5-Direct')
    if only_step is not None and (type(only_step) is not int or not 0<=only_step<model.config['horizon']):
        raise ValueError('Invalid only_step')
    records=[];active={};handles=[]
    try:
        for layer,block in enumerate(model.cross):
            def begin(module,args,kwargs,layer=layer):
                if len(records)%2!=layer:raise AssertionError('Unexpected Cross call order')
                names=('query','tools','anchor_xyz','controller_xyz','sigma','anchor_valid','tool_valid')
                values={name:(args[i] if i<len(args) else kwargs[name]) for i,name in enumerate(names)}
                step=len(records)//2
                entry=dict(step=step,layer=layer,mode=mode if only_step is None or step==only_step else 'Normal',
                           args=args,kwargs=dict(kwargs),inputs=values,pairwise={})
                records.append(entry);active[layer]=entry
            handles.append(block.register_forward_pre_hook(begin,with_kwargs=True))
            for name,module in [('q',block.q),('k',block.k),('v',block.v)]:
                def capture(module,args,output,name=name,layer=layer):active[layer][name]=output
                handles.append(module.register_forward_hook(capture))
            for head,module in enumerate(block.pairwise):
                def capture_pair(module,args,output,head=head,layer=layer):active[layer]['pairwise'][head]=output
                handles.append(module.register_forward_hook(capture_pair))
            def project_pre(module,args,layer=layer,block=block):
                row=active[layer];x=row['inputs'];b,a,width=x['query'].shape;n=x['tools'].shape[1]
                q=row['q'].reshape(b,a,block.heads,block.dim)
                k=row['k'].reshape(b,n,block.heads,block.dim)
                v=row['v'].reshape(b,n,block.heads,block.dim)
                relative=(masked(x['anchor_xyz'],x['anchor_valid'])[:,:,None]-masked(x['controller_xyz'],x['tool_valid'])[:,None])/x['sigma'][:,None,None,None]
                geometry=torch.cat((relative,relative.square().sum(-1,keepdim=True)),-1)
                scores=[];geo=[]
                for head in range(block.heads):
                    qh=q[:,:,head,None].expand(-1,-1,n,-1);kh=k[:,None,:,head].expand(-1,a,-1,-1)
                    geometric=(geometry*block.geometry_score[head]).sum(-1);geo.append(geometric)
                    score=(qh*kh).sum(-1)/math.sqrt(block.dim)
                    score=score+geometric
                    scores.append(score+row['pairwise'][head].squeeze(-1))
                scores=torch.stack(scores,2)
                allowed=x['anchor_valid'][:,:,None,None]&x['tool_valid'][:,None,None]
                scores=scores.masked_fill(~allowed,-torch.inf)
                alpha=torch.cat((scores,scores.new_zeros(b,a,block.heads,1)),-1).softmax(-1)
                original=torch.einsum('bahj,bjhd->bahd',alpha[...,:n],v).reshape(b,a,width)
                torch.testing.assert_close(original,args[0],rtol=0,atol=0)
                route=replace_weights(alpha,x['tool_valid'],x['anchor_valid'],geometry,torch.stack(geo,2),row['mode'],layer)
                checks=validate(alpha,route,x['tool_valid'],x['anchor_valid'],row['mode'])
                update=args[0] if row['mode']=='Normal' else torch.einsum('bahj,bjhd->bahd',route[...,:n],v).reshape(b,a,width)
                row.update(alpha_learned=alpha,alpha_route=route,geometry=geometry,geometry_logits=torch.stack(geo,2),
                    values=v,learned_update=args[0],route_update=update,
                    learned_projected_update=F.linear(args[0],module.weight,module.bias),checks=checks)
                return (update,*args[1:])
            handles.append(block.projection.register_forward_pre_hook(project_pre))
            def project_post(module,args,output,layer=layer):active[layer]['route_projected_update']=output
            handles.append(block.projection.register_forward_hook(project_post))
            def finish(module,args,kwargs,output,layer=layer):
                row=active[layer]
                torch.testing.assert_close(row['alpha_learned'],output[1],rtol=0,atol=0)
                row['query_after']=output[0]
            handles.append(block.register_forward_hook(finish,with_kwargs=True))
        yield records
    finally:
        for handle in reversed(handles):handle.remove()
