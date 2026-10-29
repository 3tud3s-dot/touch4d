"""Per-anchor, per-hand 18D signed seeds. No anchor-to-anchor operations."""
import torch
from torch import nn
from scripts.training.model import masked


def masked_softmax(scores,valid):
    """Normalize only allowed entries; an empty row is exactly zero, no null."""
    allowed=valid.any(-1,keepdim=True)
    logits=scores.masked_fill(~valid,-torch.inf)
    logits=torch.where(allowed,logits,torch.zeros_like(logits))
    return torch.where(valid,logits.softmax(-1),torch.zeros_like(logits))


def geometry_weights(anchor_xyz,cp,valid,anchor_valid,sigma):
    """[B,A,2,30] Gaussian weights; no threshold and no null/attention sharing."""
    anchor=masked(anchor_xyz,anchor_valid)
    points=torch.where(valid[...,None],cp,torch.zeros_like(cp))
    relative=(anchor[:,:,None,None]-points[:,None])/sigma[:,None,None,None,None]
    scores=-.5*relative.square().sum(-1)
    allowed=anchor_valid[:,:,None,None]&valid[:,None]
    if not torch.isfinite(torch.where(allowed,scores,0.)).all():
        raise FloatingPointError('Nonfinite valid anchor-controller distance')
    weights=masked_softmax(scores,allowed)
    return weights,relative


class AnchorSignedDirect(nn.Module):
    """Shared 287->128->54 mapping, evaluated independently at each [a,hand]."""
    def __init__(self,width=128):
        super().__init__()
        self.mapping=nn.Sequential(nn.LayerNorm(2*width+31),nn.Linear(2*width+31,width),
                                   nn.GELU(),nn.Linear(width,54))
        nn.init.zeros_(self.mapping[-1].weight);nn.init.zeros_(self.mapping[-1].bias)

    def forward(self,x,step,*,query,anchor_xyz,anchor_valid,hand_tokens,hand_basis12,hand_valid,
                cp,cp_valid,previous_cp,previous_valid,anchor_history,previous_d,global_hand_contribution,sigma):
        b,a,width=query.shape
        if sigma.shape!=(b,) or not torch.isfinite(sigma).all() or not (sigma>0).all():
            raise ValueError('Expected positive finite initial bandwidth [B]')
        allowed=anchor_valid[:,:,None]&hand_valid[:,None]
        current=cp.reshape(b,2,30,3)
        now_valid=cp_valid.reshape(b,2,30)&hand_valid[...,None]&torch.isfinite(current).all(-1)
        current=torch.where(now_valid[...,None],current,torch.zeros_like(current))
        initial=x['controller_points'][:,0]
        initial_state=x['action_state'][:,0]
        initial_hand=(x['action_valid'][:,0]&x['operation_hands']&torch.isfinite(initial_state).all(-1))
        valid0=(x['controller_mask'][:,0]&initial_hand[...,None]&torch.isfinite(initial).all(-1))
        initial=torch.where(valid0[...,None],initial,torch.zeros_like(initial))
        endpoints=now_valid&valid0
        cumulative=torch.where(endpoints[...,None],current-initial,torch.zeros_like(current))
        beta_a,_=geometry_weights(anchor_xyz,current,endpoints,anchor_valid,sigma)
        r_a=torch.einsum('bahj,bhjd->bahd',beta_a,cumulative)
        if step:
            past_state=x['action_state'][:,step-1]
            past_hand=x['action_valid'][:,step-1]&x['operation_hands']&torch.isfinite(past_state).all(-1)
            previous=previous_cp.reshape(b,2,30,3)
            past_valid=previous_valid.reshape(b,2,30)&past_hand[...,None]&torch.isfinite(previous).all(-1)
            previous=torch.where(past_valid[...,None],previous,torch.zeros_like(previous))
            adjacent_valid=now_valid&past_valid
            adjacent=torch.where(adjacent_valid[...,None],current-previous,torch.zeros_like(current))
            beta_delta,_=geometry_weights(anchor_xyz,current,adjacent_valid,anchor_valid,sigma)
            r_delta=torch.einsum('bahj,bhjd->bahd',beta_delta,adjacent)
        else:
            beta_delta=torch.zeros_like(beta_a);r_delta=torch.zeros_like(r_a)
        basis=torch.cat((hand_basis12[:,None].expand(-1,a,-1,-1),r_a,r_delta),-1)
        basis=masked(basis,allowed)
        beta_now,relative=geometry_weights(anchor_xyz,current,now_valid,anchor_valid,sigma)
        geom=torch.cat(((beta_now[...,None]*relative).sum(-2),
                        (beta_now*torch.linalg.vector_norm(relative,dim=-1)).sum(-1,keepdim=True)),-1)
        geom=masked(geom,allowed&now_valid.any(-1)[:,None])
        context=torch.cat((masked(query,anchor_valid)[:,:,None].expand(-1,-1,2,-1),
            masked(hand_tokens,hand_valid)[:,None].expand(-1,a,-1,-1),geom,basis,
            masked(anchor_history,anchor_valid)[:,:,None].expand(-1,-1,2,-1),
            masked(previous_d,anchor_valid)[:,:,None].expand(-1,-1,2,-1),
            masked(global_hand_contribution,hand_valid)[:,None].expand(-1,a,-1,-1)),-1)
        context=masked(context,allowed)
        matrix=self.mapping(context).reshape(b,a,2,3,18)
        seed=masked(torch.einsum('bahij,bahj->bahi',matrix,basis),allowed)
        return seed,dict(anchor_hand_basis18=basis,controller_cumulative_weights=beta_a,
                         controller_adjacent_weights=beta_delta,controller_geometry=geom)
