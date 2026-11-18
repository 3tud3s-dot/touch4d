"""Signed global/local motion mappings retained without numerical changes."""
import torch
from torch import nn
from .model import masked


def _mean_valid_displacement(current, reference, valid):
    """[B,2,30,3] pairwise endpoint-valid, within-hand mean displacement."""
    delta = torch.where(valid[..., None], current - reference, torch.zeros_like(current))
    count = valid.sum(-1, keepdim=True).clamp_min(1).to(delta.dtype)
    return delta.sum(-2) / count


class SignedGlobalTranslation(nn.Module):
    """One shared mapping per valid hand; no across-hand pre-averaging.

    Each hand contributes a 12D normalized-world signed basis:
    root cumulative, root adjacent, mean controller cumulative, mean controller
    adjacent. An object/hand-context-conditioned 3x12 matrix maps that basis
    to a 3D correction. Missing hands or zero motion always contribute zero.
    """
    def __init__(self, width=128):
        super().__init__()
        input_dim = 2 * width + 12
        self.mapping = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU(),
            nn.Linear(width, 36),
        )
        nn.init.zeros_(self.mapping[-1].weight)
        nn.init.zeros_(self.mapping[-1].bias)

    def forward(self, x, step, *, query, anchor_valid, hand_tokens,
                cp, cp_valid, previous_cp, previous_valid):
        b = cp.shape[0]
        hands = x["operation_hands"]
        hand_state = x["action_state"][:, step]
        initial_state = x["action_state"][:, 0]
        current_hands = x["action_valid"][:, step] & hands & torch.isfinite(hand_state).all(-1)
        initial_hands = x["action_valid"][:, 0] & hands & torch.isfinite(initial_state).all(-1)
        now = masked(hand_state[..., :3], current_hands)
        start = masked(initial_state[..., :3], initial_hands)
        root_cumulative = masked(now - start, current_hands & initial_hands)

        if step:
            past_state = x["action_state"][:, step - 1]
            past_hands = x["action_valid"][:, step - 1] & hands & torch.isfinite(past_state).all(-1)
            before = masked(past_state[..., :3], past_hands)
            root_adjacent = masked(now - before, current_hands & past_hands)
        else:
            past_hands = torch.zeros_like(current_hands)
            root_adjacent = torch.zeros_like(root_cumulative)

        valid0 = (x["controller_mask"][:, 0].reshape(b, 2, 30) & hands[..., None]
                  & torch.isfinite(x["controller_points"][:, 0]).all(-1))
        cp0 = torch.where(valid0[..., None], x["controller_points"][:, 0], 0.)
        current_cp = cp.reshape(b, 2, 30, 3)
        now_valid = cp_valid.reshape(b, 2, 30) & current_hands[..., None]
        first_valid = valid0 & initial_hands[..., None]
        controller_cumulative = _mean_valid_displacement(
            current_cp, cp0, now_valid & first_valid)
        if step:
            past_cp = previous_cp.reshape(b, 2, 30, 3)
            past_valid = previous_valid.reshape(b, 2, 30) & past_hands[..., None]
            controller_adjacent = _mean_valid_displacement(
                current_cp, past_cp, now_valid & past_valid)
        else:
            controller_adjacent = torch.zeros_like(controller_cumulative)

        basis = torch.cat((root_cumulative, root_adjacent,
                           controller_cumulative, controller_adjacent), dim=-1)
        basis = masked(basis, current_hands)
        # Existing per-hand context is still available, but the signed basis
        # bypasses controller cross-attention and its null competition.
        global_query = (masked(query, anchor_valid).sum(1)
                        / anchor_valid.sum(1, keepdim=True).clamp_min(1).to(query.dtype))
        ctx = torch.cat((global_query[:, None].expand(-1, 2, -1), hand_tokens, basis), -1)
        matrices = self.mapping(ctx).reshape(b, 2, 3, 12)
        per_hand = torch.einsum("bhij,bhj->bhi", matrices, basis)
        per_hand = masked(per_hand, current_hands)
        correction = per_hand.sum(1)
        return correction, dict(hand_basis=basis, hand_contribution=per_hand,
                                hand_valid=current_hands)


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
