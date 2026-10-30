"""Observed signed motion only; no learned Global/Local outputs or GT inputs.

The calculations intentionally mirror G4 SignedGlobalTranslation and G5
AnchorSignedDirect. They are independently checked against those original
traces so the recurrent input can be assembled BEFORE the current Global head.
"""
import torch
from scripts.training.model import masked
from scripts.experiments.v4_g4_global_motion.model import _mean_valid_displacement
from scripts.experiments.v4_g5_signed_local.motion import geometry_weights


def observed_hand_basis12(x, step, cp, cp_valid, previous_cp, previous_valid):
    b = cp.shape[0]
    hands = x['operation_hands']
    hand_state = x['action_state'][:, step]
    initial_state = x['action_state'][:, 0]
    current_hands = x['action_valid'][:, step] & hands & torch.isfinite(hand_state).all(-1)
    initial_hands = x['action_valid'][:, 0] & hands & torch.isfinite(initial_state).all(-1)
    now = masked(hand_state[..., :3], current_hands)
    start = masked(initial_state[..., :3], initial_hands)
    root_cumulative = masked(now - start, current_hands & initial_hands)
    if step:
        past_state = x['action_state'][:, step - 1]
        past_hands = x['action_valid'][:, step - 1] & hands & torch.isfinite(past_state).all(-1)
        before = masked(past_state[..., :3], past_hands)
        root_adjacent = masked(now - before, current_hands & past_hands)
    else:
        past_hands = torch.zeros_like(current_hands)
        root_adjacent = torch.zeros_like(root_cumulative)
    valid0 = (x['controller_mask'][:, 0].reshape(b, 2, 30) & hands[..., None]
              & torch.isfinite(x['controller_points'][:, 0]).all(-1))
    cp0 = torch.where(valid0[..., None], x['controller_points'][:, 0], 0.)
    current_cp = cp.reshape(b, 2, 30, 3)
    now_valid = cp_valid.reshape(b, 2, 30) & current_hands[..., None]
    first_valid = valid0 & initial_hands[..., None]
    controller_cumulative = _mean_valid_displacement(current_cp, cp0, now_valid & first_valid)
    if step:
        past_cp = previous_cp.reshape(b, 2, 30, 3)
        past_valid = previous_valid.reshape(b, 2, 30) & past_hands[..., None]
        controller_adjacent = _mean_valid_displacement(current_cp, past_cp, now_valid & past_valid)
    else:
        controller_adjacent = torch.zeros_like(controller_cumulative)
    basis = torch.cat((root_cumulative, root_adjacent, controller_cumulative, controller_adjacent), -1)
    return masked(basis, current_hands), current_hands


def observed_anchor_basis18(x, step, *, anchor_xyz, anchor_valid, hand_basis12,
                            hand_valid, cp, cp_valid, previous_cp, previous_valid, sigma):
    b, a, _ = anchor_xyz.shape
    allowed = anchor_valid[:, :, None] & hand_valid[:, None]
    current = cp.reshape(b, 2, 30, 3)
    now_valid = cp_valid.reshape(b, 2, 30) & hand_valid[..., None] & torch.isfinite(current).all(-1)
    current = torch.where(now_valid[..., None], current, torch.zeros_like(current))
    initial = x['controller_points'][:, 0]
    initial_state = x['action_state'][:, 0]
    initial_hand = x['action_valid'][:, 0] & x['operation_hands'] & torch.isfinite(initial_state).all(-1)
    valid0 = x['controller_mask'][:, 0] & initial_hand[..., None] & torch.isfinite(initial).all(-1)
    initial = torch.where(valid0[..., None], initial, torch.zeros_like(initial))
    endpoints = now_valid & valid0
    cumulative = torch.where(endpoints[..., None], current - initial, torch.zeros_like(current))
    beta_a, _ = geometry_weights(anchor_xyz, current, endpoints, anchor_valid, sigma)
    r_a = torch.einsum('bahj,bhjd->bahd', beta_a, cumulative)
    if step:
        past_state = x['action_state'][:, step - 1]
        past_hand = x['action_valid'][:, step - 1] & x['operation_hands'] & torch.isfinite(past_state).all(-1)
        previous = previous_cp.reshape(b, 2, 30, 3)
        past_valid = previous_valid.reshape(b, 2, 30) & past_hand[..., None] & torch.isfinite(previous).all(-1)
        previous = torch.where(past_valid[..., None], previous, torch.zeros_like(previous))
        adjacent_valid = now_valid & past_valid
        adjacent = torch.where(adjacent_valid[..., None], current - previous, torch.zeros_like(current))
        beta_delta, _ = geometry_weights(anchor_xyz, current, adjacent_valid, anchor_valid, sigma)
        r_delta = torch.einsum('bahj,bhjd->bahd', beta_delta, adjacent)
    else:
        beta_delta = torch.zeros_like(beta_a)
        r_delta = torch.zeros_like(r_a)
    basis = torch.cat((hand_basis12[:, None].expand(-1, a, -1, -1), r_a, r_delta), -1)
    return masked(basis, allowed), dict(controller_cumulative_weights=beta_a,
                                      controller_adjacent_weights=beta_delta)


def observed_signed_basis(x, step, *, anchor_xyz, anchor_valid, cp, cp_valid,
                          previous_cp, previous_valid, sigma):
    hand12, hand_valid = observed_hand_basis12(x, step, cp, cp_valid, previous_cp, previous_valid)
    basis18, diagnostics = observed_anchor_basis18(x, step, anchor_xyz=anchor_xyz,
        anchor_valid=anchor_valid, hand_basis12=hand12, hand_valid=hand_valid, cp=cp,
        cp_valid=cp_valid, previous_cp=previous_cp, previous_valid=previous_valid, sigma=sigma)
    return hand12, hand_valid, basis18, diagnostics
