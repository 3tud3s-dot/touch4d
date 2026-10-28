"""G4: signed per-hand motion -> additive global translation of V4 D0.

The production V4 class and checkpoint keys are untouched. This experiment
reproduces its complete autoregressive loop, changing only the accumulated
anchor displacement with a shared XYZ correction. The new head is exactly
zero at initialization, so G0 is a functionally exact starting point.
"""
import torch
from torch import nn

from scripts.training.gs_decoder import pool_anchor_features, interpolate_anchor_displacements
from scripts.training.model import initial_knn, masked
from scripts.training.v4_model import V4Stage1Predictor, fixed_field


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


class G4GlobalMotionPredictor(V4Stage1Predictor):
    """V4 H-step model with exactly one global translation correction per step.

    No new Anchor-Attention, RBF, loss, data path, or per-GS decoder. The
    original V4 module parameters and RNG initialization remain unchanged.
    """
    def __init__(self, horizon=15, prediction_interval=1):
        super().__init__(horizon=horizon, prediction_interval=prediction_interval)
        self.config = dict(self.config, architecture="v4_g4_global_motion_v1",
                           global_motion="signed_per_hand_global_translation_v1")
        self.global_motion = SignedGlobalTranslation(width=128)

    def forward(self, x, *, return_trace=False):
        h = self.config["horizon"]
        frame_ids = x.get("condition_frame_ids")
        if (frame_ids is None or frame_ids.dtype != torch.int64
                or frame_ids.shape != (len(x["gs_xyz"]), h)
                or not torch.all(frame_ids[:, 1:] - frame_ids[:, :-1] == self.config["prediction_interval"])):
            raise ValueError("V4 condition_frame_ids [B,H] must match configured prediction_interval")
        if any(x[key].shape[1] != h for key in ("action_state", "controller_points", "tactile")):
            raise ValueError("G4 requires exactly H sampled condition frames")
        keep = x["gs_mask"]
        if keep.dtype != torch.bool or not keep.any(1).all():
            raise ValueError("Each sample needs at least one valid GS")
        initial, attr = masked(x["gs_xyz"], keep), masked(x["gs_attr"], keep)
        if not torch.isfinite(initial).all() or not torch.isfinite(attr).all():
            raise ValueError("Nonfinite valid initial GS")
        field = fixed_field(initial, keep)
        neighbors = initial_knn(initial, keep, 16)
        batches = torch.arange(len(initial), device=initial.device)[:, None]
        current = initial
        previous_anchor = masked(current[batches, field.anchor_indices], field.anchor_mask)
        previous_d = torch.zeros_like(previous_anchor)
        previous_cp = previous_valid = None
        states, traces = [], []
        for step in range(h):
            features = masked(self.gs_encoder(torch.cat((current, attr), -1)), keep)
            features = self.local(features, current, keep, neighbors)
            query = masked(pool_anchor_features(features, field) + self.pred_time.weight[step], field.anchor_mask)
            anchors = masked(current[batches, field.anchor_indices], field.anchor_mask)
            tools, valid, cp, condition_trace = self.conditions(x, step)
            for block in self.cross:
                query, alpha = block(query, tools, anchors, cp, field.bandwidth,
                                     field.anchor_mask, valid)
            adjacent = torch.zeros_like(valid) if step == 0 else valid & previous_valid
            cp_delta = torch.zeros_like(cp) if step == 0 else masked(cp - previous_cp, adjacent)
            weights = alpha[..., :60]
            tool_motion = torch.einsum("bahj,bjd->bahd", weights, cp_delta).flatten(-2)
            history = torch.zeros_like(anchors) if step == 0 else anchors - previous_anchor
            head_mass = (weights * adjacent[:, None, None]).sum(-1)
            anchor_history_valid = field.anchor_mask[..., None].to(query.dtype) * (step > 0)
            motion_input = torch.cat((query, tool_motion, history, previous_d,
                                       head_mass, anchor_history_valid), -1)
            base_d = masked(self.motion(motion_input), field.anchor_mask)
            global_delta, global_trace = self.global_motion(
                x, step, query=query, anchor_valid=field.anchor_mask,
                hand_tokens=condition_trace["tokens62"][:, :2], cp=cp, cp_valid=valid,
                previous_cp=previous_cp, previous_valid=previous_valid)
            d = masked(base_d + global_delta[:, None], field.anchor_mask)
            next_xyz = masked(initial + interpolate_anchor_displacements(d, field, keep), keep)
            if not torch.isfinite(next_xyz).all():
                raise FloatingPointError("Nonfinite G4 prediction")
            states.append(next_xyz)
            if return_trace:
                traces.append(dict(alpha=alpha, anchor_xyz=anchors, anchor_history=history,
                                   tool_motion=tool_motion, adjacent_valid=adjacent, head_history_mass=head_mass,
                                   base_cumulative_d=base_d, global_delta=global_delta,
                                   cumulative_d=d, previous_d=previous_d, state=next_xyz,
                                   **global_trace, **condition_trace))
            previous_anchor, previous_d = anchors, d
            previous_cp, previous_valid = cp, valid
            current = next_xyz
        prediction = {"xyz": torch.stack(states, 1)}
        for key, begin, end in (("log_scales", 0, 3), ("quaternions", 3, 7),
                                ("opacity_logits", 7, 8), ("sh_dc", 8, 11)):
            prediction[key] = attr[:, None, :, begin:end].expand(-1, h, -1, -1)
        return (prediction, {"steps": traces, "field": field}) if return_trace else prediction
