"""Metadata binding and the finger -> gripper root -> world transform (metres)."""

import numpy as np

HAND_ORDER = ("left", "right")
STREAM_ORDER = ("tactilel_left", "tactilel_right", "tactiler_left", "tactiler_right")
STREAM_HANDS = (0, 0, 1, 1)
RIGID_ATOL = 1e-6
OPENING_RANGE = (0.04, 0.112)


def hand_slots(metadata: dict) -> tuple[int | None, int | None]:
    """Return fixed [left, right] -> original robot slot, rejecting ambiguous binding."""
    if metadata.get("hand_order") != list(HAND_ORDER):
        raise ValueError("metadata.hand_order must be ['left', 'right']")
    if metadata.get("stream_order") != list(STREAM_ORDER):
        raise ValueError("metadata.stream_order does not match the four tactile channels")
    slots = metadata.get("robot_slot_hands")
    mapping = metadata.get("physical_hand_to_robot_slot")
    if (not isinstance(slots, list) or len(slots) not in (1, 2)
            or any(h not in HAND_ORDER for h in slots) or len(set(slots)) != len(slots)):
        raise ValueError("robot_slot_hands must uniquely identify one or two physical hands")
    if not isinstance(mapping, dict) or set(mapping) != set(HAND_ORDER):
        raise ValueError("physical_hand_to_robot_slot must explicitly contain left and right")
    result = []
    for hand in HAND_ORDER:
        slot = mapping[hand]
        expected = slots.index(hand) if hand in slots else None
        if (slot is not None and type(slot) is not int) or slot != expected:
            raise ValueError(f"Inconsistent slot binding for {hand}: {slot!r} vs {slots!r}")
        result.append(slot)
    return tuple(result)


def pose_validity(transforms: np.ndarray, openings: np.ndarray) -> np.ndarray:
    """Absolute tolerance 1e-6, rtol=0; finite out-of-range openings remain valid."""
    finite = np.isfinite(transforms).all(axis=(-1, -2)) & np.isfinite(openings)
    safe = np.where(finite[..., None, None], transforms.astype(np.float64), np.eye(4))
    rotation = safe[..., :3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        orth = np.max(np.abs(np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3)), axis=(-1, -2))
        det = np.abs(np.linalg.det(rotation) - 1)
        bottom = np.max(np.abs(safe[..., 3, :] - [0, 0, 0, 1]), axis=-1)
    return finite & (orth <= RIGID_ATOL) & (det <= RIGID_ATOL) & (bottom <= RIGID_ATOL)


def action_conditions(actions, transforms, openings, slots):
    """Reorder original actions to [F,2,13]; controller masks play no role here."""
    if not (np.array_equal(actions[..., 0, :], transforms[..., :3, 3], equal_nan=True)
            and np.array_equal(actions[..., 1:4, :], transforms[..., :3, :3], equal_nan=True)
            and np.array_equal(actions[..., 4, 0], openings, equal_nan=True)
            and np.all(actions[..., 4, 1:] == 0)):
        raise ValueError("actions disagree with same-frame T_worlds/openings or their fixed zeros")
    state = np.zeros((len(actions), 2, 13), dtype=np.float32)
    valid = np.zeros((len(actions), 2), dtype=bool)
    slot_valid = pose_validity(transforms, openings)
    for hand, slot in enumerate(slots):
        if slot is None:
            continue
        valid[:, hand] = slot_valid[:, slot]
        values = actions[:, slot].reshape(-1, 15)[:, :13]
        with np.errstate(over="ignore", invalid="ignore"):
            state[:, hand] = np.where(valid[:, hand, None], values, 0).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("A valid action cannot be represented as finite float32")
    return state, valid


def tactile_world_coordinates(transforms, openings, slots, action_valid, valid_mask):
    """Return float32 world XYZ and spatial mask, leaving 2D tactile validity intact."""
    frames = len(transforms)
    xyz = np.zeros((frames, 4, 12, 32, 3), dtype=np.float32)
    spatial = np.zeros((frames, 4, 12, 32), dtype=bool)
    rows, cols = np.meshgrid(np.arange(12), np.arange(32), indexing="ij")
    for stream, hand in enumerate(STREAM_HANDS):
        slot = slots[hand]
        if slot is None:
            continue
        sign = 1 if stream % 2 == 0 else -1
        base = np.array([-0.04246242, 0.0835, 0.0097] if sign == 1
                        else [0.04246242, 0.0835, 0.0107])
        finger = np.stack((np.full_like(rows, sign * 0.007, dtype=np.float64),
                           -0.056 - 0.002 * cols, 0.0235 - 0.002 * rows), axis=-1)
        # Substitute safe arithmetic operands for invalid poses, then apply the mask.
        good = action_valid[:, hand]
        opening = np.where(good, openings[:, slot], OPENING_RANGE[0])
        q = 0.038 - 0.033 * (np.clip(opening, *OPENING_RANGE) - 0.04) / 0.072
        root = np.broadcast_to(base + finger * [1, -1, -1], (frames, 12, 32, 3)).copy()
        root[..., 0] += sign * q[:, None, None]
        transform = np.where(good[:, None, None], transforms[:, slot], np.eye(4))
        with np.errstate(over="ignore", invalid="ignore"):
            world = (np.einsum("fij,frcj->frci", transform[:, :3, :3], root)
                     + transform[:, None, None, :3, 3]).astype(np.float32)
        spatial[:, stream] = valid_mask[:, stream] & good[:, None, None] & np.isfinite(world).all(-1)
        xyz[:, stream] = np.where(spatial[:, stream, ..., None], world, 0)
    return xyz, spatial
