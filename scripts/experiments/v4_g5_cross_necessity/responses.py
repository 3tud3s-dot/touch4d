"""Frozen input interventions, not counterfactual physical ground truth.

Coordinates are the processed V4 normalized-world XYZ.  A half-root motion
changes the Root and every Controller of a hand by the SAME translation; it
does not halve individual Controller displacements or change rotation/opening.
Pressure, observation validity, age and time indices remain observationally
fixed.  This helper never reads targets or changes the supplied input tree.
"""
import copy
import torch


RESPONSE_MODES = ("FreezeAction", "HalfRootTranslation")


class ResponseUnavailable(ValueError):
    """The requested input intervention cannot preserve observed H1 support.

    ``diagnostics`` is JSON-compatible evidence, not a successful response.
    Callers may record this as NOT_APPLICABLE rather than fabricating a
    counterfactual or weakening the initial-step prediction equality check.
    """
    def __init__(self, diagnostics):
        self.diagnostics = copy.deepcopy(diagnostics)
        super().__init__(self.diagnostics["reason"])


def _contract(x):
    try:
        action, cp = x["action_state"], x["controller_points"]
        av, cm, hands = x["action_valid"], x["controller_mask"], x["operation_hands"]
    except KeyError as exc:
        raise ValueError("Missing processed V4 action field: " + str(exc)) from exc
    if not isinstance(action, torch.Tensor) or action.ndim != 4 or action.shape[-2:] != (2, 13):
        raise ValueError("action_state must be [B,H,2,13]")
    b, h = action.shape[:2]
    if h < 1 or b < 1 or not action.is_floating_point():
        raise ValueError("Nonempty floating V4 action sequence required")
    for name, value, shape in (
        ("controller_points", cp, (b, h, 2, 30, 3)),
        ("action_valid", av, (b, h, 2)),
        ("controller_mask", cm, (b, h, 2, 30)),
        ("operation_hands", hands, (b, 2)),
    ):
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.device != action.device:
            raise ValueError(name + " has the wrong V4 shape/device")
    if cp.dtype != action.dtype or not cp.is_floating_point():
        raise ValueError("Controller and Root coordinates must share floating dtype")
    if any(v.dtype != torch.bool for v in (av, cm, hands)):
        raise ValueError("Action, Controller and operation masks must be bool")
    return action, cp, av, cm, hands


def _masked(value, valid):
    return torch.where(valid[..., None], value, torch.zeros_like(value))


def _count(valid):
    return int(valid.sum().item())


def counterfactual_inputs(inputs, mode):
    """Return ``(independent_inputs_copy, JSON_compatible_diagnostics)``.

    FreezeAction retains the reviewed historical definition: copy the initial
    13D hand state and Controller positions, intersect current/initial validity.
    Finite checks and the existing operation-hand policy cannot resurrect data.

    HalfRootTranslation retains current rotation/opening and Controller shape.
    Where initial/current Root is unavailable, no rigid counterfactual shift is
    defined: the hand and its Controllers are disabled and the coverage loss is
    reported.  Missing initial Controller nodes alone do not invalidate a valid
    CURRENT position, since a rigid Root translation needs no initial taxel/node
    identity.  The model still computes its existing A0 endpoint-valid flags.
    If this would change any model-effective INITIAL Action/Controller support,
    raises ResponseUnavailable with a NOT_APPLICABLE diagnostics dictionary.
    This is required even when the after-valid set would make value equality
    vacuous.  The original inputs are unchanged on either success or failure.

    Optional ``tactile_xyz`` is sensor-surface WORLD geometry, not pressure or
    contact labels.  Freeze uses its initial positions; Half translates it with
    its hand.  Unsupported ancillary coordinates are zeroed and counted, without
    altering pressure validity/spatial masks.  Production V4 T3 uses fixed
    SAME-FINGER LOCAL geometry and never consumes this world XYZ field.  Pressure,
    all tactile masks, Age and temporal indices remain strictly unchanged.
    """
    if mode not in RESPONSE_MODES:
        raise ValueError("Unsupported frozen action intervention: " + str(mode))
    action, cp, av, cm, hands = _contract(inputs)
    out = copy.deepcopy(inputs)
    allowed_hands = hands[:, None]
    now_valid = av & allowed_hands & torch.isfinite(action).all(-1)
    cp_valid = cm & allowed_hands[..., None] & torch.isfinite(cp).all(-1)
    root_pair = now_valid & now_valid[:, :1]
    initial_action = action[:, :1].expand_as(action)
    initial_cp = cp[:, :1].expand_as(cp)
    if mode == "FreezeAction":
        valid = root_pair
        node_valid = cp_valid & cp_valid[:, :1]
        changed_action = _masked(initial_action, valid)
        changed_cp = _masked(initial_cp, node_valid)
        geometry_reference = initial_cp - initial_action[..., :3].unsqueeze(-2)
    else:
        valid = root_pair
        # torch.where sanitizes BEFORE subtraction, so invalid NaNs are not
        # multiplied by zero or allowed into valid-coordinate calculations.
        root_now = _masked(action[..., :3], valid)
        root_first = _masked(initial_action[..., :3], valid)
        shift = .5 * (root_first - root_now)
        changed_action = action.clone()
        changed_action[..., :3] = root_now + shift
        changed_action = _masked(changed_action, valid)
        node_valid = cp_valid & valid[..., None]
        changed_cp = _masked(_masked(cp, node_valid) + shift[..., None, :], node_valid)
        geometry_reference = cp - action[..., :3].unsqueeze(-2)
    out["action_state"], out["action_valid"] = changed_action, valid
    out["controller_points"], out["controller_mask"] = changed_cp, node_valid
    meaningful = node_valid & valid[..., None]
    after_relative = changed_cp - changed_action[..., :3].unsqueeze(-2)
    error = (after_relative - geometry_reference)[meaningful]
    geometry_error = float(error.abs().max().item()) if error.numel() else None
    h1_action_valid = valid[:, 0]
    h1_cp_valid = node_valid[:, 0]
    h1_action_support_same = torch.equal(h1_action_valid, now_valid[:, 0])
    h1_cp_support_same = torch.equal(h1_cp_valid, cp_valid[:, 0])
    if not h1_action_support_same or not h1_cp_support_same:
        raise ResponseUnavailable(dict(
            status="NOT_APPLICABLE", mode=mode,
            reason="Counterfactual would change model-effective H1 Action/Controller observation support",
            h1_action_support_bit_exact=bool(h1_action_support_same),
            h1_controller_support_bit_exact=bool(h1_cp_support_same),
            h1_action_observations_before=_count(now_valid[:, 0]),
            h1_action_observations_after=_count(h1_action_valid),
            h1_controller_observations_before=_count(cp_valid[:, 0]),
            h1_controller_observations_after=_count(h1_cp_valid),
            h1_action_support_changes=(h1_action_valid != now_valid[:, 0]).nonzero().tolist(),
            h1_controller_support_changes=(h1_cp_valid != cp_valid[:, 0]).nonzero().tolist(),
            inputs_unchanged=True, counterfactual_constructed=False,
            limitation="Missing initial Root cannot justify removing observed initial Controllers",
        ))
    first_action_same = torch.equal(changed_action[:, 0][h1_action_valid], action[:, 0][h1_action_valid])
    first_cp_same = torch.equal(changed_cp[:, 0][h1_cp_valid], cp[:, 0][h1_cp_valid])
    if not first_action_same or not first_cp_same:
        raise AssertionError("Counterfactual must preserve observed initial state")
    if bool((valid & ~av).any()) or bool((node_valid & ~cm).any()):
        raise AssertionError("Counterfactual resurrected an invalid observation")
    if not torch.isfinite(changed_action).all() or not torch.isfinite(changed_cp).all():
        raise FloatingPointError("Nonfinite counterfactual action coordinates")
    tactile_geometry = dict(present=False)
    if "tactile_xyz" in inputs:
        xyz, spatial = inputs["tactile_xyz"], inputs.get("tactile_spatial_mask")
        expected = (*action.shape[:2], 4, 12, 32)
        if (not isinstance(xyz, torch.Tensor) or xyz.shape != (*expected, 3)
                or xyz.dtype != cp.dtype or xyz.device != cp.device
                or not isinstance(spatial, torch.Tensor) or spatial.shape != expected
                or spatial.dtype != torch.bool or spatial.device != cp.device):
            raise ValueError("tactile_xyz/spatial_mask require aligned processed V4 surface geometry")
        allowed_taxels = hands[:, None].repeat_interleave(2, -1)[..., None, None]
        observed_xyz_valid = spatial & allowed_taxels & torch.isfinite(xyz).all(-1)
        if mode == "FreezeAction":
            xyz_valid = observed_xyz_valid & observed_xyz_valid[:, :1]
            moved_xyz = _masked(xyz[:, :1].expand_as(xyz), xyz_valid)
            xyz_rule = "Initial same-finger sensor-surface world XYZ held fixed"
        else:
            xyz_valid = observed_xyz_valid & valid.repeat_interleave(2, -1)[..., None, None]
            sensor_shift = shift.repeat_interleave(2, 2)[..., None, None, :]
            moved_xyz = _masked(_masked(xyz, xyz_valid) + sensor_shift, xyz_valid)
            xyz_rule = "Current same-hand sensor-surface world XYZ rigidly translated with Root/Controllers"
        out["tactile_xyz"] = moved_xyz
        first_xyz_same = torch.equal(moved_xyz[:, 0][xyz_valid[:, 0]], xyz[:, 0][xyz_valid[:, 0]])
        if not first_xyz_same or not torch.isfinite(moved_xyz).all():
            raise AssertionError("Invalid ancillary tactile geometry intervention")
        tactile_geometry = dict(
            present=True, policy=xyz_rule,
            finite_observed_positions_before=_count(observed_xyz_valid),
            supported_positions_after=_count(xyz_valid),
            unsupported_positions_after=_count(spatial & allowed_taxels & ~xyz_valid),
            initial_supported_positions_bit_exact=bool(first_xyz_same),
            unsupported_coordinates_zeroed=True, pressure_validity_not_changed=True,
            used_by_v4_model=False,
        )
    diagnostics = dict(
        status="CONSTRUCTED", mode=mode, coordinate_system="V4 normalized-world XYZ; no additional scaling",
        action_observations_before=_count(now_valid), action_observations_after=_count(valid),
        controller_observations_before=_count(cp_valid), controller_observations_after=_count(node_valid),
        root_endpoint_unavailable_hands=_count(now_valid & ~now_valid[:, :1]),
        operation_hands_unchanged=True, no_invalid_observation_revived=True,
        h1_action_support_bit_exact=bool(h1_action_support_same),
        h1_controller_support_bit_exact=bool(h1_cp_support_same),
        initial_valid_action_bit_exact=bool(first_action_same),
        initial_valid_controller_bit_exact=bool(first_cp_same),
        rigid_geometry_checked_nodes=_count(meaningful),
        relative_root_controller_max_abs_error=geometry_error,
        rotation_opening=("initial_state_frozen" if mode == "FreezeAction" else "current_state_preserved"),
        tactile_pressure_masks_age_unchanged=True,
        tactile_xyz_geometry=tactile_geometry,
        time_indices_unchanged=True, targets_read=False,
        limitation="Sensitivity intervention only: no paired counterfactual GT or causal-physics claim",
    )
    return out, diagnostics
