"""Read-only acceptance: raw DataLoader -> normalization -> inverse checks.

python -B -m scripts.data_process.check --scene 001-rope --split train --workers 2
Checks a few batches plus a batch containing the first/last selected windows.
Prints JSON only; does not save samples, train, fit parameters or edit sources.
"""

import argparse
import copy
from itertools import islice
import json

import torch

from ..dataloader.check import check_batch
from ..dataloader.collate import collate_fn
from ..dataloader.dataset import DEFAULT_ROOT
from ..dataloader.loader import make_dataloader
from .processor import DataProcessor, SUPPORTED_SCENES


def _leaves(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _leaves(child, f"{path}[{index}]")
    else:
        yield path, value


def require_unchanged(before, after):
    """Exact leaf comparison for checking that the raw batch was not mutated."""
    left, right = dict(_leaves(before)), dict(_leaves(after))
    if left.keys() != right.keys():
        raise ValueError("Batch structure changed")
    for key, a in left.items():
        b = right[key]
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.device != b.device or not torch.equal(a, b):
                raise ValueError(f"Input batch was changed at {key}")
        elif type(a) is not type(b) or a != b:
            raise ValueError(f"Input metadata was changed at {key}")


def check_processed_batch(raw, processed, processor):
    """Check independent float64 forward references and inverse consistency."""
    check_batch(raw)
    check_batch(processed)
    x, y = raw["inputs"], processed["inputs"]
    constants = processor.state_dict()
    scale, tactile_scale = constants["scale"], constants["tactile_scale"]
    changed_paths = {"inputs.gs_xyz", "inputs.gs_attr", "inputs.action_state",
                     "inputs.controller_points", "inputs.tactile_xyz", "inputs.tactile"}
    changed_paths.update(f"targets.target_xyz[{k}]" for k in range(5))
    raw_leaves, out_leaves = dict(_leaves(raw)), dict(_leaves(processed))
    if raw_leaves.keys() != out_leaves.keys():
        raise ValueError("Normalized batch structure changed")
    for key, a in raw_leaves.items():
        b = out_leaves[key]
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                raise ValueError(f"Shape/dtype/device changed at {key}")
            if a.numel() and a.data_ptr() == b.data_ptr():
                raise ValueError(f"Output aliases input storage at {key}")
        if key not in changed_paths:
            require_unchanged(a, b)

    def close(actual, expected):
        torch.testing.assert_close(actual.double(), expected.double(), atol=2e-6, rtol=2e-6)

    def expected_xyz(values, mask):
        center = torch.tensor(constants["center"], dtype=torch.float64, device=values.device)
        normalized = (values.double()-center)/scale
        return torch.where(mask[..., None], normalized, 0.)

    for field, mask in (("gs_xyz", "gs_mask"), ("controller_points", "controller_mask"),
                        ("tactile_xyz", "tactile_spatial_mask")):
        close(y[field], expected_xyz(x[field], x[mask]))
    for k in range(5):
        close(processed["targets"]["target_xyz"][k],
              expected_xyz(raw["targets"]["target_xyz"][k], raw["targets"]["target_mask"][k]))
    attr = x["gs_attr"].double().clone()
    attr[..., :3] -= torch.tensor(scale, dtype=torch.float64, device=attr.device).log()
    close(y["gs_attr"], torch.where(x["gs_mask"][..., None], attr, 0.))
    action = x["action_state"].double().clone()
    action[..., :3] = expected_xyz(x["action_state"][..., :3], x["action_valid"])
    action[..., 12] /= scale
    close(y["action_state"], torch.where(x["action_valid"][..., None], action, 0.))
    close(y["tactile"], x["tactile"].double()/tactile_scale)
    if not torch.equal(y["gs_attr"][..., 3:], x["gs_attr"][..., 3:]):
        raise ValueError("Quaternion/opacity/SH changed")
    if not torch.equal(y["action_state"][..., 3:12], x["action_state"][..., 3:12]):
        raise ValueError("Action rotation changed")
    zero = x["valid_mask"] & (x["tactile"] == 0)
    positive = x["valid_mask"] & (x["tactile"] > 0)
    if not bool(torch.all(y["tactile"][zero] == 0) & torch.all(y["tactile"][positive] > 0)):
        raise ValueError("Valid zero/positive tactile semantics changed")

    restored = processor.denormalize_batch(processed)
    check_batch(restored)
    restored_leaves = dict(_leaves(restored))
    errors = {}
    for key, a in raw_leaves.items():
        b = restored_leaves[key]
        if key in changed_paths:
            close(b, a)
            errors[key] = float((a.double()-b.double()).abs().max())
        else:
            require_unchanged(a, b)
    return {"roundtrip_max_abs_errors": errors,
            "valid_zero_tactile": int(zero.sum()), "valid_positive_tactile": int(positive.sum()),
            "positive_2d_without_3d": int((positive & ~x["tactile_spatial_mask"]).sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=SUPPORTED_SCENES)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--stage", choices=("single", "all"), default="all")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=3, help="0 checks all selected windows")
    args = parser.parse_args(argv)
    if args.max_batches < 0:
        parser.error("--max-batches must be >= 0")
    loader = make_dataloader(args.scene, args.split, root=args.root, stage=args.stage,
                             batch_size=args.batch_size, num_workers=args.workers, seed=args.seed)
    processor = DataProcessor(args.scene)
    results, checked_samples, errors = [], 0, {}

    def verify(batch):
        nonlocal checked_samples
        snapshot = copy.deepcopy(batch)
        normalized = processor(batch)
        result = check_processed_batch(batch, normalized, processor)
        require_unchanged(snapshot, batch)
        for key, value in result["roundtrip_max_abs_errors"].items():
            errors[key] = max(errors.get(key, 0.), value)
        checked_samples += len(batch["meta"]["scene"])
        results.append(result)

    iterator = loader if args.max_batches == 0 else islice(loader, args.max_batches)
    for batch in iterator:
        verify(batch)
    normal_batches, normal_samples = len(results), checked_samples
    first, last = loader.dataset[0], loader.dataset[-1]
    verify(collate_fn([first, last]))
    print(json.dumps({"status": "passed", "scene": args.scene, "split": args.split, "stage": args.stage,
                      "parameters": processor.state_dict(), "episodes": loader.dataset.episode_ids,
                      "total_windows": len(loader.dataset), "normal_batches": normal_batches,
                      "normal_samples": normal_samples, "boundary_batches": 1, "boundary_samples": 2,
                      "checked_batches": len(results), "checked_samples": checked_samples,
                      "all_windows_checked": normal_samples == len(loader.dataset),
                      "boundary_episodes": [first["meta"]["episode_id"], last["meta"]["episode_id"]],
                      "boundary_frame_ids": [first["meta"]["frame_ids"].tolist(), last["meta"]["frame_ids"].tolist()],
                      "roundtrip_max_abs_errors": errors,
                      "valid_zero_tactile": sum(r["valid_zero_tactile"] for r in results),
                      "valid_positive_tactile": sum(r["valid_positive_tactile"] for r in results),
                      "positive_2d_without_3d": sum(r["positive_2d_without_3d"] for r in results)}, indent=2))


if __name__ == "__main__":
    main()
