"""Read-only smoke check. Run from the project root with:

python -B -m scripts.dataloader.check --scene 001-rope --split train --batch-size 2
Add --max-batches 0 to check every window in the selected split/stage. Output
goes only to stdout; this does not replace the reconstruction's full validator.
"""

import argparse
from itertools import islice
import json

import torch

from .collate import DEPTH_FIELDS, batch_horizon, fixed_fields, validate_depth_batch, validate_target_attr_batch
from .dataset import DEFAULT_ROOT
from .geometry import STREAM_HANDS
from .loader import make_dataloader
from .splits import SPLITS


def check_batch(batch):
    """Check semantic invariants in actual collated data; raise on first failure."""
    inputs, targets, meta = batch["inputs"], batch["targets"], batch["meta"]

    def require(condition, message):
        if not bool(condition):
            raise ValueError(message)

    horizon = batch_horizon(batch)
    b = len(meta["scene"])
    for key, (shape, dtype) in fixed_fields(horizon).items():
        value = inputs[key]
        require(value.shape == (b, *shape) and value.dtype == dtype, f"Invalid shape/dtype for {key}")
    for key in ("target_xyz", "target_mask"):
        require(isinstance(targets[key], list) and len(targets[key]) == horizon, f"{key} length disagrees with H")
    require(targets["target_lengths"].shape == (b, horizon), "target_lengths must be [B,H]")
    require(meta["observation_timestamp_us"].shape == (b, horizon+1), "timestamps must cover H+1 frames")
    point_groups = [(inputs["gs_xyz"], inputs["gs_mask"], inputs["gs_lengths"])]
    point_groups += list(zip(targets["target_xyz"], targets["target_mask"],
                            targets["target_lengths"].unbind(dim=1)))
    for xyz, mask, lengths in point_groups:
        require(torch.equal(mask.sum(1), lengths), "point counts disagree with masks")
        require(torch.all(xyz[~mask] == 0), "point padding must be zero")
        require(torch.isfinite(xyz).all(), "nonfinite GS points")
    require(torch.all(inputs["gs_attr"][~inputs["gs_mask"]] == 0), "GS attribute padding must be zero")
    require(torch.isfinite(inputs["gs_attr"]).all(), "nonfinite GS attributes")
    validate_target_attr_batch(batch)
    for values, mask in (("action_state", "action_valid"), ("controller_points", "controller_mask"),
                         ("tactile", "valid_mask"), ("tactile_xyz", "tactile_spatial_mask")):
        require(torch.isfinite(inputs[values]).all(), f"nonfinite {values}")
        require(torch.all(inputs[values][~inputs[mask]] == 0), f"{values}: invalid entries must be zero")
    expected = inputs["valid_mask"] & inputs["action_valid"][:, :, STREAM_HANDS, None, None]
    require(torch.all(~inputs["tactile_spatial_mask"] | expected), "spatial mask exceeds tactile/pose validity")
    require(torch.all(inputs["tactile"] >= 0), "negative tactile response")
    require(torch.all(meta["frame_ids"][:, 1:] - meta["frame_ids"][:, :-1] == 1), "nonconsecutive frames")
    require(torch.all(meta["observation_timestamp_us"][:, 1:] > meta["observation_timestamp_us"][:, :-1]),
            "nonincreasing observation timestamps")
    validate_depth_batch(batch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=tuple(SPLITS))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--stage", choices=("single", "all"), default="all")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Root containing rebuilt scene directories")
    parser.add_argument("--with-depth", action="store_true", help="Require and load the separate depth extension")
    parser.add_argument("--depth-root", help="Depth extension root; defaults to root's sibling touch4d-depth-v1")
    parser.add_argument("--depth-cameras", nargs="+", help="Ordered camera IDs; default is first two shared cameras")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1, help="0 checks every selected window")
    args = parser.parse_args(argv)
    if args.max_batches < 0:
        parser.error("--max-batches must be >= 0")
    if not args.with_depth and (args.depth_root is not None or args.depth_cameras is not None):
        parser.error("--depth-root/--depth-cameras require --with-depth")
    loader = make_dataloader(args.scene, args.split, root=args.root, stage=args.stage,
                             batch_size=args.batch_size, num_workers=args.workers, seed=args.seed,
                             with_depth=args.with_depth, depth_root=args.depth_root,
                             depth_cameras=args.depth_cameras, horizon=args.horizon)
    batches = loader if args.max_batches == 0 else islice(loader, args.max_batches)
    checked_samples = checked_batches = 0
    first_shapes = None
    for batch in batches:
        check_batch(batch)
        checked_batches += 1
        checked_samples += len(batch["meta"]["scene"])
        if first_shapes is None:
            first_shapes = {key: list(value.shape) for key, value in batch["inputs"].items()}
            first_shapes["target_xyz"] = [list(x.shape) for x in batch["targets"]["target_xyz"]]
            if "target_attr" in batch["targets"]:
                first_shapes["target_attr"] = [list(x.shape) for x in batch["targets"]["target_attr"]]
            if args.with_depth:
                first_shapes.update({key: list(batch["targets"][key].shape) for key in DEPTH_FIELDS})
    print(json.dumps({"scene": args.scene, "split": args.split, "stage": args.stage,
                      "horizon": args.horizon, "episodes": loader.dataset.episode_ids, "total_windows": len(loader.dataset),
                      "checked_samples": checked_samples, "checked_batches": checked_batches,
                      "all_windows_checked": checked_samples == len(loader.dataset),
                      "with_depth": args.with_depth, "depth_cameras": loader.dataset.depth_cameras,
                      "first_batch_shapes": first_shapes, "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
