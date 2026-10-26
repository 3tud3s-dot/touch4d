"""Optional episode-local previous GS; no cross-frame point correspondences."""

import numpy as np
import torch


def validate_history(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError("history must be 1 or 2")
    return value


def add_previous(sample, info, start):
    from .dataset import read_gs, GS_ATTR_FIELDS, DataFormatError
    if start < 1 or start >= info.frames:
        raise DataFormatError("Two-frame history must remain within this episode")
    meta = sample["meta"]
    if meta["episode_id"] != info.episode_id or int(meta["frame_ids"][0]) != start:
        raise DataFormatError("History episode/current frame mismatch")
    previous = read_gs(info, start-1)
    if not info.timestamps[start-1] < info.timestamps[start]:
        raise DataFormatError("History timestamps must strictly precede current time")
    sample["inputs"].update(gs_prev_xyz=torch.from_numpy(previous["xyz"].copy()),
        gs_prev_attr=torch.from_numpy(np.concatenate([previous[k] for k in GS_ATTR_FIELDS], -1)))
    meta.update(history_frame_ids=torch.tensor([start-1, start], dtype=torch.int64),
                history_timestamp_us=torch.from_numpy(info.timestamps[start-1:start+1].copy()))
    return sample


def history_present(batch):
    x, meta = batch["inputs"], batch["meta"]
    keys = [k in x for k in ("gs_prev_xyz", "gs_prev_attr", "gs_prev_mask", "gs_prev_lengths")]
    keys += [k in meta for k in ("history_frame_ids", "history_timestamp_us")]
    if any(keys) and not all(keys):
        raise ValueError("Incomplete temporal pair fields")
    return all(keys)


def validate_pair_batch(batch):
    if not history_present(batch):
        return
    x, meta = batch["inputs"], batch["meta"]
    xyz, attr, keep = x["gs_prev_xyz"], x["gs_prev_attr"], x["gs_prev_mask"]
    b = len(meta["scene"])
    if (xyz.ndim != 3 or xyz.shape[0] != b or xyz.shape[-1] != 3 or xyz.shape[1] < 1
            or xyz.dtype != torch.float32 or xyz.device != x["gs_xyz"].device
            or attr.shape != (*xyz.shape[:2], 11) or attr.dtype != torch.float32
            or attr.device != xyz.device or keep.shape != xyz.shape[:2]
            or keep.dtype != torch.bool or keep.device != xyz.device):
        raise ValueError("Previous GS requires independently padded XYZ/attributes/Boolean mask")
    lengths = x["gs_prev_lengths"]
    if (lengths.shape != (b,) or lengths.dtype != torch.int64
            or not torch.equal(keep.sum(-1), lengths.to(keep.device)) or not (lengths > 0).all()):
        raise ValueError("Previous GS lengths/mask mismatch")
    if (not torch.isfinite(xyz).all() or not torch.isfinite(attr).all()
            or torch.any(xyz[~keep] != 0) or torch.any(attr[~keep] != 0)):
        raise ValueError("Previous GS must be finite with zero padding")
    ids, times = meta["history_frame_ids"], meta["history_timestamp_us"]
    if any(v.shape != (b, 2) or v.dtype != torch.int64 for v in (ids, times)):
        raise ValueError("History metadata must be int64 [B,2]")
    if (not torch.equal(ids[:, 1], meta["frame_ids"][:, 0])
            or not torch.equal(ids[:, 0]+1, ids[:, 1]) or (ids[:, 0] < 0).any()
            or not torch.equal(times[:, 1], meta["observation_timestamp_us"][:, 0])
            or not (times[:, 0] < times[:, 1]).all()):
        raise ValueError("History must be exactly t-1,t, with strictly causal timestamps")
