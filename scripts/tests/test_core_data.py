"""CPU-only invariants for the retained V1 -> fixed temporal cloth protocol."""
from contextlib import ExitStack, contextmanager
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.dataloader import dataset as dataset_module
from scripts.dataloader.collate import collate_fn
from scripts.dataloader.dataset import DataFormatError, Touch4DDataset
from scripts.dataloader.geometry import HAND_ORDER, STREAM_ORDER
from scripts.rebuild_dataset import rebuild

def metadata(scene="001-rope", episode=0, frames=12, hands=("right",)):
    return {"schema_version": "touch4d-v1", "script_version": "1.3.0", "scene": scene,
            "episode": episode, "frames": frames, "hand_order": list(HAND_ORDER),
            "stream_order": list(STREAM_ORDER), "robot_slot_hands": list(hands),
            "physical_hand_to_robot_slot": {h: hands.index(h) if h in hands else None for h in HAND_ORDER},
            "gs_point_counts": [1 + (episode + frame) % 4 for frame in range(frames)],
            "parameters": {"rigid_atol": 1e-6, "rigid_rtol": 0, "opening_clip_m": [0.04, 0.112],
                           "crop_rows": [0, 12], "invalid_columns": [31], "row_flip": False,
                           "max_age_us": 75000, "gs_coordinate_transform": "identity; native world metres",
                           "quaternion_order": "WXYZ"}}

def encode_actions(transforms, openings):
    values = np.zeros((*openings.shape, 5, 3), dtype=transforms.dtype)
    values[..., 0, :] = transforms[..., :3, 3]
    values[..., 1:4, :] = transforms[..., :3, :3]
    values[..., 4, 0] = openings
    return values

class MemoryCloth:
    """Supply real NPZ bytes to np.load without creating any filesystem fixtures."""

    def __init__(self, scene="008-pink-cloth", frames=307):
        self.root = Path("__memory_only_touch4d__").resolve()
        self.scene = scene
        self.files = {}
        self.add_episode(0, frames)

    def path(self, episode, name):
        return self.root / self.scene / f"episode_{episode}" / name

    def get(self, episode, name):
        return self.files[self.path(episode, name)]

    def add_episode(self, episode, frames, hands=None):
        if hands is None:
            hands = ("right",) if episode < 5 else ("right", "left")
        meta = metadata(self.scene, episode, frames, hands)
        self.files[self.path(episode, "metadata.json")] = meta
        transforms = np.broadcast_to(np.eye(4), (frames, len(hands), 4, 4)).copy()
        for slot in range(len(hands)):
            transforms[:, slot, :3, 3] = np.stack((np.arange(frames) + 100 * slot,
                                                  np.full(frames, episode), np.zeros(frames)), axis=-1)
        openings = np.full((frames, len(hands)), 0.08, dtype=np.float64)
        points = np.zeros((frames, 2, 30, 3), dtype=np.float32)
        controller_mask = np.zeros((frames, 2, 30), dtype=bool)
        for hand, name in enumerate(HAND_ORDER):
            if name in hands:
                points[:, hand] = 10 * (hand + 1)
                controller_mask[:, hand] = True
        self.files[self.path(episode, "controller.npz")] = {
            "T_worlds": transforms, "openings": openings, "actions": encode_actions(transforms, openings),
            "controller_points": points, "controller_valid_mask": controller_mask,
            "opening_clipped": np.zeros((frames, 2), dtype=bool)}
        valid = np.ones((frames, 4, 12, 32), dtype=bool)
        valid[..., 31] = False
        self.files[self.path(episode, "tactile.npz")] = {
            "tactile": np.zeros(valid.shape, dtype=np.float32), "valid_mask": valid}
        ts = 1_700_000_000_000_000 + np.arange(frames, dtype=np.int64) * 100_000
        times = {"observation_timestamp_us": ts, "tactile_age_us": np.zeros((frames, 4), dtype=np.int64),
                 "tactile_raw_index": np.tile(np.arange(frames, dtype=np.int64)[:, None], (1, 4))}
        times.update({f"raw_tactile_timestamp_us_{stream}": ts.copy() for stream in range(4)})
        self.files[self.path(episode, "timestamps.npz")] = times
        for frame, n in enumerate(meta["gs_point_counts"]):
            xyz = np.zeros((n, 3), dtype=np.float32)
            xyz[:, 0], xyz[:, 1], xyz[:, 2] = episode, frame, np.arange(n)
            self.files[self.path(episode, f"gs/frame_{frame:06d}.npz")] = {
                "xyz": xyz, "log_scales": np.full((n, 3), -2, dtype=np.float32),
                "quaternions": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
                "opacity_logits": np.full((n, 1), 0.3, dtype=np.float32),
                "sh_dc": np.full((n, 3), 0.1, dtype=np.float32), "sh_rest": np.zeros((n, 45), dtype=np.float32),
                "frame_id": np.asarray(frame, dtype=np.int64), "timestamp_us": np.asarray(ts[frame])}

    @contextmanager
    def mounted(self):
        original_load = np.load

        def read_text(path, **kwargs):
            if path not in self.files:
                raise FileNotFoundError(str(path))
            return json.dumps(self.files[path])

        def load(path, **kwargs):
            if hasattr(path, "read"):
                return original_load(path, **kwargs)
            if Path(path) not in self.files:
                raise FileNotFoundError(str(path))
            values = self.files[Path(path)]
            stream = BytesIO()
            if isinstance(values, bytes):
                stream.write(values)
            else:
                np.savez(stream, **values)
            stream.seek(0)
            return original_load(stream, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "read_text", read_text))
            stack.enter_context(patch.object(Path, "is_file", lambda path: path in self.files))
            stack.enter_context(patch.object(dataset_module.np, "load", load))
            yield self

class CoreDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.source = MemoryCloth()

    def dataset(self, split="train"):
        ranges = {"train": (0, 113, 268), "val": (0, 268, 307)}
        return Touch4DDataset("008-pink-cloth", split, self.source.root,
                              horizon=15, prediction_interval=1,
                              episode_frame_range=ranges[split])

    def test_temporal_manifest_has_140_training_and_24_heldout_windows(self):
        with self.source.mounted():
            train, val = self.dataset(), self.dataset("val")
            self.assertEqual(train.windows, tuple((0, s) for s in range(113,253)))
            self.assertEqual(val.windows, tuple((0, s) for s in range(268,292)))
            self.assertEqual(len(train),140); self.assertEqual(len(val),24)
            self.assertEqual(train[-1]["meta"]["frame_ids"].tolist(),list(range(252,268)))
            self.assertEqual(val[0]["meta"]["frame_ids"].tolist(),list(range(268,284)))
            self.assertEqual(val[-1]["meta"]["frame_ids"].tolist(),list(range(291,307)))

    def test_inputs_stop_at_t_plus_14_and_future_gs_only_changes_target(self):
        with self.source.mounted():
            sample=self.dataset()[0]
            self.assertEqual(sample["inputs"]["action_state"][:,1,0].tolist(),list(range(113,128)))
            self.assertEqual([int(x[0,1]) for x in sample["targets"]["target_xyz"]],list(range(114,129)))
            before=copy.deepcopy(sample["inputs"])
            self.source.get(0,"tactile.npz")["tactile"][128,:,:,:31]=73
            self.source.get(0,"gs/frame_000128.npz")["xyz"] += .5
            after=self.dataset()[0]
            for key,value in before.items():
                self.assertTrue(torch.equal(value,after["inputs"][key]),key)
            self.assertFalse(torch.equal(sample["targets"]["target_xyz"][-1],after["targets"]["target_xyz"][-1]))

    def test_nonfuture_tactile_selection_and_75ms_inclusive_boundary(self):
        raw=np.array([100,100000],dtype=np.int64)
        observations=np.array([90,100,175000,175001],dtype=np.int64)
        indices,ages,valid=rebuild.select_raw(raw,observations)
        np.testing.assert_array_equal(indices,[-1,0,1,1])
        np.testing.assert_array_equal(ages,[-1,0,75000,75001])
        np.testing.assert_array_equal(valid,[False,True,True,False])
        values,mask,_,_=rebuild.tactile_arrays(np.full((2,16,32),4,np.float32),np.ones((16,32),np.float32),raw,observations)
        self.assertEqual(values[1,0,0],3)
        self.assertFalse(mask[0].any());self.assertFalse(mask[3].any())
        self.assertTrue(mask[2,:,:31].all());self.assertFalse(mask[:,:,31].any())
        self.assertTrue(np.all(values[~mask]==0))

    def test_saved_tactile_provenance_cannot_point_at_a_future_record(self):
        self.source.get(0,"timestamps.npz")["tactile_raw_index"][113,0]+=1
        with self.source.mounted(),self.assertRaisesRegex(DataFormatError,"indices/ages disagree"):
            self.dataset()

    def test_stale_tactile_must_be_masked_while_75000us_remains_valid(self):
        times=self.source.get(0,"timestamps.npz")
        ts=times["observation_timestamp_us"]
        times["raw_tactile_timestamp_us_2"]=ts-75000
        times["tactile_age_us"][:,2]=75000
        with self.source.mounted():
            self.assertTrue(self.dataset()[0]["inputs"]["valid_mask"][:,2,:,:31].all())
        times["raw_tactile_timestamp_us_2"]=ts-75001
        times["tactile_age_us"][:,2]=75001
        with self.source.mounted(),self.assertRaisesRegex(DataFormatError,"75 ms"):
            self.dataset()[0]
        self.source.get(0,"tactile.npz")["valid_mask"][:,2]=False
        with self.source.mounted():
            sample=self.dataset()[0]
            self.assertFalse(sample["inputs"]["valid_mask"][:,2].any())
            self.assertTrue((sample["inputs"]["tactile_age_us"][:,2]==75001).all())

    def test_prepare_preserves_fixed_normalization_and_right_hand_causality(self):
        from scripts.training.cloth_data import Cloth008Processor, prepare_batch
        processor=Cloth008Processor("008-pink-cloth")
        state=processor.state_dict()
        self.assertEqual(state["center"],[0.21795742779917135,0.010968994921301998,0.14623293373089752])
        self.assertEqual(state["scale"],0.20123094573217368)
        self.assertEqual(Cloth008Processor.from_state_dict(state).state_dict(),state)
        model=SimpleNamespace(config={"horizon":15,"prediction_interval":1})
        with self.source.mounted():raw=collate_fn([self.dataset()[0]])
        before=copy.deepcopy(raw)
        out=prepare_batch(raw,processor,model)
        self.assertEqual(out["inputs"]["operation_hands"].tolist(),[[False,True]])
        self.assertEqual(out["inputs"]["condition_frame_ids"].tolist(),[list(range(113,128))])
        self.assertTrue(torch.equal(raw["inputs"]["gs_xyz"],before["inputs"]["gs_xyz"]))
        self.assertNotIn("condition_frame_ids",raw["inputs"])
        restored=processor.denormalize_batch(out)
        torch.testing.assert_close(restored["inputs"]["gs_xyz"],raw["inputs"]["gs_xyz"],atol=2e-6,rtol=2e-6)
        self.assertEqual(processor.state_dict(),state)
        for target,original in zip(out["targets"]["target_xyz"],before["targets"]["target_xyz"]):
            self.assertFalse(torch.equal(target,original))

    def test_prepare_rejects_wrong_hand_or_cross_boundary_window(self):
        from scripts.training.cloth_data import Cloth008Processor, prepare_batch
        model=SimpleNamespace(config={"horizon":15,"prediction_interval":1})
        processor=Cloth008Processor("008-pink-cloth")
        with self.source.mounted():raw=collate_fn([self.dataset()[0]])
        wrong_hand=copy.deepcopy(raw);wrong_hand["inputs"]["action_valid"][:,:,0]=True
        with self.assertRaisesRegex(ValueError,"right-hand-only"):
            prepare_batch(wrong_hand,processor,model)
        crossed=copy.deepcopy(raw);crossed["meta"]["frame_ids"]+=145
        with self.assertRaisesRegex(ValueError,"consecutive H15"):
            prepare_batch(crossed,processor,model)
        model.config["prediction_interval"]=2
        with self.assertRaisesRegex(ValueError,"consecutive H15"):
            prepare_batch(raw,processor,model)

    def test_metadata_identity_is_enforced_before_accepting_real_normalization_inputs(self):
        from scripts.training import cloth_data
        info=SimpleNamespace(slots=(None,0),path=Path("__memory_only_identity__"))
        with patch.object(cloth_data,"Touch4DDataset",return_value=SimpleNamespace(episodes={0:info})),patch.object(Path,"read_bytes",return_value=b"unreviewed metadata"):
            with self.assertRaisesRegex(ValueError,"metadata differs"):
                cloth_data.make_dataset({"root":"unused","depth_root":"unused"},"train")
        info.slots=(0,None)
        with patch.object(cloth_data,"Touch4DDataset",return_value=SimpleNamespace(episodes={0:info})):
            with self.assertRaisesRegex(ValueError,"right-hand-only metadata"):
                cloth_data.make_dataset({"root":"unused","depth_root":"unused"},"train")

    def test_validation_never_fits_normalization_from_validation_split(self):
        from scripts.training.cloth_data import validate_normalization
        with self.assertRaisesRegex(ValueError,"training-window initial GS"):
            validate_normalization(SimpleNamespace(split="val"))

    def test_source_paths_cannot_escape_and_episode_selection_does_not_renumber(self):
        self.assertEqual(rebuild.parse_episodes("0,1,5-6"),[0,1,5,6])
        self.assertEqual(rebuild.hand_slots("008-pink-cloth",0),["right"])
        self.assertEqual(rebuild.hand_slots("008-pink-cloth",5),["right","left"])
        for value in ["../escape","/absolute","a/../../escape"]:
            with self.subTest(value=value),self.assertRaises(ValueError):rebuild.safe_relative(value)
        with self.assertRaises(ValueError):rebuild.hand_slots("unreviewed-scene",0)


if __name__=="__main__":
    unittest.main()
