"""Structural Encoder versioning and recovery tests; no disk or CUDA writes."""

import copy
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from scripts.data_process import DataProcessor
from scripts.training.evaluate import load_model
from scripts.experiments.fixed_window import fixed_fit
from scripts.training.model import PositionPredictor
from scripts.tests.training.test_fixed_fit import single_raw
from scripts.training.train import (checkpoint_state, checkpoint_config, restore_training,
                    recover_best_score, publish_checkpoints, resolve_structure)


class StructuralCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.processor = DataProcessor("001-rope")
        self.batch = self.processor(single_raw())

    def model(self, structure="none", decoder="anchors"):
        return PositionPredictor(width=128 if structure == "anchor_attention" else 32,
            heads=4, layers=2, output_mode="gs_geometry", gs_decoder=decoder,
            gs_local_attention="knn", gs_local_k=16, gs_structure=structure)

    def state(self, model, optimizer=None, best=.4, version=None):
        optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = dict(model.config, scene="001-rope", depth_weight=0.,
                      chamfer_chunk=2, rgb_weight=1., alpha_weight=.1,
                      scale_weight=.01, chamfer_weight=1.)
        state = checkpoint_state(model, optimizer, self.processor, config, 2, best, [])
        if version is not None:
            state["version"] = version
            state["config"].pop("gs_structure")
        # Exercise serialization independently of the mocked read path below.
        data = BytesIO()
        torch.save(state, data)
        data.seek(0)
        return torch.load(data, weights_only=False, map_location="cpu")

    def step(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(self.batch["inputs"])["xyz"].square().mean().backward()
        optimizer.step()

    def test_v4_p1a_v5_anisotropic_and_v6_structural_restore_exact_next_update(self):
        cases = (("none", "anchors", "touch4d-gs-v4"),
                 ("none", "anchors_anisotropic", "touch4d-gs-v5"),
                 ("none", "free", "touch4d-gs-v5"),
                 ("anchor_attention", "anchors", None))
        for structure, decoder, version in cases:
            with self.subTest(structure=structure, decoder=decoder, version=version):
                model = self.model(structure, decoder)
                torch.nn.init.normal_(model.position_head[-1].weight, std=.02)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                self.step(model, optimizer)
                state = self.state(model, optimizer, version=version)
                original_config = copy.deepcopy(state["config"])
                with patch("scripts.training.train.torch.load", return_value=state):
                    loaded, processor, config = load_model("memory/checkpoint.pt", "cpu")
                self.assertEqual(state["config"], original_config)
                self.assertEqual(config["gs_structure"], structure)
                self.assertEqual(processor.state_dict(), self.processor.state_dict())
                before, after = model(self.batch["inputs"]), loaded(self.batch["inputs"])
                for key in before:
                    torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
                restored_optimizer = torch.optim.AdamW(loaded.parameters(), lr=9.)
                restore_training(state, loaded, restored_optimizer)
                self.step(model, optimizer)
                self.step(loaded, restored_optimizer)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)

    def test_structure_metadata_weights_and_legacy_version_must_agree(self):
        p2 = self.state(self.model("anchor_attention"))
        p1a = self.state(self.model())
        prefix = "gs_structure."
        structural_keys = [key for key in p2["model"] if key.startswith(prefix)]
        self.assertTrue(structural_keys)
        bad = copy.deepcopy(p2)
        bad["config"].pop("gs_structure")
        with self.assertRaises(ValueError):
            checkpoint_config(bad)
        bad = copy.deepcopy(p2)
        bad["model"] = {key: value for key, value in bad["model"].items()
                        if not key.startswith(prefix)}
        with self.assertRaises(ValueError):
            checkpoint_config(bad)
        bad = copy.deepcopy(p1a)
        bad["model"][structural_keys[0]] = p2["model"][structural_keys[0]]
        with self.assertRaises(ValueError):
            checkpoint_config(bad)
        for version in ("touch4d-gs-v4", "touch4d-gs-v5"):
            bad = copy.deepcopy(p2)
            bad["version"] = version
            with self.assertRaises(ValueError):
                checkpoint_config(bad)
            # Legacy mode defaults must not silently swallow structural weights.
            bad["config"].pop("gs_structure")
            with self.assertRaises(ValueError):
                checkpoint_config(bad)
        bad = copy.deepcopy(p2)
        bad["config"]["gs_structure"] = "none"
        with self.assertRaises(ValueError):
            checkpoint_config(bad)

    def test_restore_and_cli_cannot_switch_structure_or_mutate_target_model(self):
        original = self.model()
        state = self.state(original, version="touch4d-gs-v4")
        self.assertEqual(resolve_structure(None, state), "none")
        self.assertEqual(resolve_structure("none", state), "none")
        with self.assertRaises(ValueError):
            resolve_structure("anchor_attention", state)
        p2 = self.model("anchor_attention")
        previous = {key: value.clone() for key, value in p2.state_dict().items()}
        with self.assertRaises(ValueError):
            restore_training(state, p2, torch.optim.AdamW(p2.parameters()))
        for key, value in previous.items():
            torch.testing.assert_close(p2.state_dict()[key], value, atol=0, rtol=0)
        structural = self.state(p2)
        self.assertEqual(resolve_structure(None, structural), "anchor_attention")
        with self.assertRaises(ValueError):
            resolve_structure("none", structural)
        with self.assertRaises(ValueError):
            resolve_structure("unknown")
        with self.assertRaises(ValueError):
            restore_training(structural, original, torch.optim.AdamW(original.parameters()))

    def test_legacy_best_and_resaved_none_last_keep_better_best(self):
        model = self.model()
        legacy = self.state(model, best=.2, version="touch4d-gs-v4")
        latest = self.state(model, best=.4)
        canonical = dict(legacy, config=checkpoint_config(legacy))
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("scripts.training.train.read_checkpoint", return_value=canonical), \
             redirect_stdout(StringIO()):
            best = recover_best_score(latest, "memory")
        self.assertEqual(best, .2)
        with patch("scripts.training.train.atomic_checkpoint") as save:
            publish_checkpoints(Path("memory"), dict(latest, best=best), .3 < best)
        self.assertEqual([call.args[0].name for call in save.call_args_list], ["last.pt"])
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("scripts.training.train.read_checkpoint", return_value=self.state(self.model("anchor_attention"))):
            with self.assertRaises(ValueError):
                recover_best_score(latest, "memory")

    def test_structural_fixed400_retains_better_lowest_cd_and_rejects_mode_switch(self):
        last = self.state(self.model("anchor_attention"))
        last["config"].update(protocol="fixed400", fit_diagnostic=True, fit_contract=None)
        recorded = dict(update=20, rank=[.4, .5, .1], eligible=None, diagnostic_only=True)
        last["fit"] = dict(completed_updates=20, best_candidate=recorded, manifest=["same exact windows"])
        best = copy.deepcopy(last)
        recovered = dict(recorded, update=40, rank=[.2, .3, .05])
        best["fit"].update(completed_updates=40, best_candidate=recovered)
        with patch("pathlib.Path.is_file", return_value=True), \
             patch.object(fixed_fit, "read_checkpoint", return_value=best):
            self.assertEqual(fixed_fit.recover_fit_best(last, "memory"), recovered)
        with patch.object(fixed_fit, "atomic_checkpoint") as save:
            fixed_fit.publish_fit("memory", last, improved=False)
        self.assertEqual([call.args[0].name for call in save.call_args_list],
                         ["step_000020.pt", "last.pt"])
        switched = copy.deepcopy(best)
        switched["config"]["gs_structure"] = "none"
        with patch("pathlib.Path.is_file", return_value=True), \
             patch.object(fixed_fit, "read_checkpoint", return_value=switched):
            with self.assertRaises(ValueError):
                fixed_fit.recover_fit_best(last, "memory")


if __name__ == "__main__":
    unittest.main()
