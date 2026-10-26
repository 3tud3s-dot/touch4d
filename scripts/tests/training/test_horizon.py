"""Disk-free H=5/10/15 integration and legacy-checkpoint regression checks."""

import copy
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.data_process import DataProcessor
from scripts.tests.data_process.test_data_process import make_batch, add_depth_targets
from scripts.training.evaluate import load_model
from scripts.experiments.fixed_window import fixed_fit
from scripts.training.losses import chamfer_samples, objective, render_samples, depth_samples
from scripts.training.model import PositionPredictor
from scripts.training.rendering import GaussianRenderer, DepthRenderer
from scripts.tests.training.test_training import fake_gaussian_rasterizer, fake_rasterizer
from scripts.training.train import (checkpoint_state, checkpoint_config, restore_training, resolve_horizon,
                    copy_initial, GaussianMetrics, recover_best_score)
from scripts.training.visualize import (export_gaussians, geometry_data, geometry_html,
                        gs_viewer_html)


def cast_double(value):
    if isinstance(value, torch.Tensor):
        return value.double() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: cast_double(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cast_double(child) for child in value]
    return value


class HorizonTests(unittest.TestCase):
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

    def batch(self, horizon):
        raw = add_depth_targets(make_batch(horizon=horizon))
        return raw, self.processor(raw)

    def model(self, horizon=5, structure="none"):
        return PositionPredictor(width=32, heads=4, layers=2, horizon=horizon,
            output_mode="gs_geometry", gs_decoder="anchors", gs_structure=structure,
            gs_local_attention="knn", gs_local_k=4, anchor_count=8, anchor_pool_k=4)

    def config(self, model):
        return dict(model.config, scene="001-rope", depth_weight=0., chamfer_chunk=2,
                    rgb_weight=1., alpha_weight=.1, scale_weight=.01, chamfer_weight=1.)

    def test_lengths_padding_variable_counts_and_all_initial_attributes(self):
        for h in (5, 10, 15):
            raw, batch = self.batch(h)
            self.assertEqual(raw["meta"]["frame_ids"].shape[1], h+1)
            self.assertEqual(len(batch["targets"]["target_xyz"]), h)
            self.assertGreater(len(set(int(v) for v in raw["targets"]["target_lengths"][0])), 1)
            for structure in ("none", "anchor_attention"):
                model = self.model(h, structure)
                x = copy.deepcopy(batch["inputs"])
                x["gs_xyz"][~x["gs_mask"]] = float("nan")
                x["gs_attr"][~x["gs_mask"]] = float("nan")
                result = model(x)
                self.assertEqual(result["xyz"].shape, (*x["gs_xyz"].shape[:1], h, *x["gs_xyz"].shape[1:]))
                expected = copy_initial(batch["inputs"], "gs_geometry")
                for key, value in result.items():
                    self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(value, expected[key], atol=0, rtol=0)

    def test_fifteen_step_condition_prefix_causality_is_not_future_condition_access(self):
        _, batch = self.batch(15)
        for structure in ("none", "anchor_attention"):
            model = self.model(15, structure)
            torch.nn.init.normal_(model.position_head[-1].weight, std=.03)
            original = model(batch["inputs"])["xyz"]
            changed = copy.deepcopy(batch["inputs"])
            first_changed_condition = 7
            for key in ("action_state", "controller_points", "tactile"):
                changed[key][:, first_changed_condition:] += 5 * torch.randn_like(changed[key][:, first_changed_condition:])
            perturbed = model(changed)["xyz"]
            torch.testing.assert_close(perturbed[:, :7], original[:, :7], atol=0, rtol=0)
            self.assertGreater(float((perturbed[:, 7:] - original[:, 7:]).detach().abs().max()), 0.)
            self.assertFalse(torch.equal(original[:, 4], original[:, 9]))
            self.assertFalse(torch.equal(original[:, 4], original[:, 14]))

    def test_h15_last_loss_connects_to_every_earlier_anchor_output(self):
        _, batch = self.batch(15)
        x = cast_double(batch["inputs"])
        for structure in ("none", "anchor_attention"):
            model = self.model(15, structure).double()
            torch.nn.init.normal_(model.position_head[-1].weight, std=.08)
            retained = []
            def retain(module, args, output):
                output.retain_grad()
                retained.append(output)
            hook = model.position_head.register_forward_hook(retain)
            prediction = model(x)
            prediction["xyz"][:, -1].square().sum().backward()
            hook.remove()
            self.assertEqual(len(retained), 15)
            for h, output in enumerate(retained):
                self.assertIsNotNone(output.grad, f"step {h+1} disconnected")
                self.assertTrue(torch.isfinite(output.grad).all())
                self.assertGreater(float(output.grad.abs().sum()), 0., f"step {h+1} zero derivative")

    def test_horizon_and_condition_length_errors_fail_before_rollout(self):
        for invalid in (0, -1, True, 5., "15"):
            with self.assertRaises(ValueError):
                self.model(invalid)
        _, batch = self.batch(15)
        x = copy.deepcopy(batch["inputs"])
        x["action_state"] = x["action_state"][:, :14]
        with self.assertRaises(ValueError):
            self.model(15)(x)

    def test_all_horizons_contribute_equally_and_target_row_ids_do_not_matter(self):
        for h in (5, 10, 15):
            keep = torch.tensor([[True, True, False]])
            positions = torch.zeros(1, h, 3, 3, requires_grad=True)
            targets, masks = [], []
            for k in range(h):
                n = 1 + k % 3
                xyz = torch.zeros(1, n, 3)
                xyz[..., 0] = (k+1)/100.
                targets.append(xyz.requires_grad_())
                masks.append(torch.ones(1, n, dtype=torch.bool))
            batch = dict(inputs=dict(gs_xyz=torch.zeros(1, 3, 3), gs_mask=keep),
                         targets=dict(target_xyz=targets, target_mask=masks))
            loss = chamfer_samples(positions, batch, 2)
            torch.testing.assert_close(loss, torch.tensor([(h+1)/200.]))
            loss.sum().backward()
            self.assertTrue((positions.grad[:, :, :2].abs().sum((-1, -2)) > 0).all())
            self.assertTrue((positions.grad[:, :, 2] == 0).all())
            self.assertTrue(all(t.grad is None for t in targets))
            for k in range(h):
                targets[k] = targets[k].flip(1)
            torch.testing.assert_close(chamfer_samples(positions.detach(), batch, 2), loss.detach())

    def test_rgb_alpha_depth_masks_and_metrics_retain_h15_entries(self):
        raw, batch = self.batch(15)
        model = self.model(15)
        torch.nn.init.normal_(model.position_head[-1].weight, std=.03)
        renderer = GaussianRenderer(fake_gaussian_rasterizer)
        loss, values = objective(model(batch["inputs"]), batch, self.processor, renderer,
            0., 2, rgb_weight=1., alpha_weight=.1, scale_weight=.01, chamfer_weight=1.)
        self.assertEqual(tuple(values["rgb_by_horizon"].shape), (2, 15))
        self.assertEqual(tuple(values["alpha_by_horizon"].shape), (2, 15))
        self.assertTrue(values["render_time_available"].all())
        loss.backward()
        self.assertTrue(torch.isfinite(model.position_head[-1].weight.grad).all())
        tracker = GaussianMetrics(self.processor.state_dict()["scale"], self.config(model))
        tracker.add(batch["meta"]["episode_id"], values)
        self.assertEqual(len(tracker.summary()["rgb_by_horizon"]), 15)
        mask = torch.ones(1, 15, 1, 2, 2, dtype=torch.bool)
        mask[:, 6] = False
        prediction = torch.arange(1, 16, dtype=torch.float32)[None, :, None, None, None].expand_as(mask)
        score, available, times, time_valid = render_samples(prediction, torch.zeros_like(prediction), mask)
        self.assertAlmostEqual(float(score[0]), (120.-7.)/14., places=5)
        self.assertTrue(available.all())
        self.assertFalse(time_valid[0, 6])
        depth, _ = depth_samples(prediction, torch.zeros_like(prediction), mask)
        torch.testing.assert_close(depth, score)

    def test_rasterizers_render_every_unique_horizon_and_variable_gt_set(self):
        _, batch = self.batch(15)
        prediction = {key: value.clone() for key, value in copy_initial(batch["inputs"], "gs_geometry").items()}
        prediction["xyz"][..., 2] += torch.arange(15)[None, :, None] * .15
        calls = []
        def raster(**kwargs):
            calls.append(kwargs["means"].detach().clone())
            return fake_gaussian_rasterizer(**kwargs)
        renderer = GaussianRenderer(raster)
        result = renderer(prediction, batch, self.processor)
        target = renderer.render_targets(batch, self.processor)
        b, h, v, _, _ = batch["targets"]["target_depth"].shape
        self.assertEqual(len(calls), 2*b*h*v)
        self.assertEqual(result["rgb"].shape[:3], (b, 15, v))
        self.assertFalse(torch.equal(result["rgb"][:, 4], result["rgb"][:, 14]))
        self.assertGreater(len(set(len(x) for x in calls[b*h*v:])), 1)
        depth = DepthRenderer(fake_rasterizer)(prediction, batch, self.processor)
        self.assertEqual(depth.shape, batch["targets"]["target_depth"].shape)
        self.assertFalse(torch.equal(depth[:, 4], depth[:, 14]))
        self.assertFalse(target["rgb"].requires_grad)
        short = {key: value[:, :14] for key, value in prediction.items()}
        with self.assertRaises(ValueError):
            renderer(short, batch, self.processor)

    def test_v7_horizon_metadata_embedding_consistency_and_restore_restriction(self):
        for h in (5, 10, 15):
            model = self.model(h)
            optimizer = torch.optim.AdamW(model.parameters())
            state = checkpoint_state(model, optimizer, self.processor, self.config(model), 0, .4, [])
            self.assertEqual(checkpoint_config(state)["horizon"], h)
            self.assertEqual(resolve_horizon(None, state), h)
            wrong = copy.deepcopy(state)
            wrong["config"].pop("horizon")
            with self.assertRaises(ValueError):
                checkpoint_config(wrong)
            for name in ("condition_time.weight", "prediction_time.weight"):
                wrong = copy.deepcopy(state)
                wrong["model"][name] = wrong["model"][name][:-1]
                with self.assertRaises(ValueError):
                    checkpoint_config(wrong)
            with self.assertRaises(ValueError):
                resolve_horizon(10 if h == 5 else 5, state)
            other = self.model(10 if h == 5 else 5)
            with self.assertRaises(ValueError):
                restore_training(state, other, torch.optim.AdamW(other.parameters()))

    def test_h5_default_and_v6_restore_optimizer_behavior_and_better_best(self):
        torch.manual_seed(42)
        explicit = self.model(5)
        config = dict(explicit.config)
        config.pop("horizon")
        torch.manual_seed(42)
        default = PositionPredictor(**config)
        for name, value in explicit.state_dict().items():
            torch.testing.assert_close(default.state_dict()[name], value, atol=0, rtol=0)
        _, batch = self.batch(5)
        optimizer = torch.optim.AdamW(explicit.parameters(), lr=1e-4)
        def step(model, opt):
            opt.zero_grad(set_to_none=True)
            model(batch["inputs"])["xyz"].square().sum().backward()
            opt.step()
        step(explicit, optimizer)
        state = checkpoint_state(explicit, optimizer, self.processor, self.config(explicit), 0, .2, [])
        state["version"] = "touch4d-gs-v6"
        state["config"].pop("horizon")
        stream = BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=False)
        with patch("scripts.training.train.torch.load", return_value=state):
            restored, _, canonical = load_model("memory/v6.pt", "cpu")
        for name, value in explicit(batch["inputs"]).items():
            torch.testing.assert_close(restored(batch["inputs"])[name], value, atol=0, rtol=0)
        opt = torch.optim.AdamW(restored.parameters())
        restore_training(state, restored, opt)
        step(explicit, optimizer)
        step(restored, opt)
        for name, value in explicit.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
        latest = checkpoint_state(restored, opt, self.processor, canonical, 1, .4, [])
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("scripts.training.train.read_checkpoint", return_value=dict(state, config=canonical)), \
             redirect_stdout(StringIO()):
            self.assertEqual(recover_best_score(latest, "memory"), .2)

    def test_v6_structural_h5_checkpoint_keeps_original_structure_and_outputs(self):
        _, batch = self.batch(5)
        model = self.model(5, "anchor_attention")
        torch.nn.init.normal_(model.position_head[-1].weight, std=.03)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        state = checkpoint_state(model, optimizer, self.processor, self.config(model), 0, .4, [])
        state["version"] = "touch4d-gs-v6"
        state["config"].pop("horizon")
        with patch("scripts.training.train.torch.load", return_value=state):
            restored, _, config = load_model("memory/v6-p2.pt", "cpu")
        self.assertEqual(config["horizon"], 5)
        self.assertEqual(config["gs_structure"], "anchor_attention")
        for key, value in model(batch["inputs"]).items():
            torch.testing.assert_close(restored(batch["inputs"])[key], value, atol=0, rtol=0)
        forged = copy.deepcopy(state)
        forged["config"]["horizon"] = 15
        with self.assertRaises(ValueError):
            checkpoint_config(forged)

    def test_frozen_h15_manifest_provenance_frames_duplicates_and_resume(self):
        # Synthetic provenance strings test validation only; they are never
        # used as experimental evidence or as acceptance thresholds.
        document = dict(version="structural-h15-windows-v1", horizon=15,
            basis="synthetic GT-only quality audit", evidence_sha256="synthetic-audit-hash",
            windows={"001-rope": [dict(episode_id=episode, start_frame=start,
                frame_ids=list(range(start, start+16)), geometry_sha256="synthetic-geometry-hash")
                for episode, start in fixed_fit.WINDOWS["001-rope"]]})
        payload = json.dumps(document).encode()
        self.assertEqual(fixed_fit.frozen_windows(None, "001-rope", 5), (None, None))
        with self.assertRaisesRegex(ValueError, "manifest"):
            fixed_fit.frozen_windows(None, "001-rope", 15)
        with patch("pathlib.Path.read_bytes", return_value=payload):
            selected, evidence = fixed_fit.frozen_windows("memory/windows.json", "001-rope", 15)
        self.assertEqual(selected, [[1, 85], [2, 73]])
        state = dict(config=dict(window_evidence=evidence, fixed_windows=selected))
        self.assertEqual(fixed_fit.frozen_windows(None, "001-rope", 15, state), (selected, evidence))
        with patch("pathlib.Path.read_bytes", return_value=payload):
            self.assertEqual(fixed_fit.frozen_windows("memory/windows.json", "001-rope", 15, state),
                             (selected, evidence))
        changed = copy.deepcopy(document)
        changed["basis"] = "changed frozen audit"
        with patch("pathlib.Path.read_bytes", return_value=json.dumps(changed).encode()):
            with self.assertRaisesRegex(ValueError, "resume"):
                fixed_fit.frozen_windows("memory/windows.json", "001-rope", 15, state)
        mutations = (
            lambda d: d.update(horizon=10),
            lambda d: d.pop("basis"),
            lambda d: d.pop("evidence_sha256"),
            lambda d: d["windows"]["001-rope"][0]["frame_ids"].pop(),
            lambda d: d["windows"]["001-rope"][0]["frame_ids"].__setitem__(7, 999),
            lambda d: d["windows"]["001-rope"][0].update(episode_id=True),
            lambda d: d["windows"]["001-rope"][0].pop("geometry_sha256"),
            lambda d: d["windows"]["001-rope"].__setitem__(1, copy.deepcopy(d["windows"]["001-rope"][0])),
            lambda d: d["windows"]["001-rope"].pop(),
        )
        for mutate in mutations:
            changed = copy.deepcopy(document)
            mutate(changed)
            with patch("pathlib.Path.read_bytes", return_value=json.dumps(changed).encode()):
                with self.assertRaises(ValueError):
                    fixed_fit.frozen_windows("memory/windows.json", "001-rope", 15)

    def test_h15_exact_selection_rejects_actual_frame_and_dataset_mismatches(self):
        raw, _ = self.batch(15)
        def sample(value, index):
            if isinstance(value, torch.Tensor):
                return value[index:index+1].clone()
            if isinstance(value, dict):
                return {key: sample(child, index) for key, child in value.items()}
            if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                return [sample(child, index) for child in value]
            return copy.deepcopy(value[index:index+1])
        rows = [sample(raw, i) for i in range(2)]
        selected = [(int(r["meta"]["episode_id"][0]), int(r["meta"]["frame_ids"][0, 0])) for r in rows]
        class Dataset:
            horizon = 15
            windows = list(selected)
            def __getitem__(self, index):
                return rows[index]
        dataset = Dataset()
        with patch.object(fixed_fit, "collate_fn", side_effect=lambda samples: samples[0]):
            actual = fixed_fit.select_windows(dataset, "001-rope", selected)
            self.assertEqual([r["meta"]["frame_ids"].shape[1] for r in actual], [16, 16])
            with self.assertRaises(ValueError):
                fixed_fit.select_windows(dataset, "001-rope", [selected[0], selected[0]])
            dataset.windows = [selected[0], selected[0], selected[1]]
            with self.assertRaises(ValueError):
                fixed_fit.select_windows(dataset, "001-rope", selected)
            dataset.windows = list(selected)
            dataset.horizon = 10
            with self.assertRaisesRegex(ValueError, "mismatch"):
                fixed_fit.select_windows(dataset, "001-rope", selected)
            dataset.horizon = 15
            rows[0]["meta"]["frame_ids"][0, -1] += 1
            with self.assertRaisesRegex(ValueError, "mismatch"):
                fixed_fit.select_windows(dataset, "001-rope", selected)
            rows[0]["meta"]["frame_ids"][0, -1] -= 1
            manifest = fixed_fit.manifest(rows, dict(depth_cameras=["cam_a", "cam_b"], horizon=15))
            self.assertEqual(len(manifest[0]["frame_ids"]), 16)
            self.assertEqual(len(manifest[0]["positive_pixels"]), 15)
            with self.assertRaisesRegex(ValueError, "horizon"):
                fixed_fit.manifest(rows, dict(depth_cameras=["cam_a", "cam_b"], horizon=5))

    def test_copy_initial_and_metric_tracker_h15_and_legacy_h5_defaults(self):
        for h, config in ((5, {}), (15, dict(horizon=15))):
            raw, batch = self.batch(h)
            x = batch["inputs"]
            prediction = copy_initial(x, "gs_geometry")
            self.assertEqual(prediction["xyz"].shape[1], h)
            for k in range(h):
                torch.testing.assert_close(prediction["xyz"][:, k], x["gs_xyz"], atol=0, rtol=0)
            values = torch.arange(1, h+1, dtype=torch.float32)[None]/100.
            metric = dict(geometry=torch.tensor([.1]), depth_m=torch.tensor([.002]),
                depth_available=torch.tensor([True]), render_available=torch.tensor([True]),
                rgb=values.mean(1), alpha=2*values.mean(1), scale_regularization=torch.tensor([0.]),
                render_time_available=torch.ones(1, h, dtype=torch.bool),
                rgb_by_horizon=values, alpha_by_horizon=2*values)
            tracker = GaussianMetrics(.1, config)
            tracker.add(torch.tensor([1]), metric)
            summary = tracker.summary()
            np.testing.assert_allclose(summary["rgb_by_horizon"], values[0].numpy())
            np.testing.assert_allclose(summary["alpha_by_horizon"], 2*values[0].numpy())
            self.assertEqual(len(summary["episodes"]["1"]["rgb_by_horizon"]), h)
            before = copy.deepcopy((tracker.episodes, tracker.render))
            for key in ("render_time_available", "rgb_by_horizon", "alpha_by_horizon"):
                inconsistent = dict(metric, **{key: metric[key][:, :-1]})
                with self.assertRaisesRegex(ValueError, "horizon"):
                    tracker.add(torch.tensor([1]), inconsistent)
                self.assertEqual((tracker.episodes, tracker.render), before)
            other_horizon = GaussianMetrics(.1, dict(horizon=15 if h == 5 else 5))
            with self.assertRaisesRegex(ValueError, "horizon"):
                other_horizon.add(torch.tensor([1]), metric)
            self.assertEqual(other_horizon.episodes, {})
            self.assertEqual(other_horizon.render, {})
        fallback = {key: value for key, value in x.items() if key != "action_state"}
        self.assertEqual(copy_initial(fallback).shape[1], 15)
        minimal = dict(gs_xyz=x["gs_xyz"], gs_attr=x["gs_attr"])
        self.assertEqual(copy_initial(minimal, "gs_geometry")["xyz"].shape[1], 5)

    def test_export_retains_all_fifteen_future_frames_and_initial_reference_offline(self):
        raw, batch = self.batch(15)
        # Three cameras: select a repeated calibration but keep explicit distinct
        # camera IDs, so export coverage matches the real experiment's layout.
        for sample in (raw, batch):
            for key in ("target_depth", "target_depth_valid"):
                sample["targets"][key] = sample["targets"][key][:, :, [0, 1, 0]]
            for key in ("camera_intrinsics", "camera_to_world"):
                sample["targets"][key] = sample["targets"][key][:, [0, 1, 0]]
            sample["meta"]["depth_camera_ids"] = [["cam_a", "cam_b", "cam_c"] for _ in range(2)]
        prediction = {key: value.clone() for key, value in copy_initial(batch["inputs"], "gs_geometry").items()}
        prediction["xyz"][..., 2] += torch.arange(15)[None, :, None] * .15
        files, pages, saved = {}, {}, {}
        args = SimpleNamespace(output="memory/export-h15", checkpoint="memory/last_400.pt",
                               episode=1, start=85, split="train")
        def write_bytes(path, value):
            files[Path(path).name] = value
            return len(value)
        def write_text(path, value, **kwargs):
            pages[Path(path).name] = value
            return len(value)
        with patch("scripts.training.visualize.GaussianRenderer", side_effect=lambda **kwargs:
                   GaussianRenderer(fake_gaussian_rasterizer, **kwargs)), \
             patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes", autospec=True, side_effect=write_bytes), \
             patch("pathlib.Path.write_text", autospec=True, side_effect=write_text), \
             patch("numpy.savez_compressed", side_effect=lambda path, **kw: saved.update(kw)), \
             torch.no_grad(), redirect_stdout(StringIO()):
            export_gaussians(args, self.config(self.model(15)), raw, batch, prediction, self.processor)
        future = {f"frame_{h}_view_{v}_{mode}_{name}.png" for h in range(1, 16)
                  for v in range(3) for mode in ("rgb", "alpha", "depth")
                  for name in ("gt", "pred", "copy", "err")}
        self.assertEqual(len(future), 540)
        self.assertTrue(future <= set(files))
        self.assertIn("frame_0_view_0_rgb_pred.png", files)
        self.assertNotEqual(files["frame_5_view_0_rgb_pred.png"], files["frame_15_view_0_rgb_pred.png"])
        self.assertEqual(saved["predicted_xyz"].shape[0], 15)
        self.assertIn("target_xyz_15", saved)
        self.assertEqual(len(saved["frame_ids"]), 16)
        data = geometry_data(saved)
        self.assertEqual(len(data["prediction"]), 15)
        self.assertEqual(len(data["target"]), 15)
        self.assertNotEqual(data["prediction"][4]["centers"], data["prediction"][14]["centers"])
        for page in pages.values():
            self.assertNotIn("https://", page)
            self.assertNotIn("http://", page)
            self.assertIn(".max=", page)
        # Execute slider/playback JavaScript with a minimal DOM when Node is
        # available. This checks all sixteen displayed times, including h15.
        node = shutil.which("node")
        if node:
            dom = '''const devicePixelRatio=1,window={},elements={};let tick;
function setInterval(f){tick=f;return 1}function clearInterval(){}
const context={beginPath(){},stroke(){},fillRect(){},moveTo(){},lineTo(){},arc(){},fill(){}};
const document={getElementById(id){return elements[id]??={value:id==='view'?'orbit':id==='mode'?'rgb':0,checked:true,append(){},appendChild(){},getBoundingClientRect(){return {width:400,height:250}},getContext(){return context},setPointerCapture(){}}},createElement(){return {}},querySelectorAll(){return ['c0','c1','c2','e0','e1','e2'].map(id=>this.getElementById(id))}};
'''
            for filename, page in pages.items():
                script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
                slider = "frame" if filename == "geometry.html" else "time"
                extra = f'''if(+elements.{slider}.max!==15)throw Error('wrong slider maximum');for(let h=0;h<=15;h++){{elements.{slider}.value=h;elements.{slider}.oninput();if(!elements.label.textContent.includes(String(D.frame_ids[h])))throw Error('frame mismatch')}}
elements.play.onclick();tick();if(+elements.{slider}.value!==0)throw Error('playback must include all 16 frames');'''
                if filename == "index.html":
                    extra += "elements.time.value=15;elements.time.oninput();if(!elements.pred.src.includes('frame_15_view_0_rgb_pred'))throw Error('wrong final image');"
                result = subprocess.run([node], input=dom+script+extra, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
