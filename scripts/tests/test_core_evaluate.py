"""CPU checks for metric semantics, causal input framing and frozen rollout state."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts.training import cloth_evaluate as e
from scripts.training import soma_adapter as s


class MetricTests(unittest.TestCase):
    def test_independent_variable_count_chamfer(self):
        pred = torch.tensor([[0., 0, 0], [2., 0, 0]], dtype=torch.float64)
        target = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0]], dtype=torch.float64)
        self.assertAlmostEqual(e.chamfer_mm(pred, target).item(), 1000/6, places=10)

    def test_unsquared_metres_to_mm(self):
        pred = torch.tensor([[0., 0, 0]], dtype=torch.float64)
        target = torch.tensor([[2., 0, 0]], dtype=torch.float64)
        self.assertEqual(e.chamfer_mm(pred, target).item(), 2000.)

    def test_original_squared_knn_path(self):
        def knn(a, b, **_):
            distance = torch.cdist(a, b).square().min(-1).values[..., None]
            return SimpleNamespace(dists=distance)
        a = torch.tensor([[0., 0, 0], [1., 0, 0]], dtype=torch.float64)
        b = torch.tensor([[.2, 0, 0], [1., 0, 0], [2., 0, 0]], dtype=torch.float64)
        torch.testing.assert_close(e.chamfer_mm(a, b, knn), e.chamfer_mm(a, b), rtol=0, atol=0)

    def test_image_masks_and_equal_camera_average(self):
        rgb = torch.tensor([[[[.2]*3, [999.]*3]], [[[.8]*3, [.8]*3]]], dtype=torch.float64)
        alpha = rgb[..., 0]
        mask = torch.tensor([[[True, False]], [[True, True]]])
        score = e.image_errors(rgb, alpha, torch.zeros_like(rgb), torch.zeros_like(alpha), mask)
        self.assertAlmostEqual(score[0].item(), .5, places=12)
        self.assertAlmostEqual(score[1].item(), .5, places=12)
        expected_psnr = (-10*np.log10(.2**2)-10*np.log10(.8**2))/2
        self.assertAlmostEqual(score[2].item(), expected_psnr, places=12)

    def test_empty_camera_does_not_change_nonempty_camera_mean(self):
        rgb = torch.ones(2, 1, 1, 3)
        mask = torch.tensor([[[False]], [[True]]])
        score = e.image_errors(rgb, rgb[..., 0], torch.zeros_like(rgb), torch.zeros_like(rgb[..., 0]), mask)
        torch.testing.assert_close(score, torch.tensor([1., 1., 0.]), rtol=0, atol=0)

    def test_all_empty_mask_is_an_error(self):
        rgb = torch.zeros(2, 1, 1, 3)
        with self.assertRaisesRegex(ValueError, 'target-depth'):
            e.image_errors(rgb, rgb[..., 0], rgb, rgb[..., 0], torch.zeros(2, 1, 1, dtype=torch.bool))

    def test_empty_point_set_rejected(self):
        with self.assertRaises(ValueError):
            e.chamfer_mm(torch.zeros(0, 3), torch.zeros(2, 3))


class InputProtocolTests(unittest.TestCase):
    def raw(self, start=181, episode=1):
        return {'inputs': {'action_valid': torch.tensor([False, True])[None, None].expand(1, 15, 2).clone()},
                'meta': {'frame_ids': torch.arange(start, start+16)[None],
                         'episode_id': torch.tensor([episode]),
                         'observation_timestamp_us': torch.arange(16)[None]*33333},
                'targets': {'future_xyz': torch.randn(16, 4, 3)}}

    def test_conditions_end_at_t_plus_14(self):
        raw = self.raw()
        out = e.prepare(raw, lambda value: copy.deepcopy(value), SimpleNamespace(config={'horizon': 15, 'prediction_interval': 1}), 1, 181)
        self.assertEqual(out['inputs']['condition_frame_ids'].tolist(), [list(range(181, 196))])
        self.assertEqual(out['meta']['frame_ids'][0, -1].item(), 196)
        self.assertEqual(out['inputs']['operation_hands'].tolist(), [[False, True]])
        self.assertEqual(out['meta']['operation_hand_policy'], e.cloth_data.HAND_POLICY)

    def test_wrong_hand_or_nonmonotonic_time_rejected(self):
        model = SimpleNamespace(config={'horizon': 15, 'prediction_interval': 1})
        raw = self.raw(); raw['inputs']['action_valid'][0, 2, 0] = True
        with self.assertRaisesRegex(ValueError, 'right-hand'):
            e.prepare(raw, copy.deepcopy, model, 1, 181)
        raw = self.raw(); raw['meta']['observation_timestamp_us'][0, 3] = 0
        with self.assertRaisesRegex(ValueError, 'Nonmonotonic'):
            e.prepare(raw, copy.deepcopy, model, 1, 181)

    def test_cadence_uses_actual_recorded_timestamps(self):
        times = np.arange(16)*33333
        np.testing.assert_array_equal(e.validate_cadence(times), np.full(15, .033333))
        for bad in (np.arange(16)*50000, np.zeros(16), np.arange(15)*33333):
            with self.assertRaises(ValueError):
                e.validate_cadence(bad)

    def test_controller_partition_preserves_order(self):
        levels = s.controller_partition()
        self.assertEqual([len(x) for x in levels], [30, 10, 2])
        np.testing.assert_array_equal(levels[0].reshape(10, 3), np.arange(10)[:, None].repeat(3, 1))

    def test_paths_relocate_without_history_directories(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); project = root/'project'; runtime = root/'runtime'
            cfg = {'horizon': 15, 'delta': 1, 'windows': [dict(episode=p[0], start=p[1]) for p in e.CASES]}
            for key in e.PATH_FIELDS:
                cfg[key] = key
                base = project if key in ('soma_source', 'soma_config') else runtime
                base.mkdir(parents=True, exist_ok=True)
                (base/key).write_bytes(b'fixture')
            parsed = e.resolve_configuration(cfg, runtime, project)
            self.assertEqual(parsed['data_root'], runtime/'data_root')
            self.assertEqual(parsed['soma_config'], project/'soma_config')
            cfg['windows'] = list(reversed(cfg['windows']))
            with self.assertRaisesRegex(ValueError, 'five specified'):
                e.resolve_configuration(cfg, runtime, project)


class RolloutTests(unittest.TestCase):
    class Model:
        def __init__(self, bad_step=None):
            self.scene_attr = {'rope': torch.zeros(1, 1)}
            self._scene_attr_dict = {s.scene: 'rope'}
            self.frame_gap = 10
            self.calls = []
            self.bad_step = bad_step

        def _preprocess(self, previous, current, rest, action, destination, anchor, *args, **kwargs):
            self.calls.append(dict(previous=previous.clone(), current=current.clone(), rest=rest.clone(), anchor=anchor.clone()))
            return current, None

        def encode_decode(self, current, *args, **kwargs):
            out = current+.001
            if len(self.calls) == self.bad_step:
                out[0, 0] = float('nan')
            return torch.cat((torch.zeros(30, 3), out)), torch.ones(len(out), 6), None

    def call(self, model):
        initial = dict(xyz=torch.zeros(2, 3), log_scales=torch.zeros(2, 3), opacity_logits=torch.zeros(2, 1))
        original = torch.as_tensor
        def cpu_tensor(*args, **kwargs):
            kwargs.pop('device', None)
            return original(*args, **kwargs)
        actions = torch.arange(16)[:, None, None].expand(16, 30, 3).float()
        with patch.object(s.torch, 'as_tensor', side_effect=cpu_tensor):
            return s.soma_predict(model, initial, torch.ones(2, 6), [(np.zeros(1), np.zeros(1))], actions, torch.zeros(1, 3), 512)

    def test_h10_refresh_uses_prediction_and_preserves_boundary_history(self):
        model = self.Model()
        xyz, _ = self.call(model)
        self.assertEqual(xyz.shape, (15, 2, 3))
        torch.testing.assert_close(xyz[-1], torch.full((2, 3), .015), atol=1e-8, rtol=0)
        at_nine, at_ten = model.calls[9], model.calls[10]
        torch.testing.assert_close(at_nine['rest'], torch.zeros(2, 3), rtol=0, atol=0)
        self.assertTrue(torch.equal(at_ten['previous'], at_ten['current']))
        self.assertTrue(torch.equal(at_ten['rest'], at_ten['current']))
        self.assertTrue(torch.equal(at_ten['anchor'], torch.full((30, 3), 10.)))

    def test_nonfinite_rollout_cannot_be_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite model rollout'):
            self.call(self.Model(bad_step=5))


class NativeConstructorDependencyTests(unittest.TestCase):
    def test_relocation_selects_continuous_before_build_without_changing_math(self):
        make_env = lambda: SimpleNamespace(data_dir='/deleted/historical/package', dt=1/30,
                                          real_dt={s.scene: 1/15}, gravity=[0, 0, -9.8])
        cfg = SimpleNamespace(
            model=SimpleNamespace(test_rollout_mode='segmented', data_dir='/deleted/stage1-cache',
                                  gs_scene=[{'name': s.scene, 'model_path': '/deleted/old.ply'}],
                                  dt=1/15, frame_gap=10, use_rotation=False,
                                  backbone={'num_encoder_layers': 16, 'dt': 1/15}),
            data=SimpleNamespace(train={'dataset': {'env_cfg': make_env()}},
                                 val={'env_cfg': make_env()}, test={'env_cfg': make_env()}))
        before = copy.deepcopy(cfg)
        with tempfile.TemporaryDirectory() as name:
            package = Path(name)/'baseline'/s.scene
            geometry = package/'pi3/gs/point_cloud/iteration_10000/point_cloud.ply'
            record = s.relocate_runtime_config(cfg, package, geometry)
            self.assertEqual(cfg.model.test_rollout_mode, 'continuous')
            self.assertEqual(cfg.model.data_dir, str(package.parent.resolve()))
            self.assertEqual(cfg.model.gs_scene[0]['model_path'], str(geometry.resolve()))
            self.assertEqual(record['relocated_dataset_splits'], ['train', 'val', 'test'])
            self.assertFalse(record['stage1_prediction_cache_required'])
            for env in (cfg.data.train['dataset']['env_cfg'], cfg.data.val['env_cfg'], cfg.data.test['env_cfg']):
                self.assertEqual(env.data_dir, str(package.parent.resolve()))
                self.assertEqual(env.dt, 1/30)
                self.assertEqual(env.gravity, [0, 0, -9.8])
        self.assertEqual(cfg.model.dt, before.model.dt)
        self.assertEqual(cfg.model.frame_gap, before.model.frame_gap)
        self.assertEqual(cfg.model.backbone, before.model.backbone)

    def test_exact_native_continuous_constructor_never_reads_stage1_cache(self):
        project = Path(__file__).resolve().parents[2]
        path = project/'third_party/soma/mmgs/models/simulators/gs_simulator_embodied_stage2.py'
        module = ast.parse(path.read_text(encoding='utf-8'))
        klass = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'GsSimulatorEmbodiedS2')
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '_init_scene_init_pos_cov')
        forbidden = unittest.mock.Mock(side_effect=AssertionError('Historical Stage-1 cache accessed'))
        namespace = {'os': SimpleNamespace(path=__import__('os').path, listdir=forbidden),
                     'torch': SimpleNamespace(load=forbidden)}
        # Execute the exact vendored method with CPU spies; importing the full
        # third-party CUDA/DGL model is neither needed nor permitted here.
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        load_raw = unittest.mock.Mock()
        obj = SimpleNamespace(test_rollout_mode='continuous', data_dir='/deleted/historical/cache',
                              _init_scene_init_pos_cov_from_raw_gaussian=load_raw)
        scenes = [{'name': s.scene}]
        namespace['_init_scene_init_pos_cov'](obj, scenes)
        load_raw.assert_called_once_with(scenes)
        forbidden.assert_not_called()
        self.assertEqual(obj.scene_init_pos, {})
        obj.test_rollout_mode = 'segmented'
        with self.assertRaisesRegex(AssertionError, 'Stage-1 cache accessed'):
            namespace['_init_scene_init_pos_cov'](obj, scenes)

    def test_frozen_config_file_dependency_closure(self):
        project = Path(__file__).resolve().parents[2]
        config = project/'third_party/soma/configs/SoMA/deform360_v0_stage2.py'
        actual = s.configuration_files(config)
        expected = {config.resolve(),
                    (config.parent/'../_base_/schedules/adam_hood.py').resolve(),
                    (config.parent/'../_base_/default_runtime.py').resolve()}
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), 3)


class CompletedRunTests(unittest.TestCase):
    def records(self):
        model, processor = {'architecture': 'fixture'}, {'center': [0, 0, 0], 'scale': 1}
        config = {'epochs': 8, 'horizon': 15, 'prediction_interval': 1, 'seed': 42,
                  'lr': 1e-4, 'weight_decay': 1e-4, 'root': 'old-root', 'source_sha': 'old-code'}
        orders = [list(range(113, 253)) for _ in range(8)]
        history = [dict(starts=order.copy(), successful_updates=140) for order in orders]
        identity = dict(mode='formal', model=model, processor=processor, config=config,
                        base_sha=e.cloth_data.BASE_SHA, epoch_orders=orders, held_out_starts=list(range(268, 292)),
                        source_bundle_sha256='b'*64)
        state = dict(version='touch4d-g5-direct-008-episode0-v1', identity=identity, history=history)
        completion = dict(status='completed', mode='formal', checkpoint_sha256='a'*64,
                          source_bundle_sha256='b'*64, updates=1120,
                          trained_starts=[x for order in orders for x in order], evaluated_starts=list(range(268, 292)))
        return state, completion, model, processor, config

    def validate(self, state, completion, model, processor, config):
        e.validate_completed_state(state, completion, 'a'*64, 'b'*64, model, processor, config)

    def test_valid_completed_run_allows_only_path_relocation(self):
        state, completion, model, processor, config = self.records()
        expected = dict(config, root='relocated', source_sha='new-manifest-origin')
        self.validate(state, completion, model, processor, expected)

    def test_wrong_checkpoint_hash_or_source_is_rejected(self):
        for key, value in [('checkpoint_sha256', 'c'*64), ('source_bundle_sha256', 'c'*64)]:
            state, completion, model, processor, config = self.records()
            completion[key] = value
            with self.assertRaises(ValueError):
                self.validate(state, completion, model, processor, config)

    def test_incomplete_or_wrong_update_ledger_rejected(self):
        for change in ('missing_epoch', 'repeated_window', 'count', 'total'):
            state, completion, model, processor, config = self.records()
            if change == 'missing_epoch': state['history'].pop()
            if change == 'repeated_window': state['history'][0]['starts'][0] = 114
            if change == 'count': state['history'][0]['successful_updates'] = 139
            if change == 'total': completion['updates'] = 1119
            with self.assertRaises(ValueError):
                self.validate(state, completion, model, processor, config)

    def test_changed_training_math_or_normalization_rejected(self):
        state, completion, model, processor, config = self.records()
        expected = dict(config, lr=1e-3)
        with self.assertRaises(ValueError): self.validate(state, completion, model, processor, expected)
        with self.assertRaises(ValueError): self.validate(state, completion, model, {'scale': 2}, config)

    def test_smoke_or_unfinished_validation_is_not_a_formal_completion(self):
        for change in ('smoke', 'status', 'holdout'):
            state, completion, model, processor, config = self.records()
            if change == 'smoke': completion['mode'] = 'smoke'
            if change == 'status': completion['status'] = 'running'
            if change == 'holdout': completion['evaluated_starts'].pop()
            with self.assertRaises(ValueError): self.validate(state, completion, model, processor, config)


class VisualExportTests(unittest.TestCase):
    def test_rgb_alpha_crop_and_all_horizons(self):
        rgb = np.zeros((3, 720, 1280, 3), dtype=np.uint8)
        alpha = np.zeros((3, 720, 1280), dtype=np.uint8)
        alpha[:, 100:120, 200:230] = 255
        rgb[:, 100:120, 200:230] = [210, 20, 140]
        self.assertEqual(e.roi_for([(rgb[0], alpha[0])]), (160, 60, 270, 160))
        frames = [(rgb, alpha)]*15
        visual = {k: frames for k in ('reference', 'soma_e10', 'g5_e8')}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root/'comparison_assets').mkdir()
            e.save_visual(root, 1, 181, ['cam0', 'cam1', 'cam2'], visual)
            files = list((root/'comparison_assets').glob('*.webp'))
            self.assertEqual(len(files), 45)
            with Image.open(root/'comparison_assets/ep1_start181_cam2_h15.webp') as image:
                self.assertEqual(image.size, (330, 254))


if __name__ == '__main__':
    unittest.main()
