"""Configurable V4 time contracts and safe single-window experiment primitives."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.tests.dataloader.test_dataloader import MemoryReconstruction
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import build_dataset, prepare_batch, save_checkpoint, load_checkpoint
from scripts.training.losses import chamfer_samples


class ConfigurableV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_default_and_four_time_protocols_dataset_processor_and_forward(self):
        self.assertEqual(V4Stage1Predictor().config['horizon'], 10)
        self.assertEqual(V4Stage1Predictor().config['prediction_interval'], 1)
        source = MemoryReconstruction(frames=53)
        for h, delta in ((5, 1), (10, 1), (5, 10), (10, 2)):
            with self.subTest(h=h, delta=delta), source.mounted():
                dataset = build_dataset('001-rope', 'train', source.root, horizon=h, prediction_interval=delta)
                sample = dataset[0]
                self.assertEqual(sample['meta']['frame_ids'].tolist(), list(range(0, (h+1)*delta, delta)))
                self.assertEqual(sample['meta']['prediction_interval'], delta)
                self.assertEqual(sample['inputs']['action_state'][:, 1, 0].tolist(), list(range(0, h*delta, delta)))
                self.assertTrue(torch.all(sample['meta']['observation_timestamp_us'].diff() == delta*100000))
                self.assertEqual([float(v[0, 1]) for v in sample['targets']['target_xyz']], list(range(delta, (h+1)*delta, delta)))
                model = V4Stage1Predictor(h, delta)
                batch = prepare_batch(collate_fn([sample]), DataProcessor('001-rope'), model)
                self.assertEqual(batch['inputs']['condition_frame_ids'].tolist(), [list(range(0, h*delta, delta))])
                pred = model(batch['inputs'])
                self.assertEqual(pred['xyz'].shape[1], h)
                self.assertTrue(torch.isfinite(pred['xyz']).all())

    def test_ten_step_gradients_and_causality(self):
        torch.manual_seed(42)
        model = V4Stage1Predictor()
        torch.nn.init.normal_(model.motion[-1].weight, std=.003)
        batch = prepare_batch(synthetic_batch(64), DataProcessor('001-rope'), model)
        pred, trace = model(batch['inputs'], return_trace=True)
        grads = torch.autograd.grad(pred['xyz'][:, -1].square().mean(),
                                    [s['state'] for s in trace['steps']], retain_graph=True, allow_unused=True)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        # Connected zero/underflow derivatives are distinct from an absent graph.
        loss = chamfer_samples(pred['xyz'], batch, 32).mean(); loss.backward()
        for name in ('motion', 'cross', 'conditions.controller', 'gs_encoder'):
            gs = [p.grad for n, p in model.named_parameters() if n.startswith(name) and p.grad is not None]
            self.assertTrue(gs)
            self.assertGreater(sum(float(g.double().square().sum()) for g in gs), 0.)
        x = copy.deepcopy(batch['inputs']); x['controller_points'][:, 6:] += .1; x['tactile'][:, 6:] += .2
        torch.testing.assert_close(model(x)['xyz'][:, :6], pred['xyz'][:, :6], rtol=0, atol=0)

    def test_checkpoint_time_mismatch_and_atomic_failure(self):
        model = V4Stage1Predictor(); opt = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            with patch('scripts.training.v4_runtime.torch.save', side_effect=RuntimeError('interrupted write')):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(path, model, opt, 0, source_sha='test', processor_state={})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            save_checkpoint(path, model, opt, 0, source_sha='test', processor_state={}, run_state={'run_id': 'a'})
            original = path.read_bytes()
            for h, delta in ((5, 10), (10, 2), (5, 1)):
                with self.assertRaises(ValueError): load_checkpoint(path, V4Stage1Predictor(h, delta))
            with self.assertRaises(ValueError):
                save_checkpoint(path, model, opt, 1, source_sha='test', processor_state={},
                                replace_existing=True, run_state={'run_id': 'different'})
            self.assertEqual(path.read_bytes(), original)
            with patch('scripts.training.v4_runtime.torch.save', side_effect=RuntimeError('interrupted write')):
                with self.assertRaises(RuntimeError):
                    save_checkpoint(path, model, opt, 1, source_sha='test', processor_state={},
                                    replace_existing=True, run_state={'run_id': 'a'})
            self.assertEqual(path.read_bytes(), original)

    def test_legacy_v4_h5_checkpoint_exact_regression(self):
        # The previous frozen source can still instantiate and save H5/delta10.
        import subprocess
        source = subprocess.check_output(['git', 'show', '8af4c409:scripts/training/v4_model.py'], text=True)
        namespace = {'__name__': 'scripts.training._frozen_v4', '__package__': 'scripts.training'}
        exec(compile(source, 'frozen_8af4c409_v4_model.py', 'exec'), namespace)
        torch.manual_seed(8); old = namespace['V4Stage1Predictor']()
        torch.manual_seed(8); new = V4Stage1Predictor(5, 10)
        self.assertEqual(old.config, new.config)
        batch = prepare_batch(synthetic_batch(64, 5, 10), DataProcessor('001-rope'), new)
        for k, v in old.state_dict().items(): torch.testing.assert_close(v, new.state_dict()[k], rtol=0, atol=0)
        torch.nn.init.normal_(old.motion[-1].weight, std=.003); new.load_state_dict(old.state_dict())
        torch.testing.assert_close(old(batch['inputs'])['xyz'], new(batch['inputs'])['xyz'], rtol=0, atol=0)

    def test_successful_update_and_resume_ledger(self):
        from scripts.experiments.v4_stage1.single_window import successful_update, check_resume, frozen_config
        from scripts.training.rendering import GaussianRenderer
        from scripts.experiments.v4_stage1.fixtures import analytic_rasterizer
        model = V4Stage1Predictor()
        processor = DataProcessor('001-rope')
        batch = prepare_batch(synthetic_batch(48), processor, model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        record = successful_update(model, optimizer, batch, processor, GaussianRenderer(analytic_rasterizer))
        self.assertGreater(record['actual_updates']['motion']['update_norm'], 0.)
        self.assertTrue(torch.isfinite(torch.tensor(record['loss'])))
        config = frozen_config('sha', 'data', 'depth')
        state = dict(source_sha='sha', successful_steps=1, run_state=dict(config=config, manifest={'data': 'hash'},
                     run_id='test', updates=[dict(update=1)]))
        self.assertEqual(check_resume(state, config, {'data': 'hash'}, 'test')['updates'][0]['update'], 1)
        for field, value in (('source_sha', 'different'), ('successful_steps', 2)):
            bad = copy.deepcopy(state); bad[field] = value
            with self.assertRaises(ValueError): check_resume(bad, config, {'data': 'hash'}, 'test')
        with self.assertRaises(ValueError): check_resume(state, config, {'data': 'changed'}, 'test')
        with self.assertRaises(ValueError): check_resume(state, dict(config, lr=1e-3), {'data': 'hash'}, 'test')
