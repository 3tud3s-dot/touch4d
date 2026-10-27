"""CPU-only validation of reversible loss/optimizer audit instrumentation."""
import copy
import json
import unittest
from unittest.mock import patch

import torch

from scripts.data_process import DataProcessor
from scripts.experiments.debug_audit.gradients import audit_gradients, cosine
from scripts.experiments.spatial_dynamics.conditions import hand_inputs
from scripts.experiments.spatial_dynamics.runner import paired_model
from scripts.tests.data_process.test_data_process import make_batch, add_depth_targets
from scripts.tests.training.test_training import fake_gaussian_rasterizer
from scripts.training.rendering import GaussianRenderer


class GradientAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setup_case(self):
        processor = DataProcessor('001-rope')
        batch = processor(add_depth_targets(make_batch()))
        batch['inputs'] = hand_inputs(batch['inputs'], torch.tensor([1, 5]), ['001-rope'] * 2)
        model, _ = paired_model('S0', width=16, heads=4, layers=2, controller_features=7,
            horizon=5, gs_local_attention='knn', gs_local_k=3, gs_decoder='anchors',
            anchor_count=4, anchor_pool_k=3, gs_structure='anchor_attention',
            output_mode='gs_geometry')
        torch.nn.init.normal_(model.position_head[-1].weight, std=.01)
        torch.nn.init.normal_(model.fusion[-1].weight, std=.01)
        return model, batch, processor, GaussianRenderer(rasterizer=fake_gaussian_rasterizer)

    def test_cosine_handles_absent_zero_and_opposite(self):
        a = {'a': torch.tensor([1., 2.]), 'b': None}
        self.assertAlmostEqual(cosine(a, a, list(a)), 1.)
        self.assertAlmostEqual(cosine(a, {'a': -a['a'], 'b': None}, list(a)), -1.)
        self.assertIsNone(cosine(a, {'a': torch.zeros(2), 'b': None}, list(a)))

    def test_loss_gradients_updates_and_complete_restore(self):
        model, batch, processor, renderer = self.setup_case()
        model.eval()
        model.position_head.train()
        model.trace = ['sentinel']
        saved = copy.deepcopy(model.state_dict())
        modes = {name: m.training for name, m in model.named_modules()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter) * .1
        rng = torch.get_rng_state().clone()
        result = audit_gradients(model, batch, processor, renderer)
        json.dumps(result, allow_nan=False)
        self.assertTrue(result['model_state_restored_exact'])
        self.assertEqual(result['optimizer_start'], 'fresh_zero_moments')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(model.trace, ['sentinel'])
        self.assertEqual(modes, {name: m.training for name, m in model.named_modules()})
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]), name)
        for parameter in model.parameters():
            self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter) * .1))
        for component in ('cd', 'rgb', 'alpha', 'full'):
            row = result[component]
            modules = row['modules']
            self.assertGreater(modules['anchor_displacement_head']['raw_gradient_norm'], 0)
            self.assertGreater(modules['controller_encoder']['actual_update_norm'], 0)
            self.assertGreater(max(row['prediction_change_mm_by_horizon']), 0)
            self.assertLessEqual(row['global_gradient_norm_after_clip'], 1.00001)
            self.assertGreater(row['global_gradient_norm_before_clip'], 0)
            self.assertEqual(len(row['prediction_change_mm_by_horizon']), 5)
        common = result['full']['unweighted_components']
        self.assertAlmostEqual(result['full']['loss'], common['geometry'] + common['rgb']
            + .1 * common['alpha'] + .01 * common['scale_regularization'], places=6)
        self.assertAlmostEqual(result['cd']['loss'], common['geometry'], places=7)
        self.assertAlmostEqual(result['rgb']['loss'], common['rgb'], places=7)
        self.assertAlmostEqual(result['alpha']['loss'], common['alpha'], places=7)
        self.assertIsNotNone(result['comparisons']['cd_vs_full']['all_parameters']['actual_adamw_update_cosine'])

    def test_restored_moments_are_used_without_mutating_supplied_state(self):
        model, batch, processor, renderer = self.setup_case()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        saved = copy.deepcopy(state)
        result = audit_gradients(model, batch, processor, renderer, state)
        self.assertEqual(result['optimizer_start'], 'checkpoint_moments')
        for key in saved['state']:
            for subkey, value in saved['state'][key].items():
                self.assertTrue(torch.equal(value, state['state'][key][subkey]))
        self.assertEqual(state['param_groups'], saved['param_groups'])

    def test_nonfinite_or_missing_supervision_fails_and_restores(self):
        model, batch, processor, renderer = self.setup_case()
        saved = copy.deepcopy(model.state_dict())
        batch['targets']['target_depth_valid'][:, 1].zero_()
        with self.assertRaisesRegex(ValueError, 'Missing target render supervision'):
            audit_gradients(model, batch, processor, renderer)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]))
        model, batch, processor, renderer = self.setup_case()
        def broken(*args, **kwargs):
            raise FloatingPointError('Injected nonfinite test')
        with patch('scripts.experiments.debug_audit.gradients.objective', broken):
            with self.assertRaisesRegex(FloatingPointError, 'Injected nonfinite'):
                audit_gradients(model, batch, processor, renderer)

    def test_reject_optimizer_setting_drift(self):
        model, batch, processor, renderer = self.setup_case()
        state = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-4).state_dict()
        with self.assertRaisesRegex(ValueError, 'Frozen optimizer settings disagree'):
            audit_gradients(model, batch, processor, renderer, state)


if __name__ == '__main__':
    unittest.main()
