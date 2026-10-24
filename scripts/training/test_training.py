"""Disk-free tests: python -B -m unittest scripts.training.test_training -v.

Real gsplat backward smoke is opt-in inside an allocated server GPU job:
TOUCH4D_TEST_CUDA_RENDER=1 python -B -m unittest scripts.training.test_training
"""

import copy
import math
from io import BytesIO
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
import os
import random
import unittest

import numpy as np
import torch

from ..data_process import DataProcessor
from ..data_process.test_data_process import make_batch, add_depth_targets
from .losses import chamfer_samples, depth_samples, nearest_distances, objective
from .model import PositionPredictor, rotation_vector_to_quaternion, quaternion_multiply
from .evaluate import load_model
from .rendering import DepthRenderer, GaussianRenderer, assemble_gaussians
from .train import (EpisodeMetrics, GaussianMetrics, checkpoint_state, evaluate_loader, restore_training,
                    train_epoch, recover_best_score, publish_checkpoints, read_checkpoint, parser,
                    VERSION, LEGACY_VERSION, INDEPENDENT_VERSION, DISPLACEMENT_VERSION,
                    resolve_controller_features, resolve_prediction_mode, resolve_output_mode)
from .visualize import png_bytes, viewer_html, main as visualize_main


def config():
    return dict(scene='001-rope', width=32, heads=4, layers=2, depth_weight=0.,
                chamfer_chunk=2, max_train_batches=0)


def fake_rasterizer(**args):
    # Analytic differentiable mock, not evidence of real CUDA kernel correctness.
    value = args['means'][:, 2].mean()
    depth = value.expand(1, args['height'], args['width'], 1)
    return depth, torch.ones_like(depth), {}


def fake_gaussian_rasterizer(**args):
    """Analytic geometry-sensitive image; real gsplat is tested separately."""
    q = args['quats']
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    rotation = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1,3,3)
    covariance = (rotation * args['scales'].square()[:,None,:]) @ rotation.transpose(-1,-2)
    shape = covariance[:,0,1] + covariance[:,0,2] + covariance[:,1,2]
    geometry = args['means'].mean() + args['scales'].mean() + shape.mean()
    alpha_value = torch.sigmoid(geometry)
    rgb_value = torch.sigmoid(geometry + torch.arange(3, device=q.device)*.1)
    h, width = args['height'], args['width']
    rgb = rgb_value.expand(1,h,width,3)
    depth = args['means'][:,2].mean().expand(1,h,width,1)
    alpha = alpha_value.expand(1,h,width,1)
    return torch.cat((rgb,depth),-1), alpha, {}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.processor = DataProcessor('001-rope')
        self.raw = add_depth_targets(make_batch())
        self.batch = self.processor(self.raw)
        self.model = PositionPredictor(width=32, heads=4, layers=2)

    def activate_head(self):
        # Exercise the learned path; a zero head would make causality and
        # invariance tests pass even if the condition path were broken.
        torch.nn.init.normal_(self.model.position_head[-1].weight, std=.02)

    def test_relative_controller_features_masks_and_scale(self):
        coordinates = torch.arange(5, dtype=torch.float32)[None, :, None, None, None].expand(1, 5, 2, 30, 3).clone() * .125
        valid = torch.ones(1, 5, 2, 30, dtype=torch.bool)
        valid[:, 0, 0, 0] = False  # Missing reference, later coordinates remain usable.
        valid[:, 2, 0, 1] = False  # Missing current sample.
        valid[:, :, 1] = False    # Entire missing hand.
        coordinates[~valid] = float('nan')
        x = dict(controller_points=coordinates, controller_mask=valid)
        u = self.model.controller_inputs(x)
        self.assertEqual(tuple(u.shape), (1, 5, 2, 30, 7))
        self.assertTrue(torch.isfinite(u).all())
        torch.testing.assert_close(u[0, 4, 0, 2], torch.tensor([.5, .5, .5, .5, .5, .5, 1.]))
        torch.testing.assert_close(u[0, 4, 0, 0], torch.tensor([.5, .5, .5, 0., 0., 0., 0.]))
        self.assertTrue(torch.all(u[~valid] == 0))
        self.assertTrue(torch.all(u[:, 0, ..., 3:6] == 0))
        torch.testing.assert_close(u[:, 0, ..., 6], valid[:, 0].float())
        # A common coordinate offset cancels; do not scale the difference twice.
        moved = dict(x, controller_points=coordinates + 7.)
        torch.testing.assert_close(self.model.controller_inputs(moved)[..., 3:], u[..., 3:], atol=0, rtol=0)
        legacy = PositionPredictor(32, 4, 2, controller_features=3)
        torch.testing.assert_close(legacy.controller_inputs(x), u[..., :3], atol=0, rtol=0)
        full = copy.deepcopy(self.batch['inputs'])
        full['controller_mask'][:, 0, 1, 0] = False
        full['controller_mask'][:, 1, 1, 0] = True
        full['controller_points'][:, 0, 1, 0] = float('nan')
        _, token_mask = self.model.encode_conditions(full)
        self.assertTrue(token_mask[:, 1, 2 + 30].all())
        self.assertTrue(torch.all(self.model.controller_inputs(full)[:, 1, 1, 0, 3:] == 0))

    def test_relative_features_static_motion_gradients_and_first_horizon(self):
        self.activate_head()
        x = copy.deepcopy(self.batch['inputs'])
        x['controller_mask'].fill_(True)
        x['action_valid'].fill_(True)
        x['controller_points'] = x['controller_points'][:, :1].expand_as(x['controller_points']).clone()
        x['controller_points'] += torch.arange(5)[None, :, None, None, None] * .1
        before = self.model(x)
        frozen = copy.deepcopy(x)
        for key in ('controller_points', 'action_state'):
            frozen[key] = x[key][:, :1].expand_as(x[key]).clone()
        u = self.model.controller_inputs(frozen)
        self.assertTrue(torch.all(u[..., 3:6] == 0))
        self.assertTrue(torch.all(u[..., 6] == 1))
        torch.testing.assert_close(self.model(frozen)[:, 0], before[:, 0], atol=0, rtol=0)
        before.square().sum().backward()
        gradient = self.model.controller_encoder[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[:, 3:6].abs().sum().item(), 0.)
        legacy = PositionPredictor(32, 4, 2, controller_features=3)
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(self.model) - count(legacy), 4 * 32)

    def test_checkpoint_old_and_new_loading_and_resume(self):
        cases = [(3, LEGACY_VERSION, 'independent'),
                 (3, INDEPENDENT_VERSION, 'independent'), (7, INDEPENDENT_VERSION, 'independent'),
                 (3, DISPLACEMENT_VERSION, 'independent'), (7, DISPLACEMENT_VERSION, 'independent'),
                 (3, DISPLACEMENT_VERSION, 'autoregressive'), (7, DISPLACEMENT_VERSION, 'autoregressive'),
                 (3, VERSION, 'independent'), (7, VERSION, 'independent'),
                 (3, VERSION, 'autoregressive'), (7, VERSION, 'autoregressive')]
        for features, version, mode in cases:
            with self.subTest(features=features, version=version, mode=mode):
                model = PositionPredictor(32, 4, 2, controller_features=features, prediction_mode=mode)
                torch.nn.init.normal_(model.position_head[-1].weight, std=.02)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                # Populate optimizer moments using synthetic, in-memory data.
                def step(m, opt):
                    opt.zero_grad(set_to_none=True)
                    m(self.batch['inputs']).square().sum().backward()
                    opt.step()
                step(model, optimizer)
                state = copy.deepcopy(checkpoint_state(model, optimizer, self.processor, config(), 2, .5, []))
                state['version'] = version
                if version == LEGACY_VERSION:
                    del state['config']['controller_features']
                if version in (LEGACY_VERSION, INDEPENDENT_VERSION):
                    del state['config']['prediction_mode']
                if version != VERSION:
                    del state['config']['output_mode']
                with patch('scripts.training.train.torch.load', return_value=state):
                    loaded, _, cfg = load_model('memory/checkpoint.pt', 'cpu')
                    canonical = read_checkpoint('memory/checkpoint.pt')
                self.assertEqual(cfg['controller_features'], features)
                self.assertEqual('controller_features' in state['config'], version != LEGACY_VERSION)
                self.assertEqual('prediction_mode' in state['config'], version in (DISPLACEMENT_VERSION, VERSION))
                self.assertEqual(cfg['prediction_mode'], mode)
                self.assertEqual(cfg['output_mode'], 'position')
                self.assertEqual(resolve_output_mode(None, state), 'position')
                with self.assertRaisesRegex(ValueError, 'different output_mode'):
                    resolve_output_mode('gs_geometry', state)
                torch.testing.assert_close(loaded(self.batch['inputs']), model(self.batch['inputs']), atol=0, rtol=0)
                self.assertEqual(resolve_controller_features(None, state), features)
                self.assertEqual(resolve_controller_features(features, state), features)
                self.assertEqual(resolve_prediction_mode(None, state), mode)
                self.assertEqual(resolve_prediction_mode(mode, state), mode)
                with self.assertRaisesRegex(ValueError, 'different prediction_mode'):
                    resolve_prediction_mode('autoregressive' if mode == 'independent' else 'independent', state)
                with self.assertRaisesRegex(ValueError, 'different controller_features'):
                    resolve_controller_features(10 - features, state)
                other_opt = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
                restore_training(state, loaded, other_opt)
                step(model, optimizer)
                step(loaded, other_opt)
                for a, b in zip(model.parameters(), loaded.parameters()):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                # An old best may coexist with a newly saved last in the same 3D run.
                resaved = checkpoint_state(loaded, other_opt, self.processor, cfg, 3, .5, [])
                self.assertEqual(resaved['version'], VERSION)
                with patch('pathlib.Path.is_file', return_value=True), \
                     patch('scripts.training.train.read_checkpoint', return_value=canonical):
                    self.assertEqual(recover_best_score(state, 'memory'), .5)
                    self.assertEqual(recover_best_score(resaved, 'memory'), .5)
        args = parser().parse_args(['--scene', '001-rope'])
        self.assertEqual(resolve_controller_features(args.controller_features), 7)
        self.assertEqual(resolve_controller_features(3), 3)
        self.assertEqual(resolve_prediction_mode(args.prediction_mode), 'autoregressive')
        self.assertEqual(resolve_prediction_mode('independent'), 'independent')
        self.assertEqual(resolve_output_mode(args.output_mode), 'gs_geometry')

    def test_checkpoint_feature_mismatches_fail_before_restore(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        state = checkpoint_state(self.model, optimizer, self.processor, config(), 0, .4, [])
        for mutation in ('missing', 'bad_mode', 'wrong_shape', 'legacy_seven'):
            broken = copy.deepcopy(state)
            if mutation == 'missing': del broken['config']['controller_features']
            elif mutation == 'bad_mode': broken['config']['controller_features'] = 4
            elif mutation == 'wrong_shape': broken['model']['controller_encoder.0.weight'] = torch.zeros(32, 3)
            else: broken['version'] = LEGACY_VERSION
            with self.subTest(mutation=mutation), patch('scripts.training.train.torch.load', return_value=broken):
                with self.assertRaises(ValueError): read_checkpoint('memory/broken.pt')
            with patch.object(self.model, 'load_state_dict') as load, patch.object(optimizer, 'load_state_dict') as opt_load:
                with self.assertRaises(ValueError): restore_training(broken, self.model, optimizer)
                load.assert_not_called()
                opt_load.assert_not_called()
        wrong = PositionPredictor(32, 4, 2, controller_features=3)
        with patch.object(wrong, 'load_state_dict') as load, patch.object(optimizer, 'load_state_dict') as opt_load:
            with self.assertRaisesRegex(ValueError, 'architecture differs'):
                restore_training(state, wrong, optimizer)
            load.assert_not_called()
            opt_load.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'disagrees with model'):
            checkpoint_state(self.model, optimizer, self.processor, dict(config(), controller_features=3), 0, .4, [])
        other = copy.deepcopy(state)
        other['config']['controller_features'] = 3
        with patch('pathlib.Path.is_file', return_value=True), \
             patch('scripts.training.train.read_checkpoint', return_value=other):
            with self.assertRaisesRegex(ValueError, 'configurations differ'):
                recover_best_score(state, 'memory')

    def test_checkpoint_prediction_mode_mismatches_fail_before_restore(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        state = checkpoint_state(self.model, optimizer, self.processor, config(), 0, .4, [])
        for mutation in ('missing', 'unknown', 'legacy_autoregressive'):
            broken = copy.deepcopy(state)
            if mutation == 'missing': del broken['config']['prediction_mode']
            elif mutation == 'unknown': broken['config']['prediction_mode'] = 'unknown'
            else: broken['version'] = INDEPENDENT_VERSION
            with self.subTest(mutation=mutation), patch('scripts.training.train.torch.load', return_value=broken):
                with self.assertRaises(ValueError): read_checkpoint('memory/broken.pt')
            with patch.object(self.model, 'load_state_dict') as load, patch.object(optimizer, 'load_state_dict') as opt_load:
                with self.assertRaises(ValueError): restore_training(broken, self.model, optimizer)
                load.assert_not_called(); opt_load.assert_not_called()
        # Same-sized weights must not permit a silent change in rollout semantics.
        wrong = PositionPredictor(32, 4, 2, prediction_mode='independent')
        self.assertEqual({k: tuple(v.shape) for k, v in wrong.state_dict().items()},
                         {k: tuple(v.shape) for k, v in self.model.state_dict().items()})
        with patch.object(wrong, 'load_state_dict') as load, patch.object(optimizer, 'load_state_dict') as opt_load:
            with self.assertRaisesRegex(ValueError, 'architecture differs'):
                restore_training(state, wrong, optimizer)
            load.assert_not_called(); opt_load.assert_not_called()
        other = copy.deepcopy(state)
        other['config']['prediction_mode'] = 'independent'
        with patch('pathlib.Path.is_file', return_value=True), \
             patch('scripts.training.train.read_checkpoint', return_value=other):
            with self.assertRaisesRegex(ValueError, 'configurations differ'):
                recover_best_score(state, 'memory')

    def test_zero_initialized_displacement_and_step_reference(self):
        x = self.batch['inputs']
        keep = x['gs_mask'][:, None].expand(-1, 5, -1)
        initial = x['gs_xyz'][:, None].expand(-1, 5, -1, -1)
        self.assertTrue(torch.all(self.model.position_head[-1].weight == 0))
        self.assertTrue(torch.all(self.model.position_head[-1].bias == 0))
        torch.testing.assert_close(self.model(x), initial, atol=0, rtol=0)
        torch.nn.init.constant_(self.model.position_head[-1].bias, .25)
        out = self.model(x)
        self.assertEqual(tuple(out.shape), (*keep.shape, 3))
        increments = torch.arange(1, 6)[None, :, None, None] * .25
        torch.testing.assert_close(out[keep], (initial+increments)[keep])
        self.assertTrue(torch.all(out[~keep] == 0))
        shifted = copy.deepcopy(x)
        shifted['gs_xyz'] += 100
        torch.testing.assert_close(self.model(shifted)[keep], (initial+100+increments)[keep])
        legacy = PositionPredictor(32, 4, 2, prediction_mode='independent')
        legacy.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(legacy(x)[keep], (initial+.25)[keep], atol=0, rtol=0)

    def test_autoregressive_reencodes_each_state_and_keeps_full_gradient_chain(self):
        self.activate_head()
        x = self.batch['inputs']
        encoded, deltas = [], []
        a = self.model.gs_encoder.register_forward_pre_hook(lambda module, args: encoded.append(args[0]))
        b = self.model.position_head.register_forward_hook(lambda module, args, out: deltas.append(out))
        try:
            prediction = self.model(x)
        finally:
            a.remove(); b.remove()
        self.assertEqual(len(encoded), 5)
        for k, values in enumerate(encoded):
            previous = x['gs_xyz'] if k == 0 else prediction[:, k-1]
            torch.testing.assert_close(values[..., :3], previous, atol=0, rtol=0)
            torch.testing.assert_close(values[..., 3:], x['gs_attr'], atol=0, rtol=0)
        # Only final-step Chamfer supervises this check. Its graph must still
        # reach all previous head calls and all re-encoded predicted states.
        p = prediction[0, -1, x['gs_mask'][0]]
        t = self.batch['targets']['target_xyz'][-1][0, self.batch['targets']['target_mask'][-1][0]]
        final_loss = .5 * (nearest_distances(p, t).mean() + nearest_distances(t, p).mean())
        # encoded[0] is observed input data, not a differentiable prediction.
        gradients = torch.autograd.grad(final_loss, deltas + encoded[1:], retain_graph=True, allow_unused=True)
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)
        final_loss.backward()
        self.assertGreater(self.model.gs_encoder[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_independent_mode_matches_original_forward(self):
        self.activate_head()
        model = PositionPredictor(32, 4, 2, prediction_mode='independent')
        model.load_state_dict(self.model.state_dict())
        x = self.batch['inputs']; keep = x['gs_mask']
        initial = torch.where(keep[..., None], torch.cat((x['gs_xyz'], x['gs_attr']), -1), 0.)
        features = model.gs_encoder(initial)
        tokens, token_valid = model.encode_conditions(x)
        expected = []
        for k in range(5):
            memory = torch.cat((model.null_token.expand(len(initial), -1, -1), tokens[:, :k+1].flatten(1, 2)), 1)
            valid = torch.cat((torch.ones(len(initial), 1, dtype=torch.bool), token_valid[:, :k+1].flatten(1, 2)), 1)
            query = features + model.prediction_time.weight[k]
            for block in model.blocks: query = block(query, memory, valid)
            expected.append(torch.where(keep[..., None], initial[..., :3]+model.position_head(query), 0.))
        torch.testing.assert_close(model(x), torch.stack(expected, 1), atol=0, rtol=0)
        rolled = self.model(x)
        torch.testing.assert_close(rolled[:, 0], expected[0], atol=0, rtol=0)
        self.assertFalse(torch.equal(rolled[:, 1:], torch.stack(expected, 1)[:, 1:]))

    def test_every_horizon_cannot_read_later_conditions(self):
        self.activate_head()
        x = self.batch['inputs']
        original = self.model(x)
        for k in range(4):
            other = copy.deepcopy(x)
            for field in ('action_state', 'controller_points', 'tactile'):
                other[field][:, k+1:] += torch.randn_like(other[field][:, k+1:])*20
            for field in ('action_valid', 'controller_mask', 'valid_mask', 'tactile_spatial_mask'):
                other[field][:, k+1:] = ~other[field][:, k+1:]
            other['tactile_age_us'][:, k+1:] += 10000
            predicted = self.model(other)
            torch.testing.assert_close(predicted[:, :k+1], original[:, :k+1], atol=0, rtol=0)
        changed = copy.deepcopy(x)
        changed['action_state'] += .5
        self.assertFalse(torch.equal(original, self.model(changed)))

    def test_padding_and_invalid_conditions_do_not_affect_real_queries(self):
        self.activate_head()
        x = self.batch['inputs']
        baseline = self.model(x)
        other = copy.deepcopy(x)
        for field in ('gs_xyz', 'gs_attr'):
            other[field][~x['gs_mask']] = float('nan')
        for field, valid in [('action_state', 'action_valid'), ('controller_points', 'controller_mask'), ('tactile', 'valid_mask')]:
            other[field][~x[valid]] = float('nan')
        prediction = self.model(other)
        torch.testing.assert_close(baseline, prediction, atol=0, rtol=0)
        prediction.square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_all_missing_conditions_and_valid_zero_tactile(self):
        self.activate_head()
        x = copy.deepcopy(self.batch['inputs'])
        x['tactile'].zero_()
        _, valid_zero = self.model.encode_conditions(x)
        self.assertTrue(valid_zero[:, :, -4:].any())
        for key in ('action_valid', 'controller_mask', 'valid_mask', 'tactile_spatial_mask'):
            x[key].zero_()
        pred = self.model(x)
        self.assertTrue(torch.isfinite(pred).all())
        pred.sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_point_order_equivariance_and_variable_number_of_queries(self):
        self.activate_head()
        x = self.batch['inputs']
        baseline = self.model(x)
        other = copy.deepcopy(x)
        order = torch.arange(x['gs_xyz'].shape[1]-1, -1, -1)
        for key in ('gs_xyz', 'gs_attr', 'gs_mask'):
            other[key] = other[key][:, order]
        torch.testing.assert_close(self.model(other), baseline[:, :, order])
        for key in ('gs_xyz', 'gs_attr', 'gs_mask'):
            other[key] = other[key][:, :1]
        self.assertEqual(self.model(other).shape[2], 1)

    def test_chamfer_unordered_unequal_counts_and_gradients(self):
        q = torch.tensor([[0.,0,0], [2.,0,0]], requires_grad=True)
        t = torch.tensor([[3.,0,0], [0.,0,0], [0.,0,0]])
        torch.testing.assert_close(nearest_distances(q, t, chunk=1), torch.tensor([0.,1.]))
        nearest_distances(q, t, chunk=1).sum().backward()
        torch.testing.assert_close(q.grad, torch.tensor([[0.,0,0],[-1.,0,0]]))
        b = copy.deepcopy(self.batch)
        prediction = b['inputs']['gs_xyz'][:, None].expand(-1,5,-1,-1).clone().requires_grad_()
        before = chamfer_samples(prediction,b,2)
        for k in range(5):
            order = torch.arange(b['targets']['target_xyz'][k].shape[1]-1,-1,-1)
            b['targets']['target_xyz'][k] = b['targets']['target_xyz'][k][:,order]
            b['targets']['target_mask'][k] = b['targets']['target_mask'][k][:,order]
            b['targets']['target_xyz'][k][~b['targets']['target_mask'][k]]=float('nan')
        torch.testing.assert_close(before,chamfer_samples(prediction,b,1))
        before.mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.all(prediction.grad[~b['inputs']['gs_mask'][:,None].expand(-1,5,-1)]==0))

    def test_dense_identical_sets_zero_for_default_and_other_chunks(self):
        points=torch.ones(64,3)
        points[:,0]+=torch.arange(64)*1e-4
        reference=points.flip(0).clone()
        query=points.clone().requires_grad_()
        for chunk in (2,32,512):
            result=nearest_distances(query,reference,chunk)
            torch.testing.assert_close(result,torch.zeros(64),atol=0,rtol=0)
            result.sum().backward()
            torch.testing.assert_close(query.grad,torch.zeros_like(query),atol=0,rtol=0)
            query.grad=None
        # Near, nonidentical neighbours must also agree with explicit subtraction.
        reference[:,1]+=3e-5
        expected=torch.linalg.vector_norm(query.detach()[:,None]-reference[None],dim=-1).min(1).values
        torch.testing.assert_close(nearest_distances(query,reference),expected,atol=0,rtol=0)
        batch={'inputs':{'gs_xyz':query[None], 'gs_mask':torch.ones(1,64,dtype=torch.bool)},
               'targets':{'target_xyz':[points.flip(0)[None]]*5,
                          'target_mask':[torch.ones(1,64,dtype=torch.bool)]*5}}
        torch.testing.assert_close(chamfer_samples(query[None,None].expand(1,5,-1,-1),batch),
                                   torch.zeros(1),atol=0,rtol=0)

    def test_depth_target_mask_empty_views_and_missing_prediction(self):
        target=torch.ones(2,5,2,2,2)
        valid=torch.zeros_like(target,dtype=torch.bool)
        valid[0,0,0,0,0]=True
        prediction=torch.zeros_like(target,requires_grad=True)
        loss,available=depth_samples(prediction,target,valid)
        torch.testing.assert_close(loss,torch.tensor([1.,0.]))
        self.assertEqual(available.tolist(),[True,False])
        loss.sum().backward()
        self.assertEqual(prediction.grad[0,0,0,0,0].item(),-1.)
        self.assertEqual(prediction.grad[~valid].abs().sum().item(),0.)
        poisoned=target.clone();poisoned[~valid]=float('nan')
        torch.testing.assert_close(depth_samples(prediction,poisoned,valid)[0],loss)

    def test_world_restoration_attributes_and_camera_inverse(self):
        prediction=self.batch['inputs']['gs_xyz'][:,None].expand(-1,5,-1,-1).clone().requires_grad_()
        gs=assemble_gaussians(prediction,self.batch,self.processor,0,0)
        keep=self.raw['inputs']['gs_mask'][0]
        torch.testing.assert_close(gs['means'],self.raw['inputs']['gs_xyz'][0,keep],atol=1e-6,rtol=0)
        torch.testing.assert_close(gs['scales'],self.raw['inputs']['gs_attr'][0,keep,:3].exp())
        torch.testing.assert_close(gs['quats'],self.raw['inputs']['gs_attr'][0,keep,3:7])
        torch.testing.assert_close(gs['opacities'],self.raw['inputs']['gs_attr'][0,keep,7].sigmoid())
        calls=[]
        def record(**args):
            calls.append(args)
            return fake_rasterizer(**args)
        rendered=DepthRenderer(record)(prediction,self.batch,self.processor)
        self.assertEqual(tuple(rendered.shape),tuple(self.raw['targets']['target_depth'].shape))
        torch.testing.assert_close(calls[0]['viewmats'][0],torch.linalg.inv(self.raw['targets']['camera_to_world'][0,0]))
        self.assertEqual(calls[0]['render_mode'],'ED')
        self.assertEqual(len(calls[0]['means']),int(keep.sum()))
        rendered.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(),0.)

    def test_optimizer_step_validation_baseline_and_checkpoint_resume(self):
        cfg=config(); renderer=DepthRenderer(fake_rasterizer)
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        before=self.model.position_head[-1].weight.detach().clone()
        result=train_epoch(self.model,optimizer,[self.raw],self.processor,renderer,cfg,'cpu')
        self.assertTrue(np.isfinite(result['objective']))
        self.assertFalse(torch.equal(before,self.model.position_head[-1].weight))
        report=evaluate_loader(self.model,[self.raw],self.processor,renderer,cfg,'cpu')
        self.assertEqual(set(report),{'model','copy_initial'})
        self.assertTrue(self.model.training)
        buffer=BytesIO()
        torch.save(checkpoint_state(self.model,optimizer,self.processor,cfg,0,report['model']['objective'],[]),buffer)
        expected_random=torch.rand(2)
        # Save an independent snapshot before either model takes its next step.
        buffer.seek(0); saved=torch.load(buffer,weights_only=False)
        other=PositionPredictor(32,4,2)
        other_opt=torch.optim.AdamW(other.parameters(),lr=1e-3)
        restore_training(saved,other,other_opt)
        torch.testing.assert_close(torch.rand(2),expected_random,atol=0,rtol=0)
        train_epoch(self.model,optimizer,[self.raw],self.processor,renderer,cfg,'cpu')
        train_epoch(other,other_opt,[self.raw],self.processor,renderer,cfg,'cpu')
        for a,b in zip(self.model.parameters(),other.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertGreater(self.model.gs_encoder[0].weight.grad.abs().sum().item(), 0.)

    def test_chamfer_default_has_no_depth_gradient(self):
        self.assertEqual(parser().parse_args(['--scene','001-rope']).depth_weight, 0.)
        self.activate_head()
        pred=self.model(self.batch['inputs'])
        def metric_renderer(prediction,batch,processor):
            self.assertFalse(torch.is_grad_enabled())
            return DepthRenderer(fake_rasterizer)(prediction,batch,processor)
        loss,parts=objective(pred,self.batch,self.processor,metric_renderer)
        expected=chamfer_samples(pred,self.batch).mean()
        torch.testing.assert_close(loss,expected,atol=0,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(loss,pred,retain_graph=True)[0],
                                   torch.autograd.grad(expected,pred)[0],atol=0,rtol=0)
        self.assertFalse(parts['depth_m'].requires_grad)

    def test_absolute_checkpoint_rejected_before_loading_weights(self):
        old={'version':'touch4d-position-v0'}
        with patch('scripts.training.train.torch.load',return_value=old):
            with self.assertRaisesRegex(ValueError,'absolute-position V0'):
                read_checkpoint('memory/old.pt')
        optimizer=torch.optim.AdamW(self.model.parameters())
        with patch.object(self.model,'load_state_dict') as load:
            with self.assertRaisesRegex(ValueError,'absolute-position V0'):
                restore_training(old,self.model,optimizer)
            load.assert_not_called()

    def test_episode_equal_weight_and_depth_unavailability(self):
        tracker=EpisodeMetrics(.1,1.)
        tracker.add(torch.tensor([1,1,5]),dict(geometry=torch.tensor([0.,0.,2.]),
            depth_m=torch.tensor([.001,.001,0.]),depth_available=torch.tensor([True,True,False])))
        r=tracker.summary()
        self.assertEqual(r['geometry'],1.)
        self.assertAlmostEqual(r['depth_mae_mm'],1.,places=5)
        self.assertIsNone(r['episodes']['5']['depth_mae_mm'])

    def test_checkpoint_interruption_preserves_better_best_on_resume(self):
        from pathlib import Path
        base=dict(config=config(),processor=self.processor.state_dict(),epoch=0,best=.8)
        files={'last.pt':copy.deepcopy(base),'best.pt':copy.deepcopy(base)}
        newer=dict(base,epoch=1,best=.4)
        def interrupted(path,state):
            if path.name=='last.pt':raise OSError('simulated publication failure')
            files[path.name]=copy.deepcopy(state)
        with patch('scripts.training.train.atomic_checkpoint',side_effect=interrupted):
            with self.assertRaisesRegex(OSError,'publication failure'):
                publish_checkpoints(Path('memory'),newer,True)
        self.assertEqual(files['last.pt']['epoch'],0)
        self.assertEqual(files['best.pt']['best'],.4)
        with patch('pathlib.Path.is_file',return_value=True), \
             patch('scripts.training.train.read_checkpoint',side_effect=lambda p:copy.deepcopy(files[p.name])), \
             redirect_stdout(StringIO()):
            best=recover_best_score(files['last.pt'],Path('memory'))
        self.assertEqual(best,.4)
        def commit(path,state):files[path.name]=copy.deepcopy(state)
        resumed=dict(base,epoch=1,best=min(best,.6))
        with patch('scripts.training.train.atomic_checkpoint',side_effect=commit):
            publish_checkpoints(Path('memory'),resumed,.6<best)
        # Worse rerun cannot replace the .4 model, and last now retains its score.
        self.assertEqual(files['best.pt'],newer)
        self.assertEqual(files['last.pt']['best'],.4)
        with patch('scripts.training.train.atomic_checkpoint',side_effect=commit):
            publish_checkpoints(Path('memory'),dict(resumed,epoch=2,best=.3),True)
        self.assertEqual(files['best.pt']['best'],.3)
        self.assertEqual(files['last.pt']['best'],.3)

    def test_best_checkpoint_recovery_checks_missing_and_incompatible_files(self):
        state=dict(config=config(),processor=self.processor.state_dict(),best=.4)
        with patch('pathlib.Path.is_file',return_value=False):
            with self.assertRaisesRegex(ValueError,'missing best'):
                recover_best_score(state,'memory')
        for mutation in ('config','processor','worse','nan'):
            other=copy.deepcopy(state)
            if mutation=='config':other['config']['depth_weight']=2.
            elif mutation=='processor':other['processor']['scale']*=2
            elif mutation=='worse':other['best']=.5
            else:other['best']=float('nan')
            with self.subTest(mutation=mutation), patch('pathlib.Path.is_file',return_value=True), \
                 patch('scripts.training.train.read_checkpoint',return_value=other):
                with self.assertRaisesRegex(ValueError,'Cannot safely resume'):
                    recover_best_score(state,'memory')
        other=copy.deepcopy(state)
        other['config']['epochs']=40
        # Increasing total epochs is compatible; equal best is a normal resume.
        with patch('pathlib.Path.is_file',return_value=True), \
             patch('scripts.training.train.read_checkpoint',return_value=other):
            self.assertEqual(recover_best_score(state,'memory'),.4)

    def test_empty_depth_sample_does_not_dilute_objective(self):
        batch=copy.deepcopy(self.batch)
        batch['targets']['target_depth_valid'][1]=False
        pred=self.model(batch['inputs'])
        loss,parts=objective(pred,batch,self.processor,DepthRenderer(fake_rasterizer),1.,2)
        torch.testing.assert_close(loss,parts['geometry'].mean()+parts['depth_m'][0]/self.processor.state_dict()['scale'])

    def test_html_is_offline_and_png_is_valid_container(self):
        image=png_bytes(np.zeros((2,3,3),dtype=np.uint8))
        self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
        text=viewer_html({'initial':[]},'<script>bad</script>')
        self.assertNotIn('<script>bad</script>',text)
        self.assertNotIn('https://',text)
        self.assertIn('GS 中心点',text)

    def test_visualization_export_uses_real_counts_and_world_coordinates(self):
        # Exercise the full exporter with no filesystem writes and no GPU.
        cfg=config(); sample=object()
        ds=SimpleNamespace(windows=[(3,0)])
        class Dataset:
            windows=ds.windows
            def __getitem__(self,index):return sample
        captured={}
        def capture(path,**arrays):captured.update(arrays)
        fake=DepthRenderer(fake_rasterizer)
        with patch('scripts.training.visualize.load_model',return_value=(self.model,self.processor,cfg)), \
             patch('scripts.training.visualize.build_loader',return_value=SimpleNamespace(dataset=Dataset())), \
             patch('scripts.training.visualize.collate_fn',return_value=self.raw), \
             patch('scripts.training.visualize.DepthRenderer',return_value=fake), \
             patch('scripts.training.visualize.np.savez_compressed',side_effect=capture), \
             patch('pathlib.Path.mkdir'), patch('pathlib.Path.write_bytes'), patch('pathlib.Path.write_text'), \
             redirect_stdout(StringIO()):
            visualize_main(['--checkpoint','memory/last.pt','--episode','3','--start','0','--device','cpu'])
        count=int(self.raw['inputs']['gs_mask'][0].sum())
        self.assertEqual(captured['predicted_xyz'].shape,(5,count,3))
        np.testing.assert_array_equal(captured['initial_xyz'],self.raw['inputs']['gs_xyz'][0,:count].numpy())
        self.assertEqual(len(captured['target_xyz_1']),int(self.raw['targets']['target_lengths'][0,0]))

    def test_nonfinite_prediction_fails_before_render(self):
        pred=self.model(self.batch['inputs'])
        pred[0,0,0,0]=float('nan')
        def forbidden(*args):raise AssertionError('Renderer must not be called')
        with self.assertRaisesRegex(FloatingPointError,'predicted position'):
            objective(pred,self.batch,self.processor,forbidden)

    def geometry_fixture(self):
        model = PositionPredictor(32,4,2,output_mode='gs_geometry')
        batch = copy.deepcopy(self.batch)
        # Unequal principal scales make orientation observable.
        keep = batch['inputs']['gs_mask']
        batch['inputs']['gs_attr'][...,:3] = torch.tensor([-1.5,-2.,-2.5])
        batch['inputs']['gs_attr'][~keep] = 0
        if 'target_attr' not in batch['targets']:
            batch['targets']['target_attr'] = []
            for xyz, valid in zip(batch['targets']['target_xyz'], batch['targets']['target_mask']):
                attr = torch.zeros(*xyz.shape[:-1],11)
                attr[...,:3] = torch.tensor([-1.5,-2.,-2.5])
                attr[...,3] = 1
                attr[~valid] = 0
                batch['targets']['target_attr'].append(attr)
        return model, batch

    def activate_geometry_heads(self, model):
        for head in (model.position_head, model.scale_head, model.rotation_head):
            torch.nn.init.normal_(head[-1].weight,std=.02)

    def test_rotation_vector_zero_derivative_and_world_left_composition(self):
        zero = torch.zeros(3,requires_grad=True)
        q = rotation_vector_to_quaternion(zero)
        torch.testing.assert_close(q,torch.tensor([1.,0,0,0]),atol=0,rtol=0)
        jacobian = torch.autograd.functional.jacobian(rotation_vector_to_quaternion,zero)
        self.assertTrue(torch.isfinite(jacobian).all())
        torch.testing.assert_close(jacobian,torch.cat((torch.zeros(1,3),.5*torch.eye(3))),atol=1e-7,rtol=0)
        qx = rotation_vector_to_quaternion(torch.tensor([math.pi/2,0.,0.]))
        qz = rotation_vector_to_quaternion(torch.tensor([0.,0.,math.pi/2]))
        torch.testing.assert_close(quaternion_multiply(qz,qx),torch.full((4,),.5),atol=1e-6,rtol=0)
        # Opposite order gives a different rotation; detect an accidental right product.
        self.assertFalse(torch.allclose(quaternion_multiply(qx,qz),torch.full((4,),.5)))

    def test_geometry_zero_identity_and_increment_accumulation(self):
        model,batch = self.geometry_fixture()
        x=batch['inputs']; keep=x['gs_mask'][:,None].expand(-1,5,-1)
        initial = {'xyz':x['gs_xyz'],'log_scales':x['gs_attr'][...,:3],
                   'quaternions':x['gs_attr'][...,3:7], 'opacity_logits':x['gs_attr'][...,7:8],
                   'sh_dc':x['gs_attr'][...,8:11]}
        output=model(x)
        for head in (model.position_head,model.scale_head,model.rotation_head):
            self.assertTrue(torch.all(head[-1].weight==0))
            self.assertTrue(torch.all(head[-1].bias==0))
        for key,value in initial.items():
            torch.testing.assert_close(output[key],value[:,None].expand_as(output[key]),atol=0,rtol=0)
        with torch.no_grad():
            model.position_head[-1].bias.fill_(.25)
            model.scale_head[-1].bias.copy_(torch.tensor([.1,-.1,.2]))
            model.rotation_head[-1].bias.copy_(torch.tensor([0.,0.,math.pi/12]))
        output=model(x)
        steps=torch.arange(1,6)[None,:,None,None]
        torch.testing.assert_close(output['xyz'][keep],(initial['xyz'][:,None]+.25*steps)[keep])
        expected_scale=initial['log_scales'][:,None]+steps*torch.tensor([.1,-.1,.2])
        torch.testing.assert_close(output['log_scales'][keep],expected_scale[keep])
        delta=rotation_vector_to_quaternion(torch.tensor([0.,0.,math.pi/12]))
        expected=initial['quaternions']
        for k in range(5):
            expected=quaternion_multiply(delta.expand_as(expected),expected)
            torch.testing.assert_close(output['quaternions'][:,k][x['gs_mask']],expected[x['gs_mask']],atol=1e-6,rtol=0)
        for key in ('opacity_logits','sh_dc'):
            torch.testing.assert_close(output[key],initial[key][:,None].expand_as(output[key]),atol=0,rtol=0)
        for value in output.values():
            self.assertTrue(torch.all(value[~keep]==0))
        # Changing orientation does not implicitly rotate the Gaussian center.
        with self.assertRaises(ValueError):
            PositionPredictor(32,4,2,output_mode='gs_geometry',prediction_mode='independent')

    def test_geometry_reencodes_updated_attributes_and_full_cross_step_gradients(self):
        model,batch=self.geometry_fixture();self.activate_geometry_heads(model)
        encoded=[];increments={key:[] for key in ('xyz','scale','rotation')}
        hooks=[model.gs_encoder.register_forward_pre_hook(lambda _,args:encoded.append(args[0]))]
        for name,head in zip(increments,(model.position_head,model.scale_head,model.rotation_head)):
            hooks.append(head.register_forward_hook(lambda _,args,out,key=name:increments[key].append(out)))
        try:out=model(batch['inputs'])
        finally:
            for hook in hooks:hook.remove()
        keep=batch['inputs']['gs_mask']
        self.assertEqual(len(encoded),5)
        for k in range(1,5):
            torch.testing.assert_close(encoded[k][...,:3],out['xyz'][:,k-1],atol=0,rtol=0)
            torch.testing.assert_close(encoded[k][...,3:6],out['log_scales'][:,k-1],atol=0,rtol=0)
            torch.testing.assert_close(encoded[k][...,6:10],out['quaternions'][:,k-1],atol=0,rtol=0)
        # Last-step supervision must reach every earlier geometric update.
        loss=out['xyz'][:,-1][keep].square().sum()+out['log_scales'][:,-1][keep].square().sum()
        loss+=(out['quaternions'][:,-1][keep]*torch.tensor([.2,.3,.5,.7])).sum()
        nodes=[v for values in increments.values() for v in values]+encoded[1:]
        grads=torch.autograd.grad(loss,nodes,retain_graph=True,allow_unused=True)
        for gradient in grads:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(),0.)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_geometry_causality_padding_and_point_order(self):
        model,batch=self.geometry_fixture();self.activate_geometry_heads(model)
        x=batch['inputs'];base=model(x)
        for k in range(4):
            other=copy.deepcopy(x)
            for key in ('action_state','controller_points','tactile'):
                other[key][:,k+1:]+=torch.randn_like(other[key][:,k+1:])*20
            for key in ('action_valid','controller_mask','valid_mask','tactile_spatial_mask'):
                other[key][:,k+1:]=~other[key][:,k+1:]
            other['tactile_age_us'][:,k+1:]+=10000
            result=model(other)
            for key in base:
                torch.testing.assert_close(result[key][:,:k+1],base[key][:,:k+1],atol=0,rtol=0)
        poison=copy.deepcopy(x)
        for key in ('gs_xyz','gs_attr'):poison[key][~x['gs_mask']]=float('nan')
        for key,mask in (('action_state','action_valid'),('controller_points','controller_mask'),('tactile','valid_mask')):
            poison[key][~x[mask]]=float('nan')
        result=model(poison)
        for key in base:torch.testing.assert_close(result[key],base[key],atol=0,rtol=0)
        result['quaternions'].sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        order=torch.arange(x['gs_xyz'].shape[1]-1,-1,-1)
        shuffled=copy.deepcopy(x)
        for key in ('gs_xyz','gs_attr','gs_mask'):shuffled[key]=shuffled[key][:,order]
        for key,value in model(shuffled).items():
            torch.testing.assert_close(value,base[key][:,:,order])

    def test_geometry_renderer_live_attributes_target_stopgrad_and_camera(self):
        model,batch=self.geometry_fixture()
        out={key:value.detach().requires_grad_() for key,value in model(batch['inputs']).items()}
        for value in batch['targets']['target_attr']:value.requires_grad_()
        for value in batch['targets']['target_xyz']:value.requires_grad_()
        calls=[]
        def record(**kwargs):
            calls.append((kwargs,torch.is_grad_enabled()))
            return fake_gaussian_rasterizer(**kwargs)
        renderer=GaussianRenderer(record)
        rendered=renderer(out,batch,self.processor)
        first=calls[0][0];keep=batch['inputs']['gs_mask'][0]
        self.assertEqual(first['render_mode'],'RGB+ED')
        torch.testing.assert_close(first['backgrounds'],torch.zeros(1,3),atol=0,rtol=0)
        expected_colors=(batch['inputs']['gs_attr'][0,keep,8:11]*.28209479177387814+.5).clamp(0,1)
        torch.testing.assert_close(first['colors'],expected_colors,atol=0,rtol=0)
        torch.testing.assert_close(first['viewmats'][0],torch.linalg.inv(batch['targets']['camera_to_world'][0,0]))
        torch.testing.assert_close(first['scales'],out['log_scales'][0,0,keep].exp()*self.processor.state_dict()['scale'])
        self.assertEqual(tuple(rendered['rgb'].shape),(*batch['targets']['target_depth'].shape,3))
        self.assertEqual(tuple(rendered['alpha'].shape),tuple(batch['targets']['target_depth'].shape))
        self.assertEqual(len(first['means']),int(keep.sum()))
        target=renderer.render_targets(batch,self.processor)
        self.assertTrue(all(not value.requires_grad for value in target.values()))
        self.assertTrue(any(not enabled for _,enabled in calls))
        (rendered['rgb'].sum()+rendered['alpha'].sum()).backward()
        for key in ('xyz','log_scales','quaternions'):
            self.assertTrue(torch.isfinite(out[key].grad).all())
            self.assertGreater(out[key].grad.abs().sum().item(),0.)
            self.assertTrue(torch.all(out[key].grad[~batch['inputs']['gs_mask'][:,None].expand(-1,5,-1)]==0))
        self.assertTrue(all(value.grad is None for value in batch['targets']['target_attr']))
        self.assertTrue(all(value.grad is None for value in batch['targets']['target_xyz']))
        # Recorded full-resolution calibration cannot silently render a batch
        # at another size without correspondingly changing the intrinsics.
        full_resolution=GaussianRenderer(fake_gaussian_rasterizer,image_size=(720,1280))
        with self.assertRaisesRegex(ValueError,'resolution'):
            full_resolution(out,batch,self.processor)
        with self.assertRaisesRegex(ValueError,'resolution'):
            full_resolution.render_targets(batch,self.processor)

    def test_geometry_objective_uses_rgb_alpha_gradients_with_depth_disabled(self):
        model,batch=self.geometry_fixture();self.activate_geometry_heads(model)
        prediction=model(batch['inputs'])
        loss,parts=objective(prediction,batch,self.processor,GaussianRenderer(fake_gaussian_rasterizer),
                             depth_weight=0.,chunk=2,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)
        expected=(parts['rgb']+.1*parts['alpha']+.01*parts['scale_regularization'])[parts['render_available']].mean()
        torch.testing.assert_close(loss,expected)
        self.assertFalse(parts['geometry'].requires_grad)
        self.assertFalse(parts['depth_m'].requires_grad)
        self.assertTrue(parts['rgb'].requires_grad)
        self.assertTrue(parts['alpha'].requires_grad)
        loss.backward()
        for head in (model.position_head,model.scale_head,model.rotation_head):
            grad=head[-1].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0.)

    def test_geometry_render_mask_equal_averaging_and_empty_samples(self):
        model,batch=self.geometry_fixture()
        pred=model(batch['inputs'])
        pred['log_scales']=pred['log_scales']+.2
        valid=batch['targets']['target_depth_valid']
        valid.zero_()
        # Pixel counts differ by view and time. Each valid view/time is equal,
        # so the expected RGB average is mean(mean(2,10),20)=13, not 8.5.
        pixels=[(0,0,0,0,0),(0,0,0,0,1),(0,0,1,0,0),(0,1,0,0,0)]
        rgb=torch.zeros(*valid.shape,3,requires_grad=True)
        alpha=torch.full(valid.shape,.25,requires_grad=True)
        with torch.no_grad():
            for index,value in zip(pixels,(1.,3.,10.,20.)):
                valid[index]=True;rgb[index]=value
        target_rgb=torch.zeros_like(rgb);target_alpha=torch.zeros_like(alpha)
        target_rgb[~valid]=99.;target_alpha[~valid]=1.
        class Renderer:
            def __call__(self,*_):return dict(rgb=rgb,alpha=alpha,depth=torch.zeros_like(alpha))
            def render_targets(self,*_):return dict(rgb=target_rgb,alpha=target_alpha,depth=torch.zeros_like(alpha))
        loss,parts=objective(pred,batch,self.processor,Renderer(),chunk=2)
        self.assertEqual(parts['render_available'].tolist(),[True,False])
        torch.testing.assert_close(parts['rgb'],torch.tensor([13.,0.]))
        torch.testing.assert_close(parts['rgb_by_horizon'][0],torch.tensor([6.,20.,0.,0.,0.]))
        self.assertEqual(parts['render_time_available'][0].tolist(),[True,True,False,False,False])
        self.assertAlmostEqual(loss.item(),13.+.1*.25+.01*.04,places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertEqual(rgb.grad[~valid].abs().sum().item(),0.)
        self.assertEqual(alpha.grad[~valid].abs().sum().item(),0.)
        self.assertGreater(rgb.grad[valid].abs().sum().item(),0.)
        # Empty samples also exclude scale regularization from the objective.
        valid.zero_();rgb.grad=None;alpha.grad=None
        empty,parts=objective(model(batch['inputs']),batch,self.processor,Renderer(),chunk=2)
        self.assertEqual(empty.item(),0.)
        self.assertFalse(parts['render_available'].any())
        empty.backward()
        self.assertTrue(rgb.grad is None or torch.all(rgb.grad==0))
        self.assertTrue(alpha.grad is None or torch.all(alpha.grad==0))

    def test_geometry_all_empty_batch_skips_optimizer_and_validation_rejects(self):
        model,_=self.geometry_fixture()
        raw=copy.deepcopy(self.raw)
        raw['targets']['target_depth'].zero_()
        raw['targets']['target_depth_valid'].zero_()
        cfg=dict(config(),**model.config,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.1)
        before={key:value.clone() for key,value in model.state_dict().items()}
        # A training epoch with no observations must not silently be considered
        # successful; crucially there must be no AdamW decay-only update either.
        with patch.object(optimizer,'step',wraps=optimizer.step) as step:
            with self.assertRaises(ValueError):
                train_epoch(model,optimizer,[raw],self.processor,GaussianRenderer(fake_gaussian_rasterizer),cfg,'cpu')
            step.assert_not_called()
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        with self.assertRaises(ValueError):
            evaluate_loader(model,[raw],self.processor,GaussianRenderer(fake_gaussian_rasterizer),cfg,'cpu')

    def test_geometry_visualization_exports_properties_and_real_renders(self):
        model,_=self.geometry_fixture();self.activate_geometry_heads(model)
        cfg=dict(config(),**model.config,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)
        sample=object();captured={};written={}
        class Dataset:
            windows=[(3,0)]
            def __getitem__(self,index):return sample
        def capture(path,**arrays):captured.update(arrays)
        def capture_text(path,text,**kwargs):written[str(path)]=text
        fake=GaussianRenderer(fake_gaussian_rasterizer)
        with patch('scripts.training.visualize.load_model',return_value=(model,self.processor,cfg)), \
             patch('scripts.training.visualize.build_loader',return_value=SimpleNamespace(dataset=Dataset())), \
             patch('scripts.training.visualize.collate_fn',return_value=self.raw), \
             patch('scripts.training.visualize.GaussianRenderer',return_value=fake), \
             patch('scripts.training.visualize.np.savez_compressed',side_effect=capture), \
             patch('pathlib.Path.mkdir'),patch('pathlib.Path.write_bytes'), \
             patch('pathlib.Path.write_text',autospec=True,side_effect=capture_text), \
             redirect_stdout(StringIO()):
            visualize_main(['--checkpoint','memory/gs.pt','--episode','3','--start','0','--device','cpu'])
        count=int(self.raw['inputs']['gs_mask'][0].sum())
        for key,dim in (('xyz',3),('log_scales',3),('quaternions',4),('opacity_logits',1),('sh_dc',3)):
            self.assertEqual(captured['predicted_'+key].shape,(5,count,dim))
        expected=model(self.batch['inputs'])
        world_scale=expected['log_scales'][0,:,:count].detach().numpy()+math.log(self.processor.state_dict()['scale'])
        np.testing.assert_allclose(captured['predicted_log_scales'],world_scale,atol=1e-6,rtol=0)
        self.assertEqual(captured['target_attr_1'].shape[0],int(self.raw['targets']['target_lengths'][0,0]))
        text='\n'.join(written.values())
        self.assertIn('RGB',text)
        self.assertIn('alpha',text)
        self.assertNotIn('https://',text)

    @unittest.skipUnless(os.environ.get('TOUCH4D_TEST_CUDA_RENDER')=='1', 'Real gsplat CUDA backward test requires explicit server GPU run')
    def test_real_gsplat_rgb_alpha_geometry_backward(self):
        from .train import to_device
        self.assertTrue(torch.cuda.is_available())
        raw=copy.deepcopy(self.raw);keep=raw['inputs']['gs_mask']
        raw['inputs']['gs_xyz'].zero_();raw['inputs']['gs_xyz'][...,2]=.6
        raw['inputs']['gs_xyz'][~keep]=0
        raw['inputs']['gs_attr'][...,:3]=torch.tensor([-3.,-3.5,-4.])
        raw['inputs']['gs_attr'][...,3:7]=torch.tensor([.9238795,0.,0.,.3826834])
        raw['inputs']['gs_attr'][~keep]=0
        raw['targets']['camera_to_world']=torch.eye(4).repeat(2,2,1,1)
        batch=self.processor(to_device(raw,'cuda'))
        model=PositionPredictor(32,4,2,output_mode='gs_geometry').cuda()
        output={key:value.detach().requires_grad_() for key,value in model(batch['inputs']).items()}
        rendered=GaussianRenderer()(output,batch,self.processor)
        h,w=rendered['alpha'].shape[-2:]
        weights=torch.arange(h*w,device='cuda').reshape(h,w).float()+1
        loss=(rendered['rgb']*weights[...,None]).sum()+.1*(rendered['alpha']*weights).sum()
        loss.backward()
        for key in ('xyz','log_scales','quaternions'):
            grad=output[key].grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0.)

    def test_geometry_checkpoint_roundtrip_and_output_mode_conflicts(self):
        model,batch=self.geometry_fixture();self.activate_geometry_heads(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        def step(network,opt):
            opt.zero_grad(set_to_none=True)
            loss,_=objective(network(batch['inputs']),batch,self.processor,GaussianRenderer(fake_gaussian_rasterizer),chunk=2)
            loss.backward();opt.step()
        step(model,optimizer)
        cfg=dict(config(),**model.config,rgb_weight=1.,alpha_weight=.1,scale_weight=.01)
        state=copy.deepcopy(checkpoint_state(model,optimizer,self.processor,cfg,3,.2,[]))
        self.assertEqual(state['version'],VERSION)
        self.assertEqual(state['config']['render_settings']['width'],1280)
        self.assertEqual(state['config']['render_settings']['height'],720)
        self.assertEqual(state['config']['render_settings']['background'],[0.,0.,0.])
        with patch('scripts.training.train.torch.load',return_value=state):
            loaded,_,loaded_cfg=load_model('memory/geometry.pt','cpu')
        self.assertEqual(loaded_cfg['output_mode'],'gs_geometry')
        expected=model(batch['inputs']);actual=loaded(batch['inputs'])
        for key in expected:torch.testing.assert_close(expected[key],actual[key],atol=0,rtol=0)
        other_opt=torch.optim.AdamW(loaded.parameters(),lr=1e-3)
        restore_training(state,loaded,other_opt)
        step(model,optimizer);step(loaded,other_opt)
        for a,b in zip(model.parameters(),loaded.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        legacy=PositionPredictor(32,4,2)
        with patch.object(legacy,'load_state_dict') as load:
            with self.assertRaisesRegex(ValueError,'architecture differs'):
                restore_training(state,legacy,optimizer)
            load.assert_not_called()
        for mutation in ('missing','unknown','legacy_version'):
            broken=copy.deepcopy(state)
            if mutation=='missing':del broken['config']['output_mode']
            elif mutation=='unknown':broken['config']['output_mode']='unknown'
            else:broken['version']=DISPLACEMENT_VERSION
            with self.subTest(mutation=mutation),patch('scripts.training.train.torch.load',return_value=broken):
                with self.assertRaises(ValueError):read_checkpoint('memory/broken.pt')
        for mutation in ('missing','resolution','background','rasterizer'):
            broken=copy.deepcopy(state)
            if mutation=='missing':del broken['config']['render_settings']
            elif mutation=='resolution':broken['config']['render_settings']['width']=640
            elif mutation=='background':broken['config']['render_settings']['background']=[1.,1.,1.]
            else:broken['config']['render_settings']['eps2d']=.1
            with self.subTest(render_settings=mutation),patch('scripts.training.train.torch.load',return_value=broken):
                with self.assertRaisesRegex(ValueError,'render settings'):
                    read_checkpoint('memory/broken.pt')
        incompatible=copy.deepcopy(state)
        incompatible['config']['alpha_weight']=.2
        with patch('pathlib.Path.is_file',return_value=True),patch('scripts.training.train.read_checkpoint',return_value=incompatible):
            with self.assertRaisesRegex(ValueError,'configurations differ'):
                recover_best_score(state,'memory')

    def test_geometry_validation_score_excludes_regularizer_and_weights_episodes(self):
        tracker=GaussianMetrics(.1,dict(rgb_weight=1.,alpha_weight=.1,scale_weight=.01))
        values=dict(geometry=torch.tensor([100.,100.,100.]),depth_m=torch.zeros(3),
            depth_available=torch.ones(3,dtype=torch.bool),render_available=torch.ones(3,dtype=torch.bool),
            rgb=torch.tensor([0.,0.,2.]),alpha=torch.tensor([.1,.1,.5]),
            scale_regularization=torch.tensor([30.,30.,50.]),
            rgb_by_horizon=torch.tensor([[0.]*5,[0.]*5,[2.]*5]),
            alpha_by_horizon=torch.tensor([[.1]*5,[.1]*5,[.5]*5]),
            render_time_available=torch.ones(3,5,dtype=torch.bool))
        tracker.add(torch.tensor([1,1,5]),values)
        result=tracker.summary()
        self.assertAlmostEqual(result['rgb'],1.)
        self.assertAlmostEqual(result['alpha'],.3)
        self.assertAlmostEqual(result['val_score'],1.03)
        self.assertAlmostEqual(result['objective'],1.03)
        self.assertAlmostEqual(result['training_objective'],1.43)
        self.assertEqual(result['rgb_by_horizon'],[1.]*5)

    @unittest.skipUnless(os.environ.get('TOUCH4D_TEST_CUDA_RENDER')=='1', 'Real gsplat CUDA backward test requires explicit server GPU run')
    def test_real_gsplat_depth_backward(self):
        from .train import to_device
        self.assertTrue(torch.cuda.is_available())
        raw=copy.deepcopy(self.raw)
        raw['inputs']['gs_xyz'].zero_()
        raw['inputs']['gs_xyz'][...,2]=.6
        raw['inputs']['gs_xyz'][~raw['inputs']['gs_mask']]=0
        raw['inputs']['gs_attr'][...,:3]=-3.
        raw['inputs']['gs_attr'][~raw['inputs']['gs_mask']]=0
        raw['targets']['camera_to_world']=torch.eye(4).repeat(2,2,1,1)
        batch=self.processor(to_device(raw,'cuda'))
        pred=batch['inputs']['gs_xyz'][:,None].expand(-1,5,-1,-1).clone().requires_grad_()
        depth=DepthRenderer()(pred,batch,self.processor)
        error,_=depth_samples(depth,batch['targets']['target_depth'],batch['targets']['target_depth_valid'])
        error.sum().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(),0.)


if __name__=='__main__':
    unittest.main()
