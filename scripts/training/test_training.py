"""Disk-free tests: python -B -m unittest scripts.training.test_training -v.

Real gsplat backward smoke is opt-in inside an allocated server GPU job:
TOUCH4D_TEST_CUDA_RENDER=1 python -B -m unittest scripts.training.test_training
"""

import copy
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
from .model import PositionPredictor
from .evaluate import load_model
from .rendering import DepthRenderer, assemble_gaussians
from .train import (EpisodeMetrics, checkpoint_state, evaluate_loader, restore_training,
                    train_epoch, recover_best_score, publish_checkpoints, read_checkpoint, parser,
                    VERSION, LEGACY_VERSION, INDEPENDENT_VERSION,
                    resolve_controller_features, resolve_prediction_mode)
from .visualize import png_bytes, viewer_html, main as visualize_main


def config():
    return dict(scene='001-rope', width=32, heads=4, layers=2, depth_weight=0.,
                chamfer_chunk=2, max_train_batches=0)


def fake_rasterizer(**args):
    # Analytic differentiable mock, not evidence of real CUDA kernel correctness.
    value = args['means'][:, 2].mean()
    depth = value.expand(1, args['height'], args['width'], 1)
    return depth, torch.ones_like(depth), {}


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
                if version != VERSION:
                    del state['config']['prediction_mode']
                with patch('scripts.training.train.torch.load', return_value=state):
                    loaded, _, cfg = load_model('memory/checkpoint.pt', 'cpu')
                    canonical = read_checkpoint('memory/checkpoint.pt')
                self.assertEqual(cfg['controller_features'], features)
                self.assertEqual('controller_features' in state['config'], version != LEGACY_VERSION)
                self.assertEqual('prediction_mode' in state['config'], version == VERSION)
                self.assertEqual(cfg['prediction_mode'], mode)
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
