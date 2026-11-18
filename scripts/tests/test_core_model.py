"""CPU regression for the retained model; no dataset reads or training jobs."""
import ast
import copy
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import unittest

import numpy as np
import torch


CORE_ROOT = Path(__file__).resolve().parents[2]


def synthetic_inputs(points=24, horizon=15):
    """Explicit normalized-world inputs: one single-hand and one bimanual item."""
    generator = torch.Generator().manual_seed(901)
    batch = 2
    xyz = torch.randn(batch, points, 3, generator=generator) * .08
    xyz[..., 2] += .15
    keep = torch.ones(batch, points, dtype=torch.bool)
    keep[1, -5:] = False
    attr = torch.zeros(batch, points, 11)
    attr[..., :3] = -5.
    attr[..., 3] = 1.
    attr[..., 7] = .4
    attr[..., 8:] = .1
    cp = torch.randn(batch, 1, 2, 30, 3, generator=generator).expand(-1, horizon, -1, -1, -1).clone() * .04
    for step in range(horizon):
        cp[:, step, 0, :, 0] += step * .001
        cp[:, step, 1, :, 1] -= step * .0015
        cp[:, step, :, :, 2] += step * .0005
    action = torch.zeros(batch, horizon, 2, 13)
    action[..., :3] = cp.mean(-2)
    action[..., 3:12] = torch.eye(3).flatten()
    action[..., 12] = .08
    pressure = torch.rand(batch, horizon, 4, 12, 32, generator=generator)
    valid = torch.ones_like(pressure, dtype=torch.bool)
    valid[..., 31] = False
    hands = torch.tensor([[True, False], [True, True]])
    return dict(gs_xyz=xyz, gs_attr=attr, gs_mask=keep,
                action_state=action, action_valid=torch.ones(batch, horizon, 2, dtype=torch.bool),
                controller_points=cp, controller_mask=torch.ones(batch, horizon, 2, 30, dtype=torch.bool),
                tactile=pressure, valid_mask=valid, tactile_spatial_mask=valid.clone(),
                tactile_age_us=torch.full((batch, horizon, 4), 12000, dtype=torch.int64),
                operation_hands=hands, condition_frame_ids=(73 + torch.arange(horizon))[None].expand(batch, -1).clone())


def exercise_nonzero_heads(model):
    """Avoid an uninformative exact comparison of only zero-init Copy output."""
    torch.manual_seed(2027)
    with torch.no_grad():
        for head in (model.motion[-1], model.global_motion.mapping[-1], model.direct_motion.mapping[-1]):
            head.weight.normal_(std=.003)
            head.bias.normal_(std=.001)


def oracle_snapshot(source_root, legacy):
    """Execute separate namespaces so Python's scripts package cache cannot mix them."""
    script = r'''
import json, runpy, sys
from pathlib import Path
import torch
torch.set_num_threads(1)
legacy = sys.argv[1] == 'legacy'
helper = runpy.run_path(sys.argv[2])
if legacy:
    from scripts.experiments.v4_g5_signed_local.protocol import fresh_model, configuration, probe_order
    from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
else:
    from scripts.training.cloth_protocol import fresh_model, configuration, probe_order
    from scripts.training.cloth_checkpoint import state_hash
model = fresh_model('direct')
initial = dict(config=model.config, state_keys=list(model.state_dict()),
               parameters=sum(p.numel() for p in model.parameters()), state_sha256=state_hash(model.state_dict()))
x = helper['synthetic_inputs']()
with torch.no_grad():
    fresh, fresh_trace = model(x, return_trace=True)
helper['exercise_nonzero_heads'](model)
out, trace = model(x, return_trace=True)
loss = out['xyz'].square().mean() + out['xyz'][..., 0].mean()
loss.backward()
grads = {name:p.grad for name,p in model.named_parameters() if p.grad is not None}
if not all(torch.isfinite(v).all() for v in grads.values()):
    raise AssertionError('Nonfinite backward')
root=Path(sys.argv[3])
config = configuration(root/'read-only-data-fixture', root/'read-only-depth-fixture', 'frozen-source', 'direct')
orders = [probe_order(torch.utils.data.DataLoader(range(11), batch_size=1, shuffle=True, num_workers=0,
          generator=torch.Generator().manual_seed(42+epoch))) for epoch in (0,1,7)]
result=dict(initial=initial, config=config, orders=orders,
            fresh_sha256=state_hash(fresh), active_prediction_sha256=state_hash(out),
            active_gradient_sha256=state_hash(grads), loss=float(loss.detach()),
            trace_sha256=[state_hash({k:step[k] for k in ['alpha','cumulative_d','state','local_correction','global_delta']})
                          for step in trace['steps']])
print(json.dumps(result, sort_keys=True))
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-B", "-c", script,
                             "legacy" if legacy else "core", str(Path(__file__).resolve()), str(CORE_ROOT)],
                            cwd=source_root, env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return json.loads(result.stdout)


class CoreModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fresh(self):
        from scripts.training.cloth_protocol import fresh_model
        return fresh_model()

    def test_graph_identity_and_parameter_budget(self):
        from scripts.training.cloth_checkpoint import state_hash
        model = self.fresh()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 620511)
        self.assertEqual(model.config['horizon'], 15)
        self.assertEqual(model.config['prediction_interval'], 1)
        self.assertFalse(model.config['anchor_propagation'])
        self.assertEqual(state_hash(model.state_dict()), state_hash(self.fresh().state_dict()))

    def test_fresh_h15_is_exact_masked_copy_and_inherits_attributes(self):
        x = synthetic_inputs()
        with torch.no_grad():
            output = self.fresh()(x)
        expected = torch.where(x['gs_mask'][..., None], x['gs_xyz'], 0.)[:, None].expand(-1, 15, -1, -1)
        self.assertTrue(torch.equal(output['xyz'], expected))
        for name, lo, hi in [('log_scales',0,3),('quaternions',3,7),('opacity_logits',7,8),('sh_dc',8,11)]:
            expected = torch.where(x['gs_mask'][...,None], x['gs_attr'], 0.)[:, None, :, lo:hi].expand(-1,15,-1,-1)
            self.assertTrue(torch.equal(output[name], expected), name)

    def test_active_heads_support_finite_full_h15_backward(self):
        model = self.fresh()
        exercise_nonzero_heads(model)
        output = model(synthetic_inputs())
        loss = output['xyz'].square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.count_nonzero() for g in gradients))

    def test_invalid_padding_and_missing_hand_are_masked(self):
        model = self.fresh()
        exercise_nonzero_heads(model)
        x = synthetic_inputs()
        x['gs_xyz'][~x['gs_mask']] = float('nan')
        x['gs_attr'][~x['gs_mask']] = float('nan')
        x['action_state'][0,:,1] = float('nan')
        x['controller_points'][0,:,1] = float('nan')
        with torch.no_grad():
            output = model(x)
        self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
        self.assertEqual(int(output['xyz'][1,:,-5:].count_nonzero()),0)

    def test_future_condition_changes_cannot_change_earlier_predictions(self):
        model = self.fresh()
        exercise_nonzero_heads(model)
        x = synthetic_inputs()
        changed = copy.deepcopy(x)
        changed['action_state'][:,10:,:,:3] += .2
        changed['controller_points'][:,10:] -= .1
        changed['tactile'][:,10:] += 3.
        with torch.no_grad():
            before, after = model(x), model(changed)
        self.assertTrue(torch.equal(before['xyz'][:,:10], after['xyz'][:,:10]))
        self.assertFalse(torch.equal(before['xyz'][:,10:], after['xyz'][:,10:]))

    def test_interval_and_empty_initial_state_fail_closed(self):
        model = self.fresh()
        x = synthetic_inputs()
        x['condition_frame_ids'][:,1:] += 1
        with self.assertRaises(ValueError):
            model(x)
        x = synthetic_inputs()
        x['gs_mask'][1] = False
        with self.assertRaises(ValueError):
            model(x)

    def test_rng_restore_and_optimizer_configuration(self):
        from scripts.training.cloth_checkpoint import restore_rng, rng_state
        from scripts.training.cloth_train import optimizer_for
        model = self.fresh()
        before = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(before)
        actual = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        optimizer = optimizer_for(model, {'lr':1e-4,'weight_decay':1e-4})
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-4)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'],1e-4)
        self.assertEqual(len(optimizer.state),0)

    def test_core_files_import_no_archived_experiments(self):
        for path in (CORE_ROOT/'scripts/training').glob('cloth_*.py'):
            if path.name in ('cloth_data.py','cloth_evaluate.py'):
                continue  # Owned and tested separately by the data/evaluation migration.
            tree = ast.parse(path.read_text(encoding='utf-8'))
            imports = [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            imports += [alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
            self.assertFalse(any('experiments' in name for name in imports),str(path))

    def test_exact_oracle_relocation(self):
        old = Path(os.environ.get('TOUCH4D_FROZEN_SOURCE_ROOT', str(CORE_ROOT.parent/'v4-g5-direct-008-v1')))
        if not old.is_dir():
            self.skipTest('Set TOUCH4D_FROZEN_SOURCE_ROOT to the isolated frozen Direct source checkout')
        legacy = oracle_snapshot(old, True)
        core = oracle_snapshot(CORE_ROOT, False)
        self.assertEqual(legacy, core)


if __name__ == '__main__':
    unittest.main()
