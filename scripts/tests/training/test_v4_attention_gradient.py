"""Frozen-weight diagnostics: equivalence, gradients and non-mutating hooks."""
import ast
import copy
from pathlib import Path
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.training.rendering import GaussianRenderer
from scripts.experiments.fixed_window.fixed_fit import digest, TargetCache
from scripts.experiments.v4_stage1.fixtures import synthetic_batch, analytic_rasterizer
from scripts.experiments.v4_anchor_interaction.model import paired_models
from scripts.experiments.v4_attention_gradient.core import (
    components, observe, altered_inputs, verify_equivalence, normalized_direction,
    cosine, stats, single_horizon, parameter_summary, MODULES)
from scripts.experiments.v4_attention_gradient.diagnose import audit, finite_differences


def first_batch(batch):
    if isinstance(batch,torch.Tensor):return batch[:1]
    if isinstance(batch,dict):return {k:first_batch(v) for k,v in batch.items()}
    if isinstance(batch,list):return [first_batch(v) for v in batch]
    return batch


class FrozenAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def make(self,tag='G0',h=10):
        m=paired_models(h)[tag].eval()
        # Nondegenerate synthetic fixture, not an update of any frozen checkpoint.
        with torch.no_grad():m.motion[-1].weight.normal_(0,.001)
        p=DataProcessor('001-rope');b=prepare_batch(synthetic_batch(points=20,horizon=h),p,m)
        return m,b,p,TargetCache(GaussianRenderer(rasterizer=analytic_rasterizer))

    def test_loss_and_all_parameter_gradient_equivalence_three_models(self):
        for tag in ('G0','G2','G3'):
            m,b,p,r=self.make(tag,h=5);before=digest(m.state_dict())
            out=verify_equivalence(m,b,p,r,chunk=8)
            self.assertTrue(out['passed']);self.assertEqual(before,digest(m.state_dict()))

    def test_empty_sample_reduction_matches_production(self):
        m,b,p,r=self.make(h=5);b['targets']['target_depth_valid'][0]=False
        self.assertTrue(verify_equivalence(m,b,p,r,chunk=8)['passed'])

    def test_hooks_remove_after_exception_and_identity(self):
        m,b,p,r=self.make(h=5);before=digest(m.state_dict())
        expected=m(b['inputs'])['xyz']
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with observe(m):raise RuntimeError('intentional')
        self.assertFalse(m.motion._forward_pre_hooks);self.assertFalse(m.cross[-1]._forward_hooks)
        with observe(m) as c:actual=m(b['inputs'])['xyz']
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(len(c['motion_input']),5);self.assertEqual(before,digest(m.state_dict()))

    def test_zero_tool_only_replaces_twelve_dimensions_and_removes_hooks(self):
        m,b,p,r=self.make(h=5)
        with observe(m) as normal:m(b['inputs'])
        with observe(m,True) as changed:m(b['inputs'])
        self.assertTrue(torch.all(changed['motion_input'][0][...,128:140]==0))
        torch.testing.assert_close(changed['motion_input'][0],normal['motion_input'][0],rtol=0,atol=0)
        # AR changes downstream states legitimately; first-step other fields must be equal.
        self.assertFalse(m.motion._forward_pre_hooks)

    def test_pressure_ablation_preserves_validity_age_and_input(self):
        m,b,p,r=self.make(h=5);x=b['inputs'];before=digest(x)
        changed=altered_inputs(x,'zero_tactile_pressure')
        self.assertTrue(torch.all(changed['tactile']==0))
        for n in ('valid_mask','tactile_spatial_mask','tactile_age_us','operation_hands'):
            torch.testing.assert_close(changed[n],x[n],rtol=0,atol=0)
        self.assertEqual(before,digest(x))
        frozen=altered_inputs(x,'freeze_action')
        torch.testing.assert_close(frozen['controller_points'][:,1],x['controller_points'][:,0])

    def test_direct_and_total_ar_derivatives_are_separately_measured(self):
        x=torch.tensor(2.,requires_grad=True);s1=x*3;s2=s1*5;out=torch.stack([s1,s2])
        loss=out.square().mean()
        direct,g1,g2=torch.autograd.grad(loss,[out,s1,s2])
        self.assertEqual(float(direct[0]),6.);self.assertEqual(float(g1),156.)
        self.assertEqual(float(g2),30.);self.assertNotEqual(float(g1),float(direct[0]))

    def test_zero_none_cosine_and_same_parameter_coordinates(self):
        self.assertIsNone(cosine(torch.zeros(3),torch.ones(3)));self.assertIsNone(cosine(None,torch.ones(3)))
        self.assertAlmostEqual(cosine(torch.ones(3),-torch.ones(3)),-1.)
        with self.assertRaises(ValueError):cosine(torch.ones(2),torch.ones(3))
        self.assertFalse(stats(None)['connected']);self.assertTrue(stats(torch.zeros(3))['connected'])

    def test_physical_rms_normalization_excludes_padding(self):
        d=torch.tensor([[[[1.,2.,3.],[999.,999.,999.]]]])
        mask=torch.tensor([[[True,False]]]);unit=normalized_direction(d,mask)
        self.assertAlmostEqual(float(unit[mask].square().sum(-1).mean().sqrt()),1.,places=6)
        self.assertTrue(torch.all(unit[~mask]==0))
        self.assertIsNone(normalized_direction(torch.zeros_like(d),mask))

    def test_complete_h10_gradient_audit_without_optimizer_or_parameter_mutation(self):
        m,b,p,r=self.make();b=first_batch(b);before=digest(m.state_dict())
        result,pred,raw=audit(m,b,p,r)
        self.assertEqual(before,digest(m.state_dict()));self.assertEqual(len(result['loss_gradients']),10*4*3*4)
        self.assertEqual(raw['cd_direct_Hmean'].shape,(1,10,20,3))
        for n in MODULES:
            self.assertGreater(result['parameter_gradients'][n]['parameter_count'],0)
        self.assertTrue(all(q.grad is None for q in m.parameters()))
        self.assertEqual(float(torch.from_numpy(raw['joint_total_AR_Hmean'][:,-1]).sub(torch.from_numpy(raw['joint_direct_Hmean'][:,-1])).abs().max()),0.)

    def test_finite_difference_mock_and_horizon_slicing(self):
        m,b,p,r=self.make();b=first_batch(b)
        with torch.no_grad():pred=m(b['inputs'])
        one,bb=single_horizon(pred,b,9)
        self.assertEqual(one['xyz'].shape[1],1);self.assertEqual(len(bb['targets']['target_xyz']),1)
        rows=finite_differences(pred,b,p,r,pilot=True)['rows']
        joint=[z for z in rows if z['loss']=='joint']
        self.assertTrue(all(z['autograd_directional_derivative_per_mm']<0 for z in joint))
        self.assertLess(min(z['central_relative_error'] for z in joint),.05)

    def test_diagnostic_source_has_no_optimizer_step(self):
        root=Path(__file__).parents[2]/'experiments/v4_attention_gradient'
        for f in root.glob('*.py'):
            tree=ast.parse(f.read_text(encoding='utf8'))
            for n in ast.walk(tree):
                self.assertFalse(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step',f.name)


if __name__=='__main__':unittest.main()
