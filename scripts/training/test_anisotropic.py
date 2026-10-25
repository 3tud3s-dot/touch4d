"""P1a.5 numerical, rollout and checkpoint regression tests; no data files."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .gs_decoder import (bounded_metric, anisotropic_anchor_field,
    build_initial_anchor_field, interpolate_anchor_displacements)
from .model import PositionPredictor
from . import test_gs_decoder as anchor_tests
from .test_gs_decoder import model_config
from .train import (GS_ANCHOR_VERSION, checkpoint_config, restore_training,
    checkpoint_state, recover_best_score, publish_checkpoints)
from .evaluate import load_model


class AnisotropicTests(unittest.TestCase):
    setUpClass = classmethod(anchor_tests.AnchorDecoderTests.setUpClass.__func__)
    tearDownClass = classmethod(anchor_tests.AnchorDecoderTests.tearDownClass.__func__)
    activate = anchor_tests.AnchorDecoderTests.activate
    checkpoint = anchor_tests.AnchorDecoderTests.checkpoint

    def setUp(self):
        anchor_tests.AnchorDecoderTests.setUp(self)
        self.model = PositionPredictor(**model_config(gs_decoder="anchors_anisotropic"))

    # Reuse the same contracts for both decoders, including the full five-step
    # graph, empty samples, causal condition changes and inherited attributes.
    test_constant_translation = anchor_tests.AnchorDecoderTests.test_zero_and_constant_head_are_initial_relative_not_accumulated
    test_causality_and_masks = anchor_tests.AnchorDecoderTests.test_five_step_causality_and_invalid_nan_masks
    test_missing_sample = anchor_tests.AnchorDecoderTests.test_all_missing_sample_remains_finite_and_does_not_mix_batch_geometry
    test_cross_step_chamfer = anchor_tests.AnchorDecoderTests.test_fifth_chamfer_reaches_every_previous_anchor_head_and_condition
    test_joint_loss = anchor_tests.AnchorDecoderTests.test_joint_loss_has_zero_scale_regularizer_and_finite_anchor_gradients

    def test_formula_spd_condition_bounds_and_overflow(self):
        raw = torch.randn(2,41,3,3,dtype=torch.float64)
        raw[0] *= .01
        sigma = torch.tensor([.012,.02],dtype=torch.float64)
        mask = torch.ones(2,41,dtype=torch.bool)
        metric = bounded_metric(raw,sigma,mask)
        a = .5*raw/torch.linalg.vector_norm(raw,dim=(-1,-2),keepdim=True).clamp_min(1.)
        transform = torch.eye(3,dtype=torch.float64)+a
        expected = (transform.transpose(-1,-2)@transform)/sigma[:,None,None,None].square()
        torch.testing.assert_close(metric,expected,atol=1e-10,rtol=1e-14)
        for value in (raw, torch.full_like(raw,1e300), torch.full_like(raw,-1e300)):
            dimensionless = bounded_metric(value,sigma,mask)*sigma[:,None,None,None].square()
            eigen = torch.linalg.eigvalsh(dimensionless)
            self.assertTrue(torch.isfinite(eigen).all())
            self.assertGreaterEqual(eigen.min().item(),.25-1e-12)
            self.assertLessEqual(eigen.max().item(),2.25+1e-12)
            self.assertLessEqual((eigen[...,-1]/eigen[...,0]).max().item(),9+1e-12)

    def test_metric_gradcheck_and_invalid_inputs(self):
        raw = (torch.randn(1,2,3,3,dtype=torch.float64)*.1).requires_grad_()
        sigma = torch.tensor([.03],dtype=torch.float64)
        mask = torch.ones(1,2,dtype=torch.bool)
        self.assertTrue(torch.autograd.gradcheck(lambda r:bounded_metric(r,sigma,mask),(raw,)))
        self.assertTrue(torch.autograd.gradcheck(lambda r:bounded_metric(r*10,sigma,mask),(raw,)))
        with self.assertRaises(ValueError):bounded_metric(raw,torch.zeros_like(sigma),mask)
        with self.assertRaises(ValueError):bounded_metric(raw,torch.full_like(sigma,float('nan')),mask)
        with self.assertRaises(ValueError):bounded_metric(torch.full_like(raw,float('nan')),sigma,mask)

    def test_isotropic_weights_and_learnable_weights_gradient(self):
        xyz = torch.randn(1,21,3,dtype=torch.float64)*.02
        keep = torch.ones(1,21,dtype=torch.bool)
        base = build_initial_anchor_field(xyz,keep,8,4)
        raw = torch.zeros(1,8,3,3,dtype=torch.float64,requires_grad=True)
        field = anisotropic_anchor_field(xyz,keep,base,raw)
        torch.testing.assert_close(field.weights,base.weights,atol=1e-15,rtol=1e-13)
        d = torch.randn(1,8,3,dtype=torch.float64)
        output = interpolate_anchor_displacements(d,field,keep)
        output.square().sum().backward()
        self.assertGreater(raw.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(raw.grad).all())
        shift = torch.tensor([.1,-.2,.3],dtype=torch.float64)
        field = anisotropic_anchor_field(xyz,keep,base,torch.randn_like(raw))
        result = interpolate_anchor_displacements(shift.expand_as(d),field,keep)
        torch.testing.assert_close(result,shift.expand_as(result),atol=1e-15,rtol=0)

    def test_padding_singleton_duplicates_empty_and_autocast(self):
        xyz = torch.randn(4,9,3)
        keep = torch.arange(9)[None] < torch.tensor([9,3,1,0])[:,None]
        xyz[1] = 0
        xyz[~keep] = float('nan')
        base = build_initial_anchor_field(xyz,keep)
        raw = torch.randn(4,9,3,3,requires_grad=True)
        with torch.no_grad(): raw[~base.anchor_mask] = float('nan')
        with torch.autocast('cpu',dtype=torch.bfloat16):
            field = anisotropic_anchor_field(xyz,keep,base,raw)
        self.assertEqual(field.weights.dtype,torch.float32)
        self.assertEqual(field.metric.dtype,torch.float32)
        torch.testing.assert_close(field.weights.sum(-1),keep.float())
        self.assertTrue(torch.isfinite(field.weights).all())
        loss = (field.weights*torch.randn_like(field.weights)).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue(torch.all(raw.grad[~base.anchor_mask]==0))
        xyz=torch.empty(2,0,3);keep=torch.empty(2,0,dtype=torch.bool)
        field=anisotropic_anchor_field(xyz,keep,build_initial_anchor_field(xyz,keep),torch.empty(2,0,3,3))
        self.assertEqual(tuple(field.weights.shape),(2,0,0))

    def test_shared_initial_weights_and_nonzero_head_isotropic_equivalence(self):
        torch.manual_seed(42); iso=PositionPredictor(**model_config())
        torch.manual_seed(42); ani=PositionPredictor(**model_config(gs_decoder="anchors_anisotropic"))
        for key,value in iso.state_dict().items():
            torch.testing.assert_close(value,ani.state_dict()[key],atol=0,rtol=0)
        self.assertTrue(torch.all(ani.metric_head[-1].weight==0))
        self.assertTrue(torch.all(ani.metric_head[-1].bias==0))
        torch.nn.init.normal_(iso.position_head[-1].weight,std=.02)
        ani.position_head.load_state_dict(iso.position_head.state_dict())
        before,after=iso(self.batch['inputs']),ani(self.batch['inputs'])
        for key in before:torch.testing.assert_close(after[key],before[key],atol=2e-7,rtol=2e-6)

    def test_initial_only_metric_shared_five_steps_and_fifth_gradient(self):
        self.activate()
        torch.nn.init.normal_(self.model.metric_head[-1].weight,std=.01)
        fields=[]
        def inspect(d,field,keep):
            fields.append(field)
            return interpolate_anchor_displacements(d,field,keep)
        with patch('scripts.training.model.anisotropic_anchor_field',wraps=anisotropic_anchor_field) as build, \
             patch('scripts.training.model.interpolate_anchor_displacements',side_effect=inspect):
            prediction=self.model(self.batch['inputs'])
        self.assertEqual(build.call_count,1)
        self.assertEqual(len(fields),5)
        self.assertTrue(all(f is fields[0] for f in fields))
        self.assertTrue(fields[0].metric.requires_grad)
        self.assertTrue(fields[0].weights.requires_grad)
        first_metric=fields[0].metric.detach().clone()
        prediction['xyz'][:,-1].square().sum().backward()
        for module in (self.model.metric_head[1],self.model.metric_head[-1],self.model.gs_encoder[0]):
            grads=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0)
        changed=copy.deepcopy(self.batch['inputs'])
        for key in ('action_state','controller_points','tactile'):
            changed[key]+=torch.randn_like(changed[key])*10
        fields.clear()
        with patch('scripts.training.model.interpolate_anchor_displacements',side_effect=inspect):
            self.model(changed)
        torch.testing.assert_close(fields[0].metric,first_metric,atol=0,rtol=0)

    def test_checkpoint_roundtrip_versions_optimizer_and_best_protection(self):
        for mode in ('anchors_anisotropic','anchors','free'):
            model=PositionPredictor(**model_config(gs_decoder=mode))
            torch.nn.init.normal_(model.position_head[-1].weight,std=.02)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            def update(m,o):
                o.zero_grad(set_to_none=True)
                m(self.batch['inputs'])['xyz'].square().mean().backward()
                o.step()
            update(model,opt)
            state=copy.deepcopy(checkpoint_state(model,opt,self.processor,self.checkpoint(model)['config'],2,.4,[]))
            if mode!='anchors_anisotropic':state['version']=GS_ANCHOR_VERSION
            with patch('scripts.training.train.torch.load',return_value=state):
                loaded,_,config=load_model('memory/state.pt','cpu')
            before,after=model(self.batch['inputs']),loaded(self.batch['inputs'])
            for key in before:torch.testing.assert_close(after[key],before[key],atol=0,rtol=0)
            restored=torch.optim.AdamW(loaded.parameters())
            restore_training(state,loaded,restored)
            update(model,opt);update(loaded,restored)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,loaded.state_dict()[key],atol=0,rtol=0)
            resaved=checkpoint_state(loaded,restored,self.processor,config,3,.6,[])
            canonical=copy.deepcopy(state);canonical['config']=checkpoint_config(state)
            with patch('pathlib.Path.is_file',return_value=True),patch('scripts.training.train.read_checkpoint',return_value=canonical):
                best=recover_best_score(resaved,'memory')
            self.assertEqual(best,.4)
            with patch('scripts.training.train.atomic_checkpoint') as save:
                publish_checkpoints(Path('memory'),dict(resaved,best=best),.5<best)
            self.assertEqual([c.args[0].name for c in save.call_args_list],['last.pt'])
            if mode=='anchors_anisotropic':
                forged=copy.deepcopy(state);forged['version']=GS_ANCHOR_VERSION
                with self.assertRaises(ValueError):checkpoint_config(forged)
                forged=copy.deepcopy(state);forged['config']['gs_decoder']='anchors'
                with self.assertRaises(ValueError):checkpoint_config(forged)
                original=PositionPredictor(**model_config())
                with self.assertRaises(ValueError):restore_training(state,original,torch.optim.AdamW(original.parameters()))


if __name__=='__main__':unittest.main()
