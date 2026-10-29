"""CPU checks of analysis tools, not substitutes for real CUDA acceptance."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_stage1.fixtures import analytic_rasterizer
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_g4_global_motion.evaluate import fixed_regions
from scripts.experiments.v4_g5_signed_local.model import G5PropagatePredictor
from scripts.experiments.v4_g5_signed_local.motion import geometry_weights
from scripts.experiments.v4_g5_signed_local.propagation import SignedAnchorPropagation,build_anchor_graph
from scripts.experiments.v4_g5_signed_local.diagnostics import (stats,finite_tree,cpu_tree,mapping_context,
    anchor_regions,collect,graph_diagnostics,uniform_propagation,controlled_neighbor_probe,parameter_evidence)
from scripts.experiments.v4_g5_signed_local.smoke import snapshot,metric_numbers
from scripts.experiments.v4_g5_signed_local.smoke import run_variant


class G5DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_quantiles_ignore_invalid_nan_and_empty_is_not_zero_observation(self):
        v=torch.tensor([1.,3.,float('nan')]);mask=torch.tensor([True,True,False])
        values=stats(v,'p_',mask)
        self.assertEqual(values['p_count'],2);self.assertEqual(values['p_p50'],2.)
        self.assertEqual(stats(v,'p_',torch.zeros(3,dtype=torch.bool))['p_mean'],None)
        with self.assertRaises(FloatingPointError):stats(v)
        with self.assertRaises(FloatingPointError):finite_tree({'optimizer':[torch.tensor(float('inf'))]})

    def test_far_softmax_is_normalized_not_contact_gated_and_gradient_finite(self):
        anchors=torch.tensor([[[100.,2.,1.],[101.,2.,1.]]],requires_grad=True)
        points=torch.zeros(1,2,30,3);points[0,0,:,0]=torch.linspace(0,.1,30)
        valid=torch.zeros(1,2,30,dtype=torch.bool);valid[0,0,:]=True
        points[0,1]=float('nan')
        weights,relative=geometry_weights(anchors,points,valid,torch.ones(1,2,dtype=torch.bool),torch.ones(1))
        torch.testing.assert_close(weights[:,:,0].sum(-1),torch.ones(1,2),atol=1e-6,rtol=0)
        self.assertEqual(float(weights[:,:,1].sum().detach()),0.)
        (weights*torch.arange(30.)).sum().backward()
        self.assertTrue(torch.isfinite(anchors.grad).all())

    def test_neighbor_seed_changes_vectors_not_score_and_uniform_is_operator_only(self):
        torch.manual_seed(5);module=SignedAnchorPropagation()
        xyz=torch.randn(1,12,3);valid=torch.ones(1,12,dtype=torch.bool);hands=torch.tensor([[True,True]])
        indices,mask=build_anchor_graph(xyz,valid)
        args=dict(query=torch.randn(1,12,128),hand_tokens=torch.randn(1,2,128),basis18=torch.randn(1,12,2,18),
                  seed=torch.randn(1,12,2,3)*.001,initial_xyz=xyz,current_xyz=xyz+.01,
                  anchor_valid=valid,hand_valid=hands,indices=indices,neighbor_mask=mask)
        r=controlled_neighbor_probe(module,**args)
        self.assertTrue(r['verified']);self.assertTrue(r['attention_weights_bit_exact'])
        seed=torch.tensor([[[[1.,-2.,3.],[-1.,2.,-3.]]]]).expand(1,12,2,3).clone()
        before={k:v.clone() for k,v in module.state_dict().items()}
        after=uniform_propagation(seed,indices,mask,valid,hands)
        torch.testing.assert_close(after,seed,rtol=0,atol=0)
        self.assertTrue(all(torch.equal(v,module.state_dict()[k]) for k,v in before.items()))

    def test_parameter_update_reports_decay_separately_from_learning(self):
        model=G5PropagatePredictor(horizon=1)
        opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        for p in model.parameters():p.grad=torch.zeros_like(p)
        grads={n:p.grad.clone() for n,p in model.named_parameters()};opt.step()
        rows=parameter_evidence(model,before,grads,lr=.01,decay=.1)
        self.assertEqual(rows['propagation_score']['raw_gradient_norm'],0.)
        self.assertGreater(rows['propagation_score']['actual_update_norm'],0.)
        # Float32 parameter rounding is retained, not misreported as gradient learning.
        self.assertLess(rows['propagation_score']['update_minus_adamw_decay_norm'],3e-6)

    def test_h15_hooks_regions_context_scaling_and_snapshot_do_not_change_model(self):
        torch.manual_seed(42);model=G5PropagatePredictor();processor=DataProcessor('001-rope')
        batch=prepare_batch(synthetic_batch(60,15,1),processor,model)
        # This collector has a documented batch1 diagnostic scope; use single-hand sample.
        def first(v):
            if isinstance(v,torch.Tensor):return v[:1]
            if isinstance(v,dict):return {k:first(w) for k,w in v.items()}
            if isinstance(v,list):return [first(w) for w in v]
            return v
        batch=first(batch);x=batch['inputs']
        with torch.no_grad():
            baseline=model(x)
            with mapping_context(model) as ctx:pred,trace=model(x,return_trace=True)
        self.assertTrue(all(torch.equal(v,pred[k]) for k,v in baseline.items()))
        self.assertEqual(len(ctx),15);self.assertEqual(ctx[0].shape[-1],287)
        groups,regions=anchor_regions(x,trace['field'])
        self.assertTrue(all(torch.equal(v,fixed_regions(x)[k]) for k,v in groups.items()))
        self.assertTrue(torch.equal(regions['near']|regions['middle']|regions['far'],regions['all']))
        rows,prop=collect(model,cpu_tree(x),cpu_tree(trace),cpu_tree(ctx),variant='propagate',window='test',stage='after_0',factor=1000.)
        self.assertEqual({r['horizon'] for r in rows},{1,5,10,15})
        self.assertEqual({r['hand'] for r in rows},{'right'})
        self.assertTrue(all(r['geometry_abs_max']>=0 for r in rows))
        self.assertTrue(all(r['cancellation_ratio_mean'] is None for r in rows))
        self.assertTrue(all(r['direct_mm_rms']==0 and r['propagated_mm_rms']==0 for r in prop))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'arrays').mkdir();tables=dict(geometry=[],propagation=[])
            snapshot(model,batch,pred,trace,ctx,out,'propagate','test','after_0',processor,tables)
            self.assertTrue((out/'arrays'/'propagate_test_after_0.npz').exists())
            self.assertGreater(graph_diagnostics(x,trace,1000.)['edge_mm_count'],0)
        self.assertTrue(all(torch.equal(v,model(x)[k]) for k,v in baseline.items()))

    def test_metric_units_and_original_loss_keys(self):
        p=DataProcessor('001-rope');m=dict(geometry=torch.tensor([.01]),rgb=torch.tensor([.2]),alpha=torch.tensor([.3]),
            rgb_by_horizon=torch.ones(1,15)*.2,alpha_by_horizon=torch.ones(1,15)*.3)
        r=metric_numbers(torch.tensor(.24),m,p)
        self.assertAlmostEqual(r['cd_mm'],.01*p.state_dict()['scale']*1000,places=5)
        self.assertEqual(len(r['rgb_by_horizon']),15)

    def test_bounded_runner_two_steps_frozen_validation_and_strict_restore_cpu_mock(self):
        # Control-plane test only: real CPU G5/Loss, analytic renderer and CUDA API mocks.
        # Temporary results are NOT real GPU/gsplat acceptance artifacts.
        for variant in ('direct','propagate'):
            processor=DataProcessor('001-rope');template=G5PropagatePredictor()
            batch=prepare_batch(synthetic_batch(60,15,1),processor,template)
            def first(v):
                if isinstance(v,torch.Tensor):return v[:1]
                if isinstance(v,dict):return {k:first(w) for k,w in v.items()}
                if isinstance(v,list):return [first(w) for w in v]
                return v
            batch=first(batch);steps=[];original_step=torch.optim.AdamW.step
            def counting_step(opt,*args,**kwargs):
                steps.append(1);return original_step(opt,*args,**kwargs)
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)
                for name in ('arrays','images','figures'):(out/name).mkdir()
                batches={name:batch for name in ('ep7_start105','ep3_start135','ep6_start110')}
                with patch.object(torch.nn.Module,'cuda',lambda self:self), \
                     patch('scripts.experiments.v4_g5_signed_local.smoke.to_device',lambda value,device:value), \
                     patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'synchronize'), \
                     patch.object(torch.cuda,'max_memory_allocated',return_value=0), \
                     patch.object(torch.cuda,'max_memory_reserved',return_value=0), \
                     patch.object(torch.cuda,'empty_cache'),patch.object(torch.optim.AdamW,'step',counting_step):
                    result=run_variant(variant,SimpleNamespace(),processor,batches,out,
                        GaussianRenderer(rasterizer=analytic_rasterizer),dict(geometry=[],propagation=[]),'cpu-test')
                self.assertEqual(len(steps),2);self.assertEqual(result['successful_updates'],2)
                self.assertTrue(result['optimizer_restore_exact'])
                self.assertTrue(result['validation_weights_unchanged'])
                self.assertEqual(len(result['frozen_inference']),3)
                self.assertTrue(all(result['checkpoint_restore_exact'].values()))
                self.assertTrue((out/variant/'update_1.pt').exists())
                self.assertTrue((out/variant/'smoke.pt').exists())


if __name__=='__main__':unittest.main()
