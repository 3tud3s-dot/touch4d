"""Joint gradients, output-only AR, unchanged loss, strict full-state resume."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g4_global_motion.protocol import digest
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_geometry.train import initialize as frozen_initialize
from scripts.tests.training.test_v4_g5_training import first, manifest, row
from scripts.experiments.v4_g5_geometry_joint.model import fresh_model, initialization, ARCHITECTURE
from scripts.experiments.v4_g5_geometry_joint.protocol import configuration, common_configuration, output_check
from scripts.experiments.v4_g5_geometry_joint.train import perform_epoch, make_identity
from scripts.experiments.v4_g5_geometry_joint import checkpoint as cp


def identity(root,model):
    m=manifest();orders=[[0,1]]*8
    data=dict(manifest=m,manifest_hash=digest(m),epoch_orders=orders,
        epoch_order_hashes=[digest(o) for o in orders],observed_epoch_order_hashes=[digest(o) for o in orders])
    return make_identity(root,model,DataProcessor('001-rope'),configuration('data','depth','test-source'),
                         data,{'device':'cpu'},initialization(model))


class JointGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        self.model=fresh_model()
        self.processor=DataProcessor('001-rope')
    def inputs(self):
        return prepare_batch(first(synthetic_batch(24,15,1)),self.processor,self.model)['inputs']
    def optimizer(self):return torch.optim.AdamW(self.model.parameters(),lr=1e-4,weight_decay=1e-4)
    def step(self,opt):
        opt.zero_grad(set_to_none=True)
        factor=random.random()+float(np.random.rand())+float(torch.rand(()))
        sum(p.square().sum()*factor for p in self.model.parameters()).backward();opt.step()

    def test_fresh_g5_and_geo_pair_and_parameter_coverage(self):
        torch.manual_seed(42);base=G5DirectPredictor(horizon=15,prediction_interval=1)
        self.assertTrue(same_tree(base.state_dict(),self.model.backbone.state_dict()))
        frozen=frozen_initialize(copy.deepcopy(base))
        self.assertTrue(same_tree(frozen.head.state_dict(),self.model.head.state_dict()))
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertEqual(initialization(self.model)['trainable_parameters'],646053)
        self.assertEqual({id(p) for g in self.optimizer().param_groups for p in g['params']},{id(p) for p in self.model.parameters()})

    def test_train_mode_does_not_refreeze_backbone(self):
        self.model.eval();self.model.train()
        self.assertTrue(self.model.backbone.training)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))

    def test_zero_full_h15_exact_and_nonxyz_properties(self):
        x=self.inputs()
        with torch.no_grad():
            base=self.model.backbone(x)
            self.assertTrue(same_tree(base,self.model(x,mode='geometry_on_zero')))
            self.assertTrue(same_tree(base,self.model(x,mode='geometry_off')))

    def test_nonzero_geometry_does_not_feed_ar_or_xyz(self):
        x=self.inputs()
        with torch.no_grad():
            base,bt=self.model.backbone(x,return_trace=True)
            self.model.head[-1].bias.copy_(torch.tensor([.3,-.2,.1,.2,.1,-.1]))
            out,trace=self.model(x,return_trace=True)
        self.assertTrue(torch.equal(base['xyz'],out['xyz']))
        self.assertTrue(torch.equal(base['opacity_logits'],out['opacity_logits']))
        self.assertTrue(torch.equal(base['sh_dc'],out['sh_dc']))
        for a,b in zip(bt['steps'],trace['steps']):
            self.assertTrue(torch.equal(a['final_cumulative_anchor_displacement'],b['final_cumulative_anchor_displacement']))
        self.assertFalse(torch.equal(base['log_scales'],out['log_scales']))
        with self.assertRaises(ValueError):self.model(x,mode='geometry_on_zero')

    def test_geometry_only_loss_backpropagates_to_query_and_motion(self):
        with torch.no_grad():self.model.head[-1].weight.fill_(.01)
        out,trace=self.model(self.inputs(),return_trace=True)
        self.assertTrue(trace['geometry_features'].requires_grad)
        (out['log_scales'].sum()+out['quaternions'][...,1:].sum()).backward()
        for module in (self.model.backbone.cross,self.model.backbone.motion):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)

    def test_xyz_loss_cannot_backpropagate_through_geometry(self):
        out=self.model(self.inputs());out['xyz'].square().sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.head.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.backbone.motion.parameters() if p.grad is not None),0)

    def test_zero_projection_scale_rotation_receive_first_gradient(self):
        out=self.model(self.inputs())
        (out['log_scales'].sum()+out['quaternions'][...,1:].sum()).backward()
        grad=self.model.head[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad[:3].abs().sum()),0)
        self.assertGreater(float(grad[3:].abs().sum()),0)

    def test_causal_future_conditions_cannot_change_earlier_geometry(self):
        x=self.inputs();future=copy.deepcopy(x)
        for key in ('action_state','controller_points','tactile'):future[key][:,10:]+=0.02
        with torch.no_grad():
            self.model.head[-1].weight.fill_(.01)
            a=self.model(x);b=self.model(future)
        for k in a:self.assertTrue(torch.equal(a[k][:,:10],b[k][:,:10]),k)

    def test_original_joint_loss_configuration_and_namespace(self):
        c=configuration('data','depth','source');shared=common_configuration(c)
        self.assertEqual(c['architecture'],ARCHITECTURE)
        self.assertEqual((c['epochs'],c['horizon'],c['prediction_interval']),(8,15,1))
        self.assertEqual((c['chamfer_weight'],c['rgb_weight'],c['alpha_weight'],c['scale_weight']),(1.,1.,.1,0.))
        wrong=dict(c,lr=2e-4)
        with self.assertRaises(ValueError):common_configuration(wrong)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):output_check(str(Path(d)/'wrong'/'run-001'))

    def test_full_checkpoint_strict_restore_rng_optimizer_and_alias_repair(self):
        opt=self.optimizer()
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);initial=cp.save_boundary(d,self.model,opt,ident,cp.empty_ledger())
            self.step(opt);ledger=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
            cp.save_boundary(d,self.model,opt,ident,ledger)
            torch.save(initial,Path(d)/'last.pt')
            self.step(opt);wanted=copy.deepcopy(self.model.state_dict());wanted_opt=copy.deepcopy(opt.state_dict())
            self.assertEqual(cp.restore(d,self.model,opt,ident),ledger)
            self.step(opt)
            self.assertTrue(same_tree(wanted,self.model.state_dict()))
            self.assertTrue(same_tree(wanted_opt,opt.state_dict()))
            self.assertEqual(cp.read(Path(d)/'last.pt',ident)['ledger'],ledger)

    def test_uncommitted_epoch_rolls_back_and_immutable_checkpoint(self):
        opt=self.optimizer()
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);initial=copy.deepcopy(self.model.state_dict())
            cp.save_boundary(d,self.model,opt,ident,cp.empty_ledger())
            self.step(opt);self.assertEqual(cp.restore(d,self.model,opt,ident)['successful_updates'],0)
            self.assertTrue(same_tree(initial,self.model.state_dict()))
            self.step(opt);ledger=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
            cp.save_boundary(d,self.model,opt,ident,ledger)
            with self.assertRaises(FileExistsError):cp.save_boundary(d,self.model,opt,ident,ledger)

    def test_reject_other_formats_sources_feedback_and_partial_optimizer(self):
        opt=self.optimizer()
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);state=cp.save_boundary(d,self.model,opt,ident,cp.empty_ledger())
            for version in ('touch4d-g5-direct-formal-h15-epochs-v1','touch4d-g5-geometry-head-only-formal-epochs-v1','g5-geo-joint-smoke-v1'):
                with self.assertRaises(ValueError):cp.validate(dict(state,version=version),ident)
            for key in ('run_id','source_sha'):
                with self.assertRaises(ValueError):cp.restore(d,self.model,opt,dict(ident,**{key:'wrong'}))
            wrong=copy.deepcopy(ident);wrong['geometry_config']['ar_feedback']=True
            with self.assertRaises(ValueError):cp.validate(dict(state,identity=wrong),wrong)
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model)
            with self.assertRaises(ValueError):cp.save_boundary(d,self.model,torch.optim.AdamW(self.model.head.parameters()),ident,cp.empty_ledger())

    def test_full_val_best_order_and_counter_validation(self):
        opt=self.optimizer()
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);state=cp.save_boundary(d,self.model,opt,ident,cp.empty_ledger())
            ledger=cp.empty_ledger()
            for epoch,score in ((1,.1),(2,.2)):
                self.step(opt);ledger=cp.commit_epoch(ledger,row(epoch,score),ident['data']['manifest'])
                cp.save_boundary(d,self.model,opt,ident,ledger)
            self.assertEqual(cp.read(Path(d)/'best.pt',ident)['ledger']['completed_epochs'],1)
            self.assertEqual(cp.read(Path(d)/'last.pt',ident)['ledger']['completed_epochs'],2)
            bad=copy.deepcopy(ledger);bad['history'][-1]['optimizer_update_flags']=[True,True]
            with self.assertRaises(ValueError):cp.validate(dict(state,ledger=bad),ident)
            bad=copy.deepcopy(ledger);bad['history'][-1]['validation']['model']['episodes']['3']['samples']=0
            with self.assertRaises(ValueError):cp.validate(dict(state,ledger=bad),ident)

    def test_real_runner_interface_updates_both_and_validation_is_read_only(self):
        def raster(**args):
            signal=args['means'].mean()+args['scales'].mean()*100+args['quats'][...,1:].mean()
            rgb=torch.sigmoid(signal).expand(1,args['height'],args['width'],3)
            alpha=torch.sigmoid(signal).expand(1,args['height'],args['width'],1)
            return torch.cat((rgb,alpha*.5),-1),alpha,{}
        train=first(synthetic_batch(24,15,1));a=copy.deepcopy(train);a['meta']['episode_id'][0]=3
        b=copy.deepcopy(train);b['meta']['episode_id'][0]=6;b['inputs']['operation_hands'][:]=True
        before=initialization(self.model);val=[]
        def capture(module,args):
            if not module.training:val.append(state_hash(module.state_dict()))
        hook=self.model.register_forward_pre_hook(capture)
        try:
            result=perform_epoch(self.model,self.optimizer(),self.processor,GaussianRenderer(raster),
                configuration('data','depth','test'),'cpu',[train],[a,b],[[2,73]],[[3,73],[6,73]])
        finally:hook.remove()
        after=initialization(self.model)
        self.assertEqual(result['successful_updates'],1)
        self.assertEqual(result['optimizer_update_flags'],[True])
        self.assertNotEqual(before['backbone_state_hash'],after['backbone_state_hash'])
        self.assertNotEqual(before['head_state_hash'],after['head_state_hash'])
        self.assertEqual(val,[state_hash(self.model.state_dict())]*2)


if __name__=='__main__':unittest.main()
