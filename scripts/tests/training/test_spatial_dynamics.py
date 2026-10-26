"""Research adapter contracts; CPU only, no data/model downloads."""
import copy
import tempfile
from pathlib import Path
import unittest
import subprocess
import torch
from scripts.data_process import DataProcessor
from scripts.tests.data_process.test_data_process import make_batch
from scripts.experiments.spatial_dynamics.conditions import ToolPredictor,hand_inputs,sanitize
from scripts.experiments.spatial_dynamics.model import (SpatialPredictor,CANDIDATES,initial_knn,
    build_initial_anchor_field,local_penalty,se3_decode)
from scripts.experiments.spatial_dynamics.runner import paired_model,save_checkpoint,restore,update_order,perturb


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):
        self.processor=DataProcessor('001-rope');self.batch=self.processor(make_batch())
        self.x=hand_inputs(self.batch['inputs'],torch.tensor([1,5]),['001-rope']*2)
        self.arch=dict(width=16,heads=4,layers=2,controller_features=7,horizon=5,gs_local_attention='knn',
            gs_local_k=3,gs_decoder='anchors',anchor_count=4,anchor_pool_k=3,gs_structure='anchor_attention',output_mode='gs_geometry')
    def model(self,mode='S0',**kwargs):return paired_model(mode,**kwargs,**self.arch)[0]
    def active(self):
        x=copy.deepcopy(self.x)
        for key in ('action_valid','controller_mask','valid_mask','tactile_spatial_mask'):x[key].fill_(True)
        x['valid_mask'][...,-1]=False;x['tactile_spatial_mask'][...,-1]=False
        x['tactile_age_us'].fill_(20000);x['tactile'].fill_(.5);x['operation_hands'].fill_(True)
        return x
    def nonzero(self,m):
        torch.nn.init.normal_(m.position_head[-1].weight,std=.01)
        torch.nn.init.normal_(m.fusion[-1].weight,std=.01)
        return m
    def test_common_parameters_paired(self):
        for mode in CANDIDATES:self.model(mode)
        for options in [dict(decoder='D2'),dict(feedback='F1'),dict(regularization=.01)]:self.model('S2-C1',**options)
    def test_s0_matches_original_p2_rollout(self):
        new=self.nonzero(self.model());old=ToolPredictor(**self.arch);old.load_state_dict(new.state_dict())
        for key,value in new(self.x).items():self.assertTrue(torch.equal(value,old(self.x)[key]),key)
    def test_t3_port_exact_source_commit(self):
        source=subprocess.check_output(['git','show','9142704:scripts/experiments/condition_encoder/model.py'],text=True,encoding='utf-8')
        scope={};exec(compile(source,'<frozen-9142704>','exec'),scope)
        old=scope['ConditionPredictor'](condition_mode='T3',**self.arch)
        new=self.nonzero(self.model());old.load_state_dict(new.state_dict())
        for key,value in new(self.x).items():self.assertTrue(torch.equal(value,old(self.x)[key]),key)
    def test_zero_bias_and_propagation(self):
        x=self.active()
        for a,b in [('S0','S1'),('S2-C0','S2-C1'),('S2-C1','S3-Pre'),('S2-C1','S3-Post')]:
            m=self.nonzero(self.model(a));n=self.model(b);n.load_state_dict(m.state_dict(),strict=False)
            torch.testing.assert_close(m(x)['xyz'],n(x)['xyz'],atol=2e-6,rtol=2e-6)
    def test_hand_and_fallback(self):
        x=self.active();x['operation_hands'][0,0]=False
        tokens,keep=self.model().encode_conditions(x);ids=[0,*range(2,32),62,63]
        self.assertFalse(keep[0,:,ids].any());self.assertTrue((tokens[0,:,ids]==0).all())
        x['controller_mask'].zero_();_,keep=self.model().encode_conditions(x)
        self.assertFalse(keep[0,:,62:64].any());self.assertTrue(keep[0,:,64:].all())
    def test_t3_zero_localmax_quality_nan(self):
        x=self.active();x['tactile'].zero_();m=self.model()
        mean,maxi,cov,age,obs,maxobs=m.aggregate(sanitize(x))
        self.assertTrue(obs.all());self.assertTrue((mean==0).all());self.assertTrue((cov==1).all())
        x['tactile'][0,0,0,2,8]=9
        maxi=m.aggregate(sanitize(x))[1];self.assertEqual(int((maxi[0,0,0]>0).sum()),1)
        self.assertFalse((maxi[0,0,1:]>0).any())
        x['tactile'][0,0,0,2,8]=float('nan');x['tactile_age_us'][0,1,0]=75001
        self.assertTrue(torch.isfinite(m.encode_conditions(x)[0]).all())
        self.assertFalse(sanitize(x)['valid_mask'][0,1,0].any())
    def test_a0_reference_and_validity(self):
        x=self.active();x['controller_points'][:,0]=0;x['controller_points'][:,2]=4
        x['controller_mask'][0,0]=False
        a=self.model().controller_inputs(x)
        self.assertTrue((a[0,2,...,3:]==0).all());self.assertTrue((a[0,2,...,:3]==4).all())
        self.assertTrue((a[1,2,...,3:6]==4).all())
    def test_null_only_padding(self):
        x=self.active();x['operation_hands'].zero_();x['gs_xyz'][~x['gs_mask']]=float('nan')
        for mode in CANDIDATES:
            m=self.model(mode);out=m(x)
            for v in out.values():self.assertTrue(torch.isfinite(v).all())
            self.assertTrue((out['xyz'][~x['gs_mask'][:,None].expand(-1,5,-1)]==0).all())
    def test_dynamic_selection_fallback_and_uniqueness(self):
        x=self.active();m=self.model('S4-C1');x['tactile'].zero_();m(x)
        fixed=build_initial_anchor_field(x['gs_xyz'],x['gs_mask'],4,3)
        for row in m.trace:self.assertTrue(torch.equal(row['anchor_ids'],fixed.anchor_indices))
        x['tactile'].fill_(1);m(x)
        for row in m.trace:
            for ids,keep in zip(row['anchor_ids'],row['anchor_mask']):self.assertEqual(len(ids[keep].unique()),int(keep.sum()))
    def test_all_causal_and_gradients(self):
        x=self.active()
        for mode in CANDIDATES:
            m=self.nonzero(self.model(mode));source=copy.deepcopy(x);source['controller_points'].requires_grad_();source['tactile'].requires_grad_()
            out=m(source)['xyz'];state=m.trace[0]['current'];state.retain_grad()
            altered=copy.deepcopy(x);altered['controller_points'][:,3:]+=50;altered['tactile'][:,3:]+=20
            torch.testing.assert_close(out[:,:3],m(altered)['xyz'][:,:3],atol=0,rtol=0)
            torch.testing.assert_close(m(x)['xyz'],m(dict(x,target_xyz=torch.randn(5,8,3)))['xyz'],atol=0,rtol=0)
            out[:,-1].square().mean().backward()
            self.assertGreater(float(state.grad.abs().sum()),0,mode)
            self.assertGreater(float(source['controller_points'].grad.abs().sum()),0,mode)
            self.assertGreater(float(source['tactile'].grad.abs().sum()),0,mode)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    def test_f1_increments_not_cumulative_double_add(self):
        x=self.active();a=self.model();b=self.model(feedback='F1')
        with torch.no_grad():a.position_head[-1].bias.fill_(.01);b.position_head[-1].bias.fill_(.01)
        for m,factors in [(a,[1]*5),(b,[1,2,3,4,5])]:
            out=m(x)['xyz'];keep=x['gs_mask']
            for h,f in enumerate(factors):torch.testing.assert_close((out[:,h]-x['gs_xyz'])[keep],torch.full_like(out[:,h][keep],.01*f),atol=1e-6,rtol=1e-6)
    def test_se3_identity_translation_rotation_gradient(self):
        x=self.active();xyz=x['gs_xyz'];keep=x['gs_mask'];field=build_initial_anchor_field(xyz,keep,4,3)
        rotation=torch.zeros(*field.anchor_mask.shape,3,requires_grad=True);delta=torch.ones_like(rotation)*.02
        out,q=se3_decode(xyz,field,delta,rotation,x['gs_attr'][...,3:7],keep)
        torch.testing.assert_close(out[keep],(xyz+.02)[keep],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(q[keep].norm(dim=-1),torch.ones_like(q[keep][...,0]),atol=1e-6,rtol=1e-6)
        (out.square().sum()+q[...,1:].sum()).backward();self.assertTrue(torch.isfinite(rotation.grad).all());self.assertGreater(float(rotation.grad.abs().sum()),0)
    def test_r1_translation_and_deformation(self):
        x=self.active();v=x['gs_xyz'];keep=x['gs_mask'];neighbors=initial_knn(v,keep,3)
        p=(v[:,None]+.2).expand(-1,5,-1,-1)
        self.assertLess(float(local_penalty(p,v,keep,neighbors)),1e-10)
        p=(v[:,None]*2).expand(-1,5,-1,-1)
        self.assertGreater(float(local_penalty(p,v,keep,neighbors)),0)
    def test_d1_no_contact_zero(self):
        x=self.active();x['tactile'].zero_();a=self.nonzero(self.model('S4-C1'));b=self.model('S4-C1',decoder='D1');b.load_state_dict(a.state_dict(),strict=False)
        with torch.no_grad():b.deformation_head[-1].bias.fill_(.5)
        torch.testing.assert_close(a(x)['xyz'],b(x)['xyz'],atol=0,rtol=0)
    def test_checkpoint_config_and_rng(self):
        with tempfile.TemporaryDirectory() as d:
            m=self.model();opt=torch.optim.AdamW(m.parameters());p=Path(d)/'last.pt'
            save_checkpoint(p,m,opt,self.processor,'abc',20,[]);expected=torch.rand(2)
            self.assertEqual(restore(p,m,opt,self.processor,'abc'),20);self.assertTrue(torch.equal(expected,torch.rand(2)))
            for altered in [self.model('S1'),self.model(decoder='D2'),self.model(feedback='F1')]:
                with self.assertRaises(ValueError):restore(p,altered,opt,self.processor,'abc')
    def test_perturbation_and_reproducible_order(self):
        x=self.active();zero=perturb(x,'zero_tactile')
        self.assertTrue(torch.equal(zero['valid_mask'],x['valid_mask']))
        frozen=perturb(x,'freeze_action');self.assertTrue(torch.equal(frozen['controller_points'][:,0],frozen['controller_points'][:,-1]))
        self.assertEqual(update_order(200),update_order(1200)[:200]);self.assertEqual(sorted(update_order(24)),list(range(24)))
    def test_long_horizon_shapes(self):
        for h in [10,15]:
            x=self.active()
            for k in ('action_state','action_valid','controller_points','controller_mask','tactile','valid_mask','tactile_spatial_mask','tactile_age_us'):
                x[k]=x[k][:,:1].expand(-1,h,*x[k].shape[2:])
            m=SpatialPredictor(spatial='S2-C1',**dict(self.arch,horizon=h));out=m(x)
            self.assertEqual(out['xyz'].shape[1],h)

if __name__=='__main__':unittest.main()
