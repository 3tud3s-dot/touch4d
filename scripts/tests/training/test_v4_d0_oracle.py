import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch,analytic_rasterizer
from scripts.training.v4_model import V4Stage1Predictor,fixed_field
from scripts.training.v4_runtime import prepare_batch
from scripts.training.train import copy_initial
from scripts.training.losses import objective,chamfer_samples
from scripts.training.rendering import GaussianRenderer
from scripts.experiments.fixed_window.fixed_fit import digest
from scripts.experiments.v4_d0_oracle.core import decode,one_horizon,loss_parts,edges,structure
from scripts.experiments.v4_d0_oracle.io import metadata


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(42);self.m=V4Stage1Predictor(horizon=15).eval()
        torch.nn.init.normal_(self.m.motion[-1].weight,std=.001)
        self.m.requires_grad_(False);self.p=DataProcessor('001-rope')
        self.b=prepare_batch(synthetic_batch(points=32,horizon=15,prediction_interval=1),self.p,self.m)
        with torch.no_grad():self.n,self.t=self.m(self.b['inputs'],return_trace=True)
        self.f=self.t['field'];self.x=self.b['inputs']

    def test_zero_and_production_trace_exact(self):
        d=torch.zeros_like(self.t['steps'][0]['cumulative_d']);q=decode(self.x,self.f,d)
        self.assertTrue(torch.equal(q['xyz'][:,0],copy_initial(self.x,'gs_geometry')['xyz'][:,0]))
        for h in (5,10,15):self.assertTrue(torch.equal(decode(self.x,self.f,self.t['steps'][h-1]['cumulative_d'])['xyz'][:,0],self.n['xyz'][:,h-1]))

    def test_fixed_rebuild_and_attributes(self):
        fresh=fixed_field(self.x['gs_xyz'],self.x['gs_mask']);self.assertTrue(torch.equal(fresh.weights,self.f.weights))
        q=decode(self.x,self.f,torch.randn_like(self.t['steps'][0]['cumulative_d']))
        for k in q:
            if k!='xyz':self.assertTrue(torch.equal(q[k][:,0],self.n[k][:,0]))

    def test_horizon_slice(self):
        b=one_horizon(self.b,10)
        self.assertTrue(torch.equal(b['targets']['target_xyz'][0],self.b['targets']['target_xyz'][9]))
        self.assertTrue(torch.equal(b['targets']['target_depth'],self.b['targets']['target_depth'][:,9:10]))

    def test_metadata_distinguishes_oracle_and_network(self):
        m=metadata();self.assertNotIn('no_optimizer',m);self.assertNotIn('no_backward',m)
        self.assertTrue(m['no_network_optimizer']);self.assertEqual(m['optimizer_scope'],'oracle_anchor_displacements_only')

    def test_world_export_mask(self):
        p=decode(self.x,self.f,torch.zeros_like(self.t['steps'][0]['cumulative_d']))
        world=self.p.denormalize_xyz(p['xyz'],self.x['gs_mask'][:,None])
        reference=self.p.denormalize_xyz(self.x['gs_xyz'],self.x['gs_mask'])
        torch.testing.assert_close(world[:,0],reference)

    def test_gradient_only_d_and_model_hash(self):
        before=digest(self.m.state_dict());fhash=digest(self.f.__dict__)
        d=torch.nn.Parameter(torch.zeros_like(self.t['steps'][0]['cumulative_d']));opt=torch.optim.Adam([d],lr=.001)
        loss=chamfer_samples(decode(self.x,self.f,d)['xyz'],one_horizon(self.b,15)).mean();loss.backward()
        self.assertTrue(torch.isfinite(d.grad).all());self.assertGreater(float(d.grad.norm()),0);opt.step()
        self.assertEqual(before,digest(self.m.state_dict()));self.assertEqual(fhash,digest(self.f.__dict__))
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))

    def test_joint_matches_production_and_gradients(self):
        b=one_horizon(self.b,5);r=GaussianRenderer(analytic_rasterizer);target=r.render_targets(b,self.p)
        d=torch.nn.Parameter(torch.zeros_like(self.t['steps'][0]['cumulative_d']));pred=decode(self.x,self.f,d)
        loss,_=loss_parts(pred,b,self.p,r,target);ref,_=objective(pred,b,self.p,r,chamfer_weight=1,scale_weight=0)
        torch.testing.assert_close(loss,ref);a=torch.autograd.grad(loss,d,retain_graph=True)[0];c=torch.autograd.grad(ref,d)[0];torch.testing.assert_close(a,c)

    def test_neighborhood_translation_identity(self):
        # Metrics use the first sample; rigid translation must leave edge ratios at one.
        e=edges(self.x);s,heat=structure(self.x,self.x['gs_xyz'][0]+.1,e,self.p.state_dict()['scale'])
        self.assertAlmostEqual(s['ratio_mean'],1,places=5);self.assertLess(s['distortion_p95'],1e-5)

    def test_padding_and_target_order(self):
        d=torch.zeros_like(self.t['steps'][0]['cumulative_d']);d[~self.f.anchor_mask]=float('nan')
        pred=decode(self.x,self.f,d);self.assertTrue(torch.isfinite(pred['xyz']).all())
        b=one_horizon(self.b,5);loss=chamfer_samples(pred['xyz'],b)
        b['targets']=dict(b['targets']);b['targets']['target_xyz']=[b['targets']['target_xyz'][0].flip(1)];b['targets']['target_mask']=[b['targets']['target_mask'][0].flip(1)]
        torch.testing.assert_close(loss,chamfer_samples(pred['xyz'],b))


if __name__=='__main__':unittest.main()
