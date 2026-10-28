"""Independent interaction and unchanged-rollout contracts."""
import copy
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch, analytic_rasterizer
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import prepare_batch, v4_objective, save_checkpoint, load_checkpoint
from scripts.training.rendering import GaussianRenderer
from scripts.experiments.v4_anchor_interaction.model import paired_models, AnchorInteraction, HybridCross


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def batch(self, model, h=5):
        self.processor = DataProcessor('001-rope')
        return prepare_batch(synthetic_batch(points=32, horizon=h), self.processor, model)
    def activated(self, h=5):
        models = paired_models(h)
        for m in models.values():
            with torch.no_grad(): m.motion[-1].weight.fill_(.001)
        return models

    def test_shared_initial_parameters_and_pre_post_a2a_identical(self):
        m = paired_models()
        for tag in ('G1','G2','G3'):
            for key, value in m['G0'].state_dict().items():
                torch.testing.assert_close(m[tag].state_dict()[key], value, rtol=0, atol=0)
        for a,b in zip(m['G1'].cross[0].a2a.parameters(), m['G2'].cross[1].a2a.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_g0_bit_exact_original_nonzero_head_and_checkpoint(self):
        models=self.activated(); m=models['G0']; legacy=V4Stage1Predictor(horizon=5)
        legacy.load_state_dict(m.state_dict()); b=self.batch(m)
        torch.testing.assert_close(m(b['inputs'])['xyz'],legacy(b['inputs'])['xyz'],rtol=0,atol=0)
        self.assertNotIn('anchor_interaction',m.config)

    def test_self_keys_values_excluded_but_residual_present(self):
        m=AnchorInteraction();q=torch.randn(1,3,128);mem=q.clone().requires_grad_()
        xyz=torch.randn(1,3,3); keep=torch.ones(1,3,dtype=torch.bool)
        out,alpha=m.attend(q,mem,xyz,xyz,torch.ones(1),keep,keep,0)
        self.assertEqual(alpha.shape,(1,3,4,4))
        self.assertTrue(torch.all(alpha[:,torch.arange(3),:,torch.arange(3)]==0))
        g=torch.autograd.grad(out[:,0].square().sum(),mem)[0]
        self.assertTrue(torch.all(g[:,0]==0));self.assertGreater(float(g[:,1:].abs().sum()),0)
        with torch.no_grad():
            m.projection.weight.zero_()
            for p in m.ff.parameters():p.zero_()
        out,_=m(q,xyz,torch.ones(1),keep)
        torch.testing.assert_close(out,q,rtol=0,atol=0)

    def test_padding_and_single_anchor_null_finite(self):
        m=AnchorInteraction();q=torch.randn(1,4,128);xyz=torch.randn(1,4,3)
        keep=torch.tensor([[True,False,False,False]])
        out,a=m(q,xyz,torch.ones(1),keep)
        self.assertTrue(torch.isfinite(out).all());self.assertTrue(torch.all(a[...,-1]==1))
        q[:,1:]=float('nan');xyz[:,1:]=float('nan')
        changed,_=m(q,xyz,torch.ones(1),keep)
        torch.testing.assert_close(changed,out,rtol=0,atol=0)

    def test_hybrid_joint_softmax_types_and_no_controller_queries(self):
        m=HybridCross();q=torch.randn(1,3,128);tools=torch.randn(1,60,128)
        am=torch.ones(1,3,dtype=torch.bool);tm=torch.ones(1,60,dtype=torch.bool)
        out,a=m(q,tools,torch.zeros(1,3,3),torch.zeros(1,60,3),torch.ones(1),am,tm)
        self.assertEqual(out.shape,(1,3,128));self.assertEqual(a.shape,(1,3,4,64))
        torch.testing.assert_close(a.sum(-1),torch.ones(1,3,4))
        self.assertTrue(torch.all(a[:,torch.arange(3),:,60+torch.arange(3)]==0))
        torch.testing.assert_close(a[...,:60],torch.full((1,3,4,60),1/63))
        changed,_=m(q,tools*float('nan'),torch.zeros(1,3,3),torch.zeros(1,60,3),torch.ones(1),am,tm&False)
        self.assertTrue(torch.isfinite(changed).all())

    def test_full_h5_h10_causal_hand_mask_rbf_and_motion(self):
        for h in (5,10):
            models=self.activated(h);b=self.batch(models['G0'],h)
            for tag,m in models.items():
                pred,tr=m(b['inputs'],return_trace=True)
                self.assertEqual(pred['xyz'].shape,(2,h,32,3));self.assertTrue(torch.isfinite(pred['xyz']).all())
                torch.testing.assert_close(pred['log_scales'],b['inputs']['gs_attr'][:,None,:,:3].expand(-1,h,-1,-1))
                x=copy.deepcopy(b['inputs']);x['action_state'][0,:,0]=float('nan');x['controller_points'][0,:,0]=999
                x['tactile'][0,:,:2]=1e6
                torch.testing.assert_close(m(x)['xyz'][0],pred['xyz'][0],rtol=0,atol=0)
                x=copy.deepcopy(b['inputs']);x['controller_points'][:,3:]+=.1;x['tactile'][:,3:]+=2
                torch.testing.assert_close(m(x)['xyz'][:,:3],pred['xyz'][:,:3],rtol=0,atol=0)
                altered=copy.deepcopy(b);altered['targets']['target_xyz'][0]+=1
                torch.testing.assert_close(m(altered['inputs'])['xyz'],pred['xyz'],rtol=0,atol=0)
                for k,t in enumerate(tr['steps']):
                    cp=b['inputs']['controller_points'].reshape(2,h,60,3)
                    delta=torch.zeros_like(cp[:,k]) if k==0 else torch.where(t['adjacent_valid'][...,None],cp[:,k]-cp[:,k-1],0.)
                    expected=torch.einsum('bahj,bjd->bahd',t['alpha'][...,:60],delta).flatten(-2)
                    torch.testing.assert_close(t['tool_motion'],expected,rtol=0,atol=0)
                    expected=b['inputs']['gs_xyz']+tr['field'].weights@t['cumulative_d']
                    expected=expected.masked_fill(~b['inputs']['gs_mask'][...,None],0.)
                    torch.testing.assert_close(pred['xyz'][:,k],expected,rtol=0,atol=0)

    def test_gradients_rollout_real_objective_mock_renderer_and_optimizer(self):
        for tag,m in self.activated().items():
            b=self.batch(m);pred,tr=m(b['inputs'],return_trace=True)
            loss,_=v4_objective(pred,b,self.processor,GaussianRenderer(rasterizer=analytic_rasterizer))
            cross_grads=torch.autograd.grad(pred['xyz'][:,-1].square().mean(),[t['state'] for t in tr['steps']],retain_graph=True,allow_unused=True)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in cross_grads))
            loss.backward()
            for group in ('motion','gs_encoder','conditions.controller','cross'):
                ps=[p for n,p in m.named_parameters() if n.startswith(group)]
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in ps))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None),0)
            if tag in ('G1','G2'):
                block=m.cross[0 if tag=='G1' else 1].a2a
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None),0)
            opt=torch.optim.AdamW(m.parameters(),lr=1e-4);opt.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))

    def test_checkpoint_architecture_isolation_restore_optimizer(self):
        models=self.activated();b=self.batch(models['G0'])
        with tempfile.TemporaryDirectory() as tmp:
            for tag,m in models.items():
                opt=torch.optim.AdamW(m.parameters());m(b['inputs'])['xyz'].square().mean().backward();opt.step()
                f=Path(tmp)/f'{tag}.pt';save_checkpoint(f,m,opt,1,source_sha='fixture',processor_state={})
                clone=self.activated()[tag];co=torch.optim.AdamW(clone.parameters())
                load_checkpoint(f,clone,co,processor_state={})
                torch.testing.assert_close(clone(b['inputs'])['xyz'],m(b['inputs'])['xyz'],rtol=0,atol=0)
                other=models['G1' if tag!='G1' else 'G2']
                with self.assertRaises(ValueError):load_checkpoint(f,other)

    def test_sample_order_is_frozen_and_episode_means_are_equal(self):
        from scripts.experiments.v4_anchor_interaction.run import frozen_order, reduce_rows
        windows=[(1,i) for i in range(12)]+[(5,0)]
        self.assertEqual(frozen_order(windows),frozen_order(windows))
        self.assertEqual(len(frozen_order(windows)),1600)
        self.assertEqual(set(frozen_order(windows)[:13]),set(windows))
        rows=[]
        for ep,value in [(3,1.),(3,1.),(6,5.)]:
            rows.append(dict(episode=ep,start=0,cd_mm=[value]*10,rgb=[value]*10,alpha=[value]*10,motion_rms_mm=[value]*10,signed_mean_motion_mm=[[0.,0.,value]]*10,common_joint=value))
        self.assertEqual(reduce_rows(rows)['equal']['mean_cd_mm'],3.)

    def test_near_far_regions_and_rigid_residual(self):
        from scripts.experiments.v4_anchor_interaction.run import spatial
        m=paired_models(5)['G0'];b=self.batch(m);b={k:v for k,v in b.items()}
        # Spatial diagnostics intentionally require batch1, like the protocol.
        b['inputs']={k:v[:1] if isinstance(v,torch.Tensor) else v for k,v in b['inputs'].items()}
        b['targets']['target_xyz']=[q[:1] for q in b['targets']['target_xyz']]
        b['targets']['target_mask']=[q[:1] for q in b['targets']['target_mask']]
        xyz=b['inputs']['gs_xyz'];pred={'xyz':xyz[:,None].expand(-1,5,-1,-1)+xyz.new_tensor([.01,-.02,.03])}
        stats,regions=spatial(pred,b,1000.)
        self.assertEqual(len(regions['near']),8);self.assertEqual(len(regions['far']),8)
        self.assertLess(stats['proper_rigid_removed_rms_mm'],.001)
        self.assertAlmostEqual(stats['near_far_ratio'],1.,places=4)


if __name__=='__main__':unittest.main()
