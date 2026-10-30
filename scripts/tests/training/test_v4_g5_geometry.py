"""CPU mathematics and real H15 G5 output-adapter isolation contracts."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_geometry.model import GeometryAdapter
from scripts.experiments.v4_g5_geometry.geometry import (rotvec_quaternion, quaternion_multiply,
    rotation_matrix, covariance, adapt_attributes, LOG_SCALE_BOUND)
from scripts.experiments.v4_g5_geometry import checkpoint
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.training.rendering import GaussianRenderer
from scripts.experiments.v4_g5_geometry.evaluation import summarize,assert_metrics_equal,equality


class GeometryMathTests(unittest.TestCase):
    def test_zero_identity_and_gradient(self):
        w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.equal(rotvec_quaternion(w), torch.tensor([1.,0,0,0], dtype=w.dtype)))
        self.assertTrue(torch.autograd.gradcheck(rotvec_quaternion, (w,)))

    def test_nonzero_gradcheck(self):
        w = torch.tensor([.1,-.2,.3], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(rotvec_quaternion, (w,)))

    def test_wxyz_known_axis(self):
        w = torch.tensor([0.,0.,math.pi/2], dtype=torch.float64)
        R = rotation_matrix(rotvec_quaternion(w))
        torch.testing.assert_close(R @ torch.tensor([1.,0,0], dtype=w.dtype), torch.tensor([0.,1,0],dtype=w.dtype))

    def test_world_left_multiply_nonidentity(self):
        d = rotvec_quaternion(torch.tensor([0.,.3,0.], dtype=torch.float64))
        q = rotvec_quaternion(torch.tensor([.2,0.,0.], dtype=torch.float64))
        actual = rotation_matrix(quaternion_multiply(d,q))
        torch.testing.assert_close(actual, rotation_matrix(d) @ rotation_matrix(q))
        self.assertGreater(float((actual-rotation_matrix(quaternion_multiply(q,d))).abs().max()), .01)

    def test_unit_and_double_cover(self):
        q = rotvec_quaternion(torch.randn(20,3,dtype=torch.float64))
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(20,dtype=q.dtype))
        torch.testing.assert_close(rotation_matrix(q), rotation_matrix(-q))

    def test_positive_covariance(self):
        c = covariance(torch.randn(12,3,dtype=torch.float64)-4,
                       rotvec_quaternion(torch.randn(12,3,dtype=torch.float64)))
        self.assertTrue(torch.isfinite(c).all())
        self.assertTrue((torch.linalg.eigvalsh(c)>0).all())

    def test_isotropic_rotation_invariant(self):
        scales=torch.full((7,3),-3.,dtype=torch.float64)
        q=rotvec_quaternion(torch.randn(7,3,dtype=torch.float64))
        torch.testing.assert_close(covariance(scales,q), torch.eye(3,dtype=q.dtype).expand(7,3,3)*math.exp(-6))

    def test_anisotropic_zero_rotation_gradient(self):
        w=torch.zeros(3,dtype=torch.float64,requires_grad=True)
        cov=covariance(torch.tensor([-1.,-2.,-3.],dtype=w.dtype),rotvec_quaternion(w))
        (cov[0,1]+2*cov[1,2]+3*cov[0,2]).backward()
        self.assertTrue(torch.isfinite(w.grad).all());self.assertTrue((w.grad.abs()>0).all())


class GeometryProtocolTests(unittest.TestCase):
    def test_episode_equal_not_window_equal(self):
        rows=[dict(mode='m',split='val',episode=ep,cd_mm=v,rgb=v,alpha=v,joint=v)
              for ep,count,v in ((3,55,1.),(6,45,3.)) for _ in range(count)]
        self.assertEqual(summarize(rows,'m')[-1]['joint'],2.)
        with self.assertRaises(AssertionError):summarize(rows[:-1],'m')

    def test_fixed_tolerances_reject(self):
        a=dict(cd_mm=2.4,rgb=.08,alpha=.07,joint=.1)
        assert_metrics_equal(a,a)
        for name,delta in (('cd_mm',.001),('rgb',1e-5),('alpha',1e-5),('joint',1e-5)):
            with self.assertRaises(AssertionError):assert_metrics_equal(a,dict(a,**{name:a[name]+delta}))

    def test_no_extra_update_protocol(self):
        from scripts.experiments.v4_g5_geometry.smoke import ORDER,BASE,BACKBONE_HASH
        self.assertEqual(ORDER,((2,141),(7,105),(2,141),(7,105)))
        self.assertEqual(len(BASE),40);self.assertEqual(len(BACKBONE_HASH),64)

    def test_resource_guard_and_remapped_uuid(self):
        from scripts.experiments.v4_g5_geometry.smoke import assigned_gpu_idle
        with patch('subprocess.check_output',side_effect=['GPU-assigned\n','']) as run:
            self.assertEqual(assigned_gpu_idle()['uuid'],'GPU-assigned')
            self.assertEqual(run.call_args_list[1].args[0][2],'GPU-assigned')
        with patch('subprocess.check_output',side_effect=['GPU-assigned\n','99999999\n']):
            with self.assertRaises(RuntimeError):assigned_gpu_idle()
        with patch('subprocess.check_output',return_value='GPU-a\nGPU-b\n'):
            with self.assertRaises(RuntimeError):assigned_gpu_idle()


class GeometryAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(42)
        self.model=GeometryAdapter(G5DirectPredictor(horizon=15,prediction_interval=1))
        self.processor=DataProcessor('001-rope')
        self.batch=prepare_batch(synthetic_batch(50,15,1),self.processor,self.model)
        self.x=self.batch['inputs']

    def assert_tree_equal(self,a,b):
        self.assertEqual(set(a),set(b))
        for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)

    def nonzero(self):
        with torch.no_grad(): self.model.head[-1].weight.normal_(0,.02);self.model.head[-1].bias.fill_(.02)

    def test_full_h15_off_and_zero_exact(self):
        with torch.no_grad():
            original=self.model.backbone(self.x)
            self.assert_tree_equal(original,self.model(self.x,mode='geometry_off'))
            self.assert_tree_equal(original,self.model(self.x,mode='geometry_on_zero'))
        self.assertEqual(tuple(original['xyz'].shape),(2,15,50,3))

    def test_zero_path_is_differentiable(self):
        pred=self.model(self.x,mode='geometry_on_zero')
        (pred['log_scales'].sum()+pred['quaternions'][...,1:].sum()).backward()
        grad=self.model.head[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad[:3].norm()),0);self.assertGreater(float(grad[3:].norm()),0)
        self.assertEqual(float(self.model.head[0].weight.grad.norm()),0)

    def test_nonzero_head_cannot_change_motion_or_appearance(self):
        original,t0=self.model(self.x,return_trace=True)
        self.nonzero();pred,t1=self.model(self.x,return_trace=True)
        for k in ('xyz','opacity_logits','sh_dc'):self.assertTrue(torch.equal(original[k],pred[k]),k)
        self.assert_tree_equal(t0['motion'],t1['motion'])
        for a,b in zip(t0['motion_trace']['steps'],t1['motion_trace']['steps']):
            for key in ('state','final_cumulative_anchor_displacement','g4_global_delta'):
                self.assertTrue(torch.equal(a[key],b[key]))
        self.assertFalse(torch.equal(original['log_scales'],pred['log_scales']))
        self.assert_tree_equal(t0['motion'],self.model(self.x,mode='geometry_off'))

    def test_training_mode_and_optimizer_isolation(self):
        self.model.train();self.assertTrue(self.model.head.training);self.assertFalse(self.model.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
        before=state_hash(self.model.backbone.state_dict());self.nonzero()
        opt=torch.optim.AdamW(self.model.head.parameters(),lr=1e-4,weight_decay=1e-4)
        pred=self.model(self.x);(pred['log_scales'].square().mean()+pred['quaternions'][...,1:].sum()).backward()
        self.assertGreater(float(self.model.head[0].weight.grad.norm()),0)
        opt.step();self.assertEqual(before,state_hash(self.model.backbone.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))

    def test_bounds_and_reference_initial(self):
        self.nonzero()
        with torch.no_grad(): self.model.head[-1].bias.fill_(100)
        pred,tr=self.model(self.x,return_trace=True);valid=self.x['gs_mask'][:,None].expand(-1,15,-1)
        delta=pred['log_scales']-tr['motion']['log_scales']
        self.assertLessEqual(float(delta[valid].abs().max()),LOG_SCALE_BOUND+1e-6)
        self.assertLessEqual(float(tr['geometry']['rotation_vector'][valid].norm(dim=-1).max()),math.pi/6+1e-6)
        torch.testing.assert_close(delta[:,0][self.x['gs_mask']],delta[:,-1][self.x['gs_mask']])

    def test_future_condition_causality(self):
        self.nonzero();a=self.model(self.x)
        x=copy.deepcopy(self.x);x['controller_points'][:,8:]+=2;x['action_state'][:,8:,:,:3]-=3
        b=self.model(x)
        for key in a:self.assertTrue(torch.equal(a[key][:,:8],b[key][:,:8]),key)

    def test_hook_removed_on_error(self):
        x=dict(self.x);x.pop('gs_xyz')
        with self.assertRaises(KeyError):self.model(x)
        self.assertEqual(len(self.model.backbone.cross[-1]._forward_hooks),0)

    def test_zero_mode_rejects_updated_head(self):
        self.nonzero()
        with self.assertRaises(ValueError):self.model(self.x,mode='geometry_on_zero')

    def test_padding_and_nan_isolated(self):
        self.nonzero();baseline=self.model(self.x)
        x=copy.deepcopy(self.x);mask=x['gs_mask'];x['gs_xyz'][~mask]=float('nan');x['gs_attr'][~mask]=float('nan')
        result=self.model(x);self.assert_tree_equal(result,baseline)
        for key,v in result.items():self.assertTrue(torch.isfinite(v).all(),key)

    def test_single_and_few_anchor(self):
        for n in (1,4):
            x=prepare_batch(synthetic_batch(n,15,1),self.processor,self.model)['inputs']
            self.nonzero();pred,trace=self.model(x,return_trace=True)
            self.assertTrue(torch.isfinite(pred['quaternions']).all())
            self.assertEqual(int(trace['motion_trace']['field'].anchor_mask[0].sum()),n)

    def test_fixed_rbf_constant_increment_and_padding(self):
        motion,tr,features=self.model.motion_forward(self.x);field=tr['field'];w=field.weights.clone()
        raw=torch.full((2,15,128,6),.2);raw[~field.anchor_mask[:,None,:,None].expand_as(raw)]=float('nan')
        pred,geo=adapt_attributes(raw,motion,field,self.x['gs_mask'])
        valid=self.x['gs_mask'][:,None].expand(-1,15,-1)
        torch.testing.assert_close(geo['log_scale_delta'][valid],torch.full_like(geo['log_scale_delta'][valid],LOG_SCALE_BOUND*math.tanh(.2)))
        self.assertTrue(torch.equal(w,field.weights));self.assertTrue(torch.isfinite(pred['quaternions']).all())

    def test_detached_inputs_and_head_gradient_only(self):
        pred,tr=self.model(self.x,return_trace=True)
        self.assertFalse(tr['detached_features'].requires_grad)
        self.assertFalse(tr['motion_trace']['field'].weights.requires_grad)
        pred['log_scales'].sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))

    def test_checkpoint_strict_restore_and_no_clobber(self):
        identity={'source_sha':'unit-test','backbone_sha256':'fixed'}
        opt=torch.optim.AdamW(self.model.head.parameters(),lr=1e-4)
        self.nonzero();pred=self.model(self.x);pred['log_scales'].sum().backward();opt.step()
        expected=self.model(self.x)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'geo.pt';checkpoint.save(path,self.model,opt,identity=identity,updates=1)
            with self.assertRaises(FileExistsError):checkpoint.save(path,self.model,opt,identity=identity,updates=1)
            with torch.no_grad():self.model.head[-1].bias.add_(1)
            state=checkpoint.load(path,self.model,opt,identity=identity)
            self.assertEqual(state['updates'],1);self.assert_tree_equal(expected,self.model(self.x))
            with self.assertRaises(ValueError):checkpoint.load(path,self.model,identity={'source_sha':'wrong'})
            self.model.geometry_config['ar_feedback']=True
            with self.assertRaises(ValueError):checkpoint.load(path,self.model,identity=identity)

    def test_original_loss_mock_render_forward_backward(self):
        def rasterizer(**args):
            # Explicit anisotropic, geometry-sensitive MOCK; real gsplat remains mandatory.
            c=covariance(args['scales'].log(),args['quats']).mean(0)
            signal=args['scales'].mean()+c[0,1]+2*c[1,2]+3*c[0,2]
            rgb=torch.sigmoid(signal*1000).expand(1,args['height'],args['width'],3)
            alpha=torch.sigmoid(signal*1000).expand(1,args['height'],args['width'],1)
            return torch.cat((rgb,torch.ones_like(alpha)*.5),-1),alpha,{}
        self.x['gs_attr'][...,:3]+=torch.tensor([0.,.4,-.2])
        pred=self.model(self.x);loss,metrics=v4_objective(pred,self.batch,self.processor,GaussianRenderer(rasterizer))
        loss.backward()
        grad=self.model.head[-1].weight.grad
        self.assertGreater(float(grad[:3].norm()),0);self.assertGreater(float(grad[3:].norm()),0)
        self.assertTrue(torch.isfinite(grad).all())
        torch.testing.assert_close(loss,(metrics['geometry']+metrics['rgb']+.1*metrics['alpha']).mean())


if __name__=='__main__':unittest.main()
