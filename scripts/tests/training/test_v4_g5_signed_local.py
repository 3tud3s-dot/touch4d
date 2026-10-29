"""CPU architecture contracts; these tests are not G5 performance evidence."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.training.v4_model import V4Stage1Predictor
from scripts.training.v4_runtime import prepare_batch
from scripts.training.gs_decoder import interpolate_anchor_displacements
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor,G5PropagatePredictor
from scripts.experiments.v4_g5_signed_local.motion import AnchorSignedDirect
from scripts.experiments.v4_g5_signed_local.checkpoint import save_checkpoint,load_checkpoint


class G5SignedLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def inputs(self,model,points=64):
        return prepare_batch(synthetic_batch(points,model.config['horizon'],model.config['prediction_interval']),
                             DataProcessor('001-rope'),model)['inputs']

    def activated(self,cls,horizon=5):
        torch.manual_seed(224)
        model=cls(horizon=horizon)
        with torch.no_grad():
            model.motion[-1].weight.normal_(std=.002)
            model.direct_motion.mapping[-1].weight.normal_(std=.004)
        return model

    def test_h15_shared_initialization_and_all_outputs_bit_exact(self):
        torch.manual_seed(42);g4=G4GlobalMotionPredictor(horizon=15)
        torch.manual_seed(42);direct=G5DirectPredictor(horizon=15)
        torch.manual_seed(42);prop=G5PropagatePredictor(horizon=15)
        for name,value in g4.state_dict().items():
            torch.testing.assert_close(value,direct.state_dict()[name],rtol=0,atol=0)
            torch.testing.assert_close(value,prop.state_dict()[name],rtol=0,atol=0)
        for name,value in direct.direct_motion.state_dict().items():
            torch.testing.assert_close(value,prop.direct_motion.state_dict()[name],rtol=0,atol=0)
        for model in (direct,prop):
            self.assertTrue(torch.all(model.direct_motion.mapping[-1].weight==0))
            self.assertTrue(torch.all(model.direct_motion.mapping[-1].bias==0))
        # Avoid a trivial Copy-Initial equivalence proof.
        with torch.no_grad():
            g4.motion[-1].weight.normal_(std=.002)
            g4.global_motion.mapping[-1].weight.normal_(std=.001)
            for model in (direct,prop):
                model.motion[-1].weight.copy_(g4.motion[-1].weight)
                model.global_motion.load_state_dict(g4.global_motion.state_dict())
            x=self.inputs(g4,60);baseline=g4(x)
            for model in (direct,prop):
                output,trace=model(x,return_trace=True)
                for key in baseline:torch.testing.assert_close(output[key],baseline[key],rtol=0,atol=0)
                self.assertEqual(output['xyz'].shape,(2,15,60,3))
                self.assertEqual(trace['field'].anchor_mask.shape,(2,128))
                for step in trace['steps']:
                    self.assertEqual(step['anchor_hand_basis18'].shape,(2,128,2,18))
                    self.assertTrue(torch.all(step['local_correction']==0))
                    torch.testing.assert_close(step['g4_global_delta'],step['global_delta'],rtol=0,atol=0)
        self.assertFalse(hasattr(direct,'propagation'))
        self.assertTrue(hasattr(prop,'propagation'))
        self.assertNotEqual(direct.config['architecture'],prop.config['architecture'])
        self.assertNotEqual(direct.config['checkpoint_kind'],prop.config['checkpoint_kind'])

    def test_opposite_signed_xyz_preserved_per_hand_before_sum(self):
        for cls in (G5DirectPredictor,G5PropagatePredictor):
            model=cls(horizon=5);x=self.inputs(model)
            velocity=torch.tensor([.004,-.002,.003])
            for hand,sign in ((0,1),(1,-1)):
                for t in range(5):
                    x['action_state'][1,t,hand,:3]=x['action_state'][1,0,hand,:3]+sign*t*velocity
                    x['controller_points'][1,t,hand]=x['controller_points'][1,0,hand]+sign*t*velocity
            with torch.no_grad():
                bias=model.direct_motion.mapping[-1].bias.reshape(3,18)
                bias[0,0]=1;bias[1,1]=1;bias[2,2]=1
                _,trace=model(x,return_trace=True)
            s=trace['steps'][2];basis=s['anchor_hand_basis18'];seeds=s['direct_seed_per_hand']
            valid=trace['field'].anchor_mask[1]
            for hand,sign in ((0,1),(1,-1)):
                expected=sign*2*velocity
                torch.testing.assert_close(basis[1,valid,hand,:3],expected.expand(valid.sum(),3),atol=1e-6,rtol=0)
                torch.testing.assert_close(basis[1,valid,hand,12:15],expected.expand(valid.sum(),3),atol=1e-6,rtol=0)
                torch.testing.assert_close(basis[1,valid,hand,15:],(sign*velocity).expand(valid.sum(),3),atol=1e-6,rtol=0)
                torch.testing.assert_close(seeds[1,valid,hand],expected.expand(valid.sum(),3),atol=1e-6,rtol=0)
                self.assertGreater(float(seeds[1,valid,hand].abs().max()),0.)

    def test_h1_stationary_and_no_hand_local_correction_are_zero(self):
        for cls in (G5DirectPredictor,G5PropagatePredictor):
            model=self.activated(cls);x=self.inputs(model)
            with torch.no_grad():_,trace=model(x,return_trace=True)
            self.assertTrue(torch.all(trace['steps'][0]['anchor_hand_basis18']==0))
            self.assertTrue(torch.all(trace['steps'][0]['local_correction']==0))
            stationary=copy.deepcopy(x)
            stationary['action_state'][:]=stationary['action_state'][:,:1]
            stationary['controller_points'][:]=stationary['controller_points'][:,:1]
            with torch.no_grad():_,trace=model(stationary,return_trace=True)
            self.assertTrue(all(torch.all(s['local_correction']==0) for s in trace['steps']))
            missing=copy.deepcopy(x);missing['operation_hands'][:]=False
            missing['action_state'][:]=float('nan');missing['controller_points'][:]=float('nan')
            with torch.no_grad():out,trace=model(missing,return_trace=True)
            self.assertTrue(torch.isfinite(out['xyz']).all())
            self.assertTrue(all(torch.all(s['local_correction']==0) for s in trace['steps']))

    def test_masked_hand_cannot_leak_with_active_local_and_propagation(self):
        for cls in (G5DirectPredictor,G5PropagatePredictor):
            model=self.activated(cls);x=self.inputs(model)
            with torch.no_grad():before=model(x)
            altered=copy.deepcopy(x)
            altered['action_state'][0,:,0]=float('nan')
            altered['controller_points'][0,:,0]=float('nan')
            altered['tactile'][0,:,:2]=float('nan')
            with torch.no_grad():after,trace=model(altered,return_trace=True)
            for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            self.assertTrue(all(torch.all(s['direct_seed_per_hand'][0,:,0]==0) for s in trace['steps']))
            if cls is G5PropagatePredictor:
                self.assertTrue(all(torch.all(s['propagated_seed_per_hand'][0,:,0]==0) for s in trace['steps']))

    def test_missing_controller_endpoints_and_nan_isolation(self):
        model=self.activated(G5DirectPredictor);x=self.inputs(model)
        x['controller_mask'][:,0]=False;x['controller_points'][:,0]=float('nan')
        x['controller_mask'][:,2,:,:,]=False;x['controller_points'][:,2]=float('nan')
        with torch.no_grad():out,trace=model(x,return_trace=True)
        self.assertTrue(torch.isfinite(out['xyz']).all())
        self.assertTrue(all(torch.all(s['anchor_hand_basis18'][...,12:15]==0) for s in trace['steps']))
        self.assertTrue(torch.all(trace['steps'][2]['anchor_hand_basis18'][...,15:]==0))
        self.assertTrue(torch.all(trace['steps'][3]['anchor_hand_basis18'][...,15:]==0))
        # Root-based motion can still be used when controllers are unavailable.
        x=self.inputs(model);x['controller_mask'][:]=False;x['controller_points'][:]=float('nan')
        with torch.no_grad():out,trace=model(x,return_trace=True)
        self.assertTrue(torch.isfinite(out['xyz']).all())
        self.assertTrue(all(torch.all(s['anchor_hand_basis18'][...,6:]==0) for s in trace['steps']))
        self.assertGreater(float(trace['steps'][-1]['anchor_hand_basis18'][...,0].abs().max()),0)

    def test_future_conditions_and_fake_gt_cannot_change_prefix(self):
        for cls in (G5DirectPredictor,G5PropagatePredictor):
            model=self.activated(cls);x=self.inputs(model)
            with torch.no_grad():before=model(x)
            future=copy.deepcopy(x)
            future['action_state'][:,3:,:,:3]+=.03
            future['controller_points'][:,3:]+=.03
            future['tactile'][:,3:]+=100
            future['future_gt_xyz']=torch.full_like(x['gs_xyz'],999.)
            with torch.no_grad():after=model(future)
            torch.testing.assert_close(before['xyz'][:,:3],after['xyz'][:,:3],rtol=0,atol=0)
            self.assertGreater(float((before['xyz'][:,3:]-after['xyz'][:,3:]).abs().max()),0)
            unrelated=copy.deepcopy(x);unrelated['future_gt_xyz']=torch.randn(2,5,100,3)
            with torch.no_grad():unchanged=model(unrelated)
            for key in before:torch.testing.assert_close(before[key],unchanged[key],rtol=0,atol=0)

    def direct_context(self):
        b,a=1,3
        initial=torch.zeros(b,2,30,3);initial[0,1,:15,0]=-1;initial[0,1,15:,0]=1
        current=initial.clone();current[0,1,:15,0]+=.1;current[0,1,15:,0]-=.1
        hand_valid=torch.tensor([[False,True]])
        x=dict(controller_points=initial[:,None].repeat(1,2,1,1,1),
               controller_mask=torch.ones(b,2,2,30,dtype=torch.bool),
               action_state=torch.zeros(b,2,2,13),action_valid=torch.ones(b,2,2,dtype=torch.bool),
               operation_hands=hand_valid)
        x['controller_points'][:,1]=current
        return x,dict(query=torch.zeros(b,a,128),anchor_xyz=torch.tensor([[[-.9,0.,0.],[.9,0.,0.],[4.,0.,0.]]]),
            anchor_valid=torch.ones(b,a,dtype=torch.bool),hand_tokens=torch.zeros(b,2,128),
            hand_basis12=torch.zeros(b,2,12),hand_valid=hand_valid,cp=current.reshape(b,60,3),
            cp_valid=hand_valid.repeat_interleave(30,-1),previous_cp=initial.reshape(b,60,3),
            previous_valid=hand_valid.repeat_interleave(30,-1),anchor_history=torch.zeros(b,a,3),
            previous_d=torch.zeros(b,a,3),global_hand_contribution=torch.zeros(b,2,3),sigma=torch.tensor([.1]))

    def test_direct_geometry_independent_heads_and_signed_spatial_variation(self):
        direct=AnchorSignedDirect();x,kwargs=self.direct_context()
        with torch.no_grad():direct.mapping[-1].bias.reshape(3,18)[0,12]=1
        seeds,trace=direct(x,1,**kwargs)
        self.assertGreater(float(seeds[0,0,1,0]),0)
        self.assertLess(float(seeds[0,1,1,0]),0)
        # Far anchors still receive normalized geometry, with no threshold gate.
        torch.testing.assert_close(trace['controller_cumulative_weights'][0,:,1].sum(-1),torch.ones(3),rtol=1e-6,atol=1e-7)
        before=seeds.detach().clone();changed={k:v.clone() for k,v in kwargs.items()}
        changed['anchor_xyz'][0,0]=torch.tensor([.9,0.,0.]);changed['query'][0,0]=torch.randn(128)
        after,_=direct(x,1,**changed)
        torch.testing.assert_close(before[:,1:],after[:,1:],rtol=0,atol=0)
        self.assertNotEqual(float(before[0,0,1,0]),float(after[0,0,1,0]))
        self.assertFalse(any('attention' in n or 'propagat' in n for n,_ in direct.named_modules()))

    def test_partial_endpoint_weights_normalize_only_valid_same_hand_nodes(self):
        direct=AnchorSignedDirect();x,kwargs=self.direct_context()
        x['controller_mask'][0,0,1,0]=False
        x['controller_points'][0,0,1,0]=float('nan')
        kwargs['cp_valid'][0,59]=False
        kwargs['cp'][0,59]=float('nan')
        kwargs['previous_valid'][0,31]=False
        kwargs['previous_cp'][0,31]=float('nan')
        seed,trace=direct(x,1,**kwargs)
        self.assertTrue(torch.isfinite(seed).all())
        for key in ('controller_cumulative_weights','controller_adjacent_weights'):
            w=trace[key]
            self.assertTrue(torch.all(w[:,:,0]==0))
            torch.testing.assert_close(w[0,:,1].sum(-1),torch.ones(3),rtol=1e-6,atol=1e-7)
            self.assertTrue(torch.all(w[0,:,1,29]==0))
        self.assertTrue(torch.all(trace['controller_cumulative_weights'][0,:,1,0]==0))
        self.assertTrue(torch.all(trace['controller_adjacent_weights'][0,:,1,1]==0))

    def test_local_gs_mean_not_forced_zero_and_trace_matches_fixed_rbf(self):
        model=G5DirectPredictor(horizon=5);x=self.inputs(model)
        with torch.no_grad():model.direct_motion.mapping[-1].bias.reshape(3,18)[0,0]=1
        out,trace=model(x,return_trace=True);keep=x['gs_mask']
        for s in trace['steps'][1:]:
            gs=interpolate_anchor_displacements(s['local_correction'],trace['field'],keep)
            expected=(gs*keep[...,None]).sum(1)/keep.sum(1,keepdim=True)
            torch.testing.assert_close(expected,s['local_gs_mean'],rtol=0,atol=0)
            self.assertTrue((s['local_gs_mean'][:,0]>0).all())
            self.assertTrue((s['local_gs_demeaned_rms']<1e-7).all())
            torch.testing.assert_close(s['final_cumulative_anchor_displacement'],s['cumulative_d'],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(out['xyz']).all())

    def test_first_backward_reaches_direct_output_and_optimizer(self):
        for cls in (G5DirectPredictor,G5PropagatePredictor):
            model=cls(horizon=5);x=self.inputs(model)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            ids={id(p) for g in optimizer.param_groups for p in g['params']}
            self.assertTrue(all(id(p) in ids for p in model.direct_motion.parameters()))
            target=x['gs_xyz']+torch.tensor([.01,-.02,.03])
            pred=model(x)['xyz'];(pred[:,-1]-target).square().mean().backward()
            for p in model.direct_motion.mapping[-1].parameters():
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
            before=model.direct_motion.mapping[-1].weight.detach().clone();optimizer.step()
            self.assertFalse(torch.equal(before,model.direct_motion.mapping[-1].weight))

    def test_nonzero_direct_seed_reaches_propagation_scores(self):
        model=self.activated(G5PropagatePredictor);x=self.inputs(model)
        output=model(x)['xyz'];(output[:,-1]-x['gs_xyz']-torch.tensor([.01,-.02,.03])).square().mean().backward()
        grads=[p.grad for n,p in model.propagation.named_parameters() if n.endswith('weight')]
        self.assertTrue(grads);self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)

    def test_invalid_padding_and_controller_have_no_gradient_leak(self):
        for cls in (G5DirectPredictor,G5PropagatePredictor):
            model=self.activated(cls);x=self.inputs(model)
            x['gs_xyz']=x['gs_xyz'].clone().requires_grad_()
            x['controller_mask'][:,2,1,0]=False
            x['controller_points'][:,2,1,0]=float('nan')
            x['controller_points']=x['controller_points'].clone().requires_grad_()
            with torch.no_grad():x['gs_xyz'][~x['gs_mask']]=float('nan')
            output=model(x)['xyz'];output[:,-1].square().mean().backward()
            self.assertTrue(torch.isfinite(x['gs_xyz'].grad).all())
            self.assertTrue(torch.all(x['gs_xyz'].grad[~x['gs_mask']]==0))
            self.assertTrue(torch.isfinite(x['controller_points'].grad).all())
            self.assertTrue(torch.all(x['controller_points'].grad[:,2,1,0]==0))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_h15_graph_connectivity_without_detach(self):
        for cls in (G5DirectPredictor,G5PropagatePredictor):
            model=self.activated(cls,horizon=15);x=self.inputs(model,60)
            output,trace=model(x,return_trace=True)
            states=[s['state'] for s in trace['steps'][:-1]]
            gradients=torch.autograd.grad(output['xyz'][:,-1].square().sum(),states,allow_unused=True)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(float(gradients[-1].abs().sum()),0)
            # An existing but tiny/zero early gradient is not mistaken for detach.
            self.assertEqual(output['xyz'].shape[1],15)

    def test_strict_checkpoint_roundtrip_modes_time_and_legacy_rejection(self):
        for cls,other in ((G5DirectPredictor,G5PropagatePredictor),(G5PropagatePredictor,G5DirectPredictor)):
            model=self.activated(cls);x=self.inputs(model);processor=DataProcessor('001-rope').state_dict()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            model(x)['xyz'].square().mean().backward();optimizer.step()
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'g5.pt';save_checkpoint(path,model,optimizer,source_sha='test-sha',processor_state=processor,successful_steps=1)
                restored=cls(horizon=5);opt=torch.optim.AdamW(restored.parameters(),lr=1e-4)
                state=load_checkpoint(path,restored,opt,processor_state=processor,source_sha='test-sha')
                self.assertEqual(state['successful_steps'],1)
                for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
                self.assertEqual(optimizer.state_dict()['param_groups'],opt.state_dict()['param_groups'])
                for key,values in optimizer.state_dict()['state'].items():
                    for name,value in values.items():torch.testing.assert_close(value,opt.state_dict()['state'][key][name],rtol=0,atol=0)
                with torch.no_grad():before,after=model(x),restored(x)
                for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
                with self.assertRaises(ValueError):load_checkpoint(path,other(horizon=5))
                with self.assertRaises(ValueError):load_checkpoint(path,cls(horizon=15))
                with self.assertRaises(ValueError):load_checkpoint(path,cls(horizon=5,prediction_interval=2))
                with self.assertRaises(ValueError):load_checkpoint(path,restored,source_sha='different')
                with self.assertRaises(ValueError):load_checkpoint(path,restored,processor_state={})
                with self.assertRaises(FileExistsError):save_checkpoint(path,model,source_sha='test-sha')
                for legacy in (G4GlobalMotionPredictor(horizon=5),V4Stage1Predictor(horizon=5)):
                    version=('touch4d-g4-formal-h15-epochs-v1' if isinstance(legacy,G4GlobalMotionPredictor)
                             else 'touch4d-v4-stage1-smoke-v1')
                    old=Path(d)/'legacy.pt';torch.save(dict(version=version,config=legacy.config,model=legacy.state_dict()),old)
                    with self.assertRaises(ValueError):load_checkpoint(old,restored)
                    with self.assertRaises(ValueError):load_checkpoint(path,legacy)
                    with self.assertRaises(RuntimeError):restored.load_state_dict(legacy.state_dict(),strict=True)
                buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
                restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)


if __name__=='__main__':unittest.main()
