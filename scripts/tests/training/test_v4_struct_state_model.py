"""CPU contracts for paired G5/Anchor-state variants; no performance claims."""
import copy
import io
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_struct_state.model import (
    StructStatePredictor, build_model, fresh_model, paired_initialization, VARIANTS, ARCHITECTURES)
from scripts.experiments.v4_g5_struct_state.basis import observed_signed_basis
from scripts.experiments.v4_g4_global_motion.smoke import same_tree


class StructStateModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def inputs(self, model, points=50):
        return prepare_batch(synthetic_batch(points,model.config['horizon'],1),
                             DataProcessor('001-rope'),model)['inputs']

    def model(self, variant='g6_state_struct', horizon=5, active=True):
        torch.manual_seed(42)
        model=build_model(variant,horizon=horizon)
        if active:
            with torch.no_grad():
                model.motion[-1].weight.normal_(std=.002)
                model.global_motion.mapping[-1].weight.normal_(std=.001)
                model.direct_motion.mapping[-1].weight.normal_(std=.002)
        return model

    def test_formal_pair_initialization_counts_rng_and_h15_exact(self):
        evidence=paired_initialization(True)
        self.assertTrue(evidence['shared_exact'])
        self.assertTrue(evidence['external_cpu_rng_equal'])
        self.assertEqual(evidence['parameter_counts'],{'g5_struct':620511,'g6_state_struct':775353})
        self.assertEqual(evidence['added_state_parameters'],154842)
        self.assertTrue(all(all(row.values()) for row in evidence['h15_all_attributes_exact'].values()))
        generator=torch.Generator().manual_seed(913)
        before=generator.get_state().clone()
        fresh_model('g6_state_struct')
        self.assertTrue(torch.equal(before,generator.get_state()))

    def test_nontrivial_h15_A_original_and_B_zero_readout_bit_exact(self):
        torch.manual_seed(42);original=G5DirectPredictor(horizon=15,prediction_interval=1)
        with torch.no_grad():
            original.motion[-1].weight.normal_(std=.002)
            original.global_motion.mapping[-1].weight.normal_(std=.001)
            original.direct_motion.mapping[-1].weight.normal_(std=.002)
        a=fresh_model(VARIANTS[0]);b=fresh_model(VARIANTS[1])
        a.load_state_dict(original.state_dict(),strict=True)
        shared=b.state_dict()
        for k,v in original.state_dict().items():shared[k]=v.clone()
        b.load_state_dict(shared,strict=True)
        x=self.inputs(original,60)
        with torch.no_grad():
            out=original(x)
            self.assertTrue(same_tree(out,a(x)))
            self.assertTrue(same_tree(out,b(x)))
        self.assertGreater(float((out['xyz']-x['gs_xyz'][:,None]).abs().max()),0)

    def test_pure_signed_basis_equals_original_with_active_heads_and_pathological_masks(self):
        original=self.model('g5_struct',horizon=5)
        x=self.inputs(original)
        # Both current and reference endpoint policies matter. Invalid data must
        # be ignored without accidentally suppressing still-valid root motion.
        x['controller_mask'][1,0,1,:4]=False
        x['controller_points'][1,0,1,:4]=float('nan')
        x['action_valid'][1,1,0]=False
        x['action_state'][1,1,0]=float('nan')
        x['controller_mask'][1,2,1,7:15]=False
        x['controller_points'][1,2,1,7:15]=float('nan')
        x['action_state'][0,:,0]=float('nan')
        x['controller_points'][0,:,0]=float('nan')
        reference=G5DirectPredictor(horizon=5)
        reference.load_state_dict(original.state_dict(),strict=True)
        with torch.no_grad():_,trace=reference(x,return_trace=True)
        previous_cp=previous_valid=None
        for step,s in enumerate(trace['steps']):
            _,valid,cp,_=reference.conditions(x,step)
            basis12,hand_valid,basis18,weights=observed_signed_basis(x,step,
                anchor_xyz=s['anchor_xyz'],anchor_valid=trace['field'].anchor_mask,
                cp=cp,cp_valid=valid,previous_cp=previous_cp,previous_valid=previous_valid,
                sigma=trace['field'].bandwidth)
            self.assertTrue(torch.equal(basis12,s['hand_basis12']))
            self.assertTrue(torch.equal(hand_valid,s['hand_valid']))
            self.assertTrue(torch.equal(basis18,s['anchor_hand_basis18']))
            self.assertTrue(torch.equal(weights['controller_cumulative_weights'],s['controller_cumulative_weights']))
            self.assertTrue(torch.equal(weights['controller_adjacent_weights'],s['controller_adjacent_weights']))
            previous_cp,previous_valid=cp,valid

    def test_state_basis_has_no_current_learned_global_dependency(self):
        model=self.model(horizon=3);x=self.inputs(model)
        with torch.no_grad():
            _,before=model(x,return_trace=True)
            # H1 shares exactly the same AR history no matter what Global does.
            model.global_motion.mapping[-1].weight.mul_(77)
            _,after=model(x,return_trace=True)
        self.assertTrue(torch.equal(before['steps'][0]['state_basis18'],after['steps'][0]['state_basis18']))
        for trace in (before,after):
            for s in trace['steps']:
                self.assertTrue(torch.equal(s['state_basis18'],s['anchor_hand_basis18']))
                self.assertTrue(torch.equal(s['state_hand_basis12'],s['hand_basis12']))

    def test_state_precedes_heads_and_actual_enhanced_global_context_reaches_local(self):
        model=self.model(horizon=3);x=self.inputs(model)
        with torch.no_grad():model.state_memory.readout.weight.normal_(std=.01)
        events=[];motion_queries=[];global_queries=[];local_queries=[];contributions=[]
        def memory_pre(module,args):events.append('state')
        def motion_pre(module,args):events.append('base');motion_queries.append(args[0][...,:128])
        def global_pre(module,args,kwargs):events.append('global');global_queries.append(kwargs['query'])
        def global_post(module,args,output):contributions.append(output[1]['hand_contribution'])
        def local_pre(module,args,kwargs):
            events.append('local');local_queries.append(kwargs['query'])
            self.assertIs(kwargs['global_hand_contribution'],contributions[-1])
        hooks=[model.state_memory.register_forward_pre_hook(memory_pre),model.motion.register_forward_pre_hook(motion_pre),
            model.global_motion.register_forward_pre_hook(global_pre,with_kwargs=True),
            model.global_motion.register_forward_hook(global_post),
            model.direct_motion.register_forward_pre_hook(local_pre,with_kwargs=True)]
        try:
            with torch.no_grad():_,trace=model(x,return_trace=True)
        finally:
            for hook in hooks:hook.remove()
        self.assertEqual(events,['state','base','global','local']*3)
        for i,s in enumerate(trace['steps']):
            self.assertTrue(torch.equal(motion_queries[i],s['enhanced_query']))
            self.assertIs(global_queries[i],s['enhanced_query'])
            self.assertIs(local_queries[i],s['enhanced_query'])
        self.assertGreater(float((trace['steps'][0]['enhanced_query']-trace['steps'][0]['base_query']).abs().max()),0)

    def test_state_input_301_semantics_Z0_mask_and_history(self):
        model=self.model();x=self.inputs(model)
        starts=[]
        hook=model.state_memory.register_forward_pre_hook(lambda m,args:starts.append(args[1]))
        try:out,trace=model(x,return_trace=True)
        finally:hook.remove()
        valid=trace['field'].anchor_mask
        self.assertTrue(torch.equal(starts[0],trace['steps'][0]['object_anchor_features']))
        for k,s in enumerate(trace['steps']):
            v=s['state_input'];self.assertEqual(v.shape,(2,128,301))
            self.assertTrue(torch.equal(v[...,:128],s['object_anchor_features']))
            self.assertTrue(torch.equal(v[...,128:256],s['base_query']))
            self.assertTrue(torch.equal(v[...,256:292],s['state_basis18'].flatten(-2)))
            self.assertTrue(torch.equal(v[...,292:295],s['anchor_history']))
            self.assertTrue(torch.equal(v[...,295:298],s['previous_d']))
            self.assertTrue(torch.equal(v[...,298:300],
                s['hand_valid'][:,None].expand(-1,128,-1).to(v.dtype)*valid[...,None]))
            self.assertTrue(torch.equal(v[...,300],valid.to(v.dtype)*(k>0)))
            self.assertEqual(int(s['state_z'][~valid].count_nonzero()),0)
            self.assertEqual(int(v[~valid].count_nonzero()),0)
            if k:self.assertIs(starts[k],trace['steps'][k-1]['state_z'])

    def test_single_double_missing_hand_and_nan_are_safe(self):
        model=self.model();x=self.inputs(model)
        with torch.no_grad():
            model.state_memory.readout.weight.normal_(std=.01)
            clean=model(x)
            altered=copy.deepcopy(x)
            altered['action_state'][0,:,0]=float('nan')
            altered['controller_points'][0,:,0]=float('nan')
            altered['tactile'][0,:,:2]=float('nan')
            output,trace=model(altered,return_trace=True)
        self.assertTrue(same_tree(clean,output))
        for s in trace['steps']:
            self.assertEqual(int(s['state_basis18'][0,:,0].count_nonzero()),0)
            self.assertTrue(torch.isfinite(s['state_z']).all())
        missing=copy.deepcopy(x);missing['operation_hands'][:]=False
        missing['action_state'][:]=float('nan');missing['controller_points'][:]=float('nan')
        with torch.no_grad():out,trace=model(missing,return_trace=True)
        self.assertTrue(torch.isfinite(out['xyz']).all())
        self.assertTrue(all(s['state_basis18'].count_nonzero()==0 for s in trace['steps']))

    def test_attributes_stay_initial_no_geometry_or_a2a(self):
        for variant in VARIANTS:
            model=self.model(variant);x=self.inputs(model)
            if model.has_state:
                with torch.no_grad():model.state_memory.readout.weight.normal_(std=.01)
            with torch.no_grad():out=model(x)
            for key,begin,end in (('log_scales',0,3),('quaternions',3,7),('opacity_logits',7,8),('sh_dc',8,11)):
                expected=torch.where(x['gs_mask'][...,None],x['gs_attr'],0.)[:,None,:,begin:end].expand_as(out[key])
                self.assertTrue(torch.equal(expected,out[key]))
            self.assertFalse(hasattr(model,'propagation'))
            self.assertFalse(hasattr(model,'geometry_head'))

    def test_window_state_resets_and_future_input_prefix_is_causal(self):
        model=self.model();x=self.inputs(model)
        with torch.no_grad():model.state_memory.readout.weight.normal_(std=.01)
        future=copy.deepcopy(x)
        for key in ('action_state','controller_points','tactile'):future[key][:,3:]+=0.02
        future['future_gt_xyz']=torch.full_like(x['gs_xyz'],1e4)
        with torch.no_grad():
            first=model(x);changed=model(future);again=model(x)
        self.assertTrue(same_tree(first,again))
        for key in first:self.assertTrue(torch.equal(first[key][:,:3],changed[key][:,:3]),key)
        self.assertGreater(float((first['xyz'][:,3:]-changed['xyz'][:,3:]).abs().max()),0)

    def test_legal_past_state_influences_future_when_readout_nonzero(self):
        model=self.model();x=self.inputs(model)
        with torch.no_grad():model.state_memory.readout.weight.normal_(std=.02)
        calls=[0]
        def perturb(module,args,output):
            index=calls[0];calls[0]+=1
            return output+.2 if index==1 else output
        with torch.no_grad():baseline=model(x)
        hook=model.state_memory.cell.register_forward_hook(perturb)
        try:
            with torch.no_grad():changed=model(x)
        finally:hook.remove()
        self.assertTrue(torch.equal(baseline['xyz'][:,:1],changed['xyz'][:,:1]))
        self.assertGreater(float((baseline['xyz'][:,2:]-changed['xyz'][:,2:]).abs().max()),0)

    def test_projection_then_gru_receive_gradients_no_detach(self):
        model=self.model();x=self.inputs(model)
        out,trace=model(x,return_trace=True)
        loss=(out['xyz']-x['gs_xyz'][:,None]-.01).square().mean();loss.backward()
        projection=model.state_memory.readout.weight.grad
        self.assertTrue(torch.isfinite(projection).all());self.assertGreater(float(projection.abs().sum()),0)
        self.assertEqual(sum(float(p.grad.abs().sum()) for p in model.state_memory.cell.parameters() if p.grad is not None),0.)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():model.state_memory.readout.weight.add_(-1e-3*projection)
        out,trace=model(x,return_trace=True)
        early=trace['steps'][0]['state_z'];early.retain_grad()
        out['xyz'][:,-1].square().sum().backward()
        grad=sum(float(p.grad.abs().sum()) for p in model.state_memory.cell.parameters() if p.grad is not None)
        self.assertGreater(grad,0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIsNotNone(early.grad);self.assertGreater(float(early.grad.abs().sum()),0)

    def test_fresh_zero_motion_heads_state_branch_activates_within_four_updates(self):
        torch.manual_seed(42);model=build_model('g6_state_struct',horizon=4)
        x=self.inputs(model);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        target=x['gs_xyz'][:,None]+torch.tensor([.01,-.02,.03])
        history=[]
        for step in range(4):
            opt.zero_grad(set_to_none=True);out=model(x)
            (out['xyz']-target).square().mean().backward()
            projection=sum(float(p.grad.abs().sum()) for p in model.state_memory.readout.parameters() if p.grad is not None)
            cell=sum(float(p.grad.abs().sum()) for p in model.state_memory.cell.parameters() if p.grad is not None)
            history.append((projection,cell))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            opt.step()
        self.assertGreater(max(p for p,c in history),0)
        self.assertGreater(history[-1][1],0)

    def test_frozen_readout_off_equals_same_shared_network_not_retrained_A(self):
        b=self.model()
        with torch.no_grad():b.state_memory.readout.weight.normal_(std=.02)
        a=build_model('g5_struct',horizon=5)
        a.load_state_dict({k:v for k,v in b.state_dict().items() if not k.startswith('state_memory.')},strict=True)
        x=self.inputs(b)
        with torch.no_grad():
            self.assertTrue(same_tree(a(x),b(x,state_readout_off=True)))
            self.assertFalse(torch.equal(b(x)['xyz'],b(x,state_readout_off=True)['xyz']))
        with self.assertRaises(ValueError):a(x,state_readout_off=True)

    def test_model_state_strict_restore_and_architectures_differ(self):
        model=self.model();x=self.inputs(model)
        with torch.no_grad():model.state_memory.readout.weight.normal_(std=.01);expected=model(x)
        stream=io.BytesIO();torch.save(dict(config=model.config,state=model.state_dict()),stream)
        stream.seek(0);state=torch.load(stream,weights_only=False)
        restored=build_model('g6_state_struct',horizon=5)
        self.assertEqual(state['config'],restored.config);restored.load_state_dict(state['state'],strict=True)
        with torch.no_grad():self.assertTrue(same_tree(expected,restored(x)))
        a=build_model('g5_struct',horizon=5)
        self.assertNotEqual(a.config['architecture'],restored.config['architecture'])
        with self.assertRaises(RuntimeError):a.load_state_dict(state['state'],strict=True)
        with self.assertRaises(RuntimeError):restored.load_state_dict(a.state_dict(),strict=True)
        with self.assertRaises(ValueError):build_model('unknown')


if __name__=='__main__':unittest.main()
