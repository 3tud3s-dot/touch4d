"""Frozen routing contracts. No optimizer, backward, training or real checkpoint."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.training.v4_runtime import prepare_batch
from scripts.training.model import masked
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_attention_routing_audit.routing import (
    MODES,MASS_MATCHED,replace_weights,validate,conditional,permutation,WEIGHT_ATOL)
from scripts.experiments.v4_g5_attention_routing_audit.interventions import intervention,replay_step


class WeightRoutingTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(731)
        valid=torch.ones(2,60,dtype=torch.bool);valid[0,:30]=False;valid[0,37]=False;valid[1,5:9]=False
        anchors=torch.tensor([[True,True,True,False],[True,False,True,True]])
        relative=torch.randn(2,4,60,3)
        geometry=torch.cat((relative,relative.square().sum(-1,keepdim=True)),-1)
        geo=torch.stack([(geometry*torch.randn(4)).sum(-1) for _ in range(4)],2)
        allowed=anchors[:,:,None,None]&valid[:,None,None,:]
        scores=torch.randn(2,4,4,60).masked_fill(~allowed,-torch.inf)
        alpha=torch.cat((scores,torch.zeros(2,4,4,1)),-1).softmax(-1)
        return alpha,valid,anchors,geometry,geo

    def test_exact_seven_modes_and_normal_identity(self):
        self.assertEqual(len(MODES),7)
        a,v,m,g,s=self.fixture()
        self.assertIs(replace_weights(a,v,m,g,s,'Normal',0),a)

    def test_every_mode_mask_null_padding_and_row_sum(self):
        a,v,m,g,s=self.fixture()
        for mode in MODES:
            b=replace_weights(a,v,m,g,s,mode,0);validate(a,b,v,m,mode)
            self.assertTrue(torch.isfinite(b).all())
            torch.testing.assert_close(b.sum(-1),torch.ones_like(b[...,0]),rtol=0,atol=WEIGHT_ATOL)
            self.assertEqual(int(b[0,:,:,:30].count_nonzero()),0)

    def test_uniform_raw_includes_null_equally(self):
        a,v,m,g,s=self.fixture();b=replace_weights(a,v,m,g,s,'UniformRaw',0)
        for batch in range(2):
            for anchor in range(4):
                if not m[batch,anchor]:continue
                expected=torch.full_like(b[batch,anchor,:,v[batch].nonzero().flatten()],1/(int(v[batch].sum())+1))
                torch.testing.assert_close(b[batch,anchor,:,v[batch].nonzero().flatten()],expected,rtol=0,atol=0)
                torch.testing.assert_close(b[batch,anchor,:,-1],torch.full_like(b[batch,anchor,:,-1],1/(int(v[batch].sum())+1)),rtol=0,atol=0)

    def test_global_mass_and_per_hand_preservation(self):
        a,v,m,g,s=self.fixture()
        for mode in MODES[2:]:
            b=replace_weights(a,v,m,g,s,mode,1)
            torch.testing.assert_close(b[...,-1],a[...,-1],rtol=0,atol=0)
            torch.testing.assert_close(b[...,:60].sum(-1),a[...,:60].sum(-1),rtol=0,atol=WEIGHT_ATOL)
            if mode in MASS_MATCHED:
                for h in range(2):torch.testing.assert_close(b[...,h*30:(h+1)*30].sum(-1),a[...,h*30:(h+1)*30].sum(-1),rtol=0,atol=WEIGHT_ATOL)

    def test_global_uniform_does_not_claim_to_preserve_hand_masses(self):
        valid=torch.ones(1,60,dtype=torch.bool);mask=torch.ones(1,1,dtype=torch.bool)
        a=torch.zeros(1,1,4,61);a[...,:30]=.9/30;a[...,30:60]=.05/30;a[...,-1]=.05
        geometry=torch.zeros(1,1,60,4);scores=torch.zeros(1,1,4,60)
        global_=replace_weights(a,valid,mask,geometry,scores,'UniformGlobalMass',0)
        hand=replace_weights(a,valid,mask,geometry,scores,'UniformPerHandMass',0)
        self.assertFalse(torch.allclose(global_[...,:30].sum(-1),a[...,:30].sum(-1)))
        torch.testing.assert_close(hand[...,:30].sum(-1),torch.full((1,1,4),.9),rtol=0,atol=WEIGHT_ATOL)

    def test_fixed_rbf_and_learned_geometry_can_prefer_opposite_distances(self):
        v=torch.zeros(1,60,dtype=torch.bool);v[0,[0,1]]=True;m=torch.ones(1,1,dtype=torch.bool)
        a=torch.zeros(1,1,4,61);a[...,0]=.3;a[...,1]=.5;a[...,-1]=.2
        g=torch.zeros(1,1,60,4);g[0,0,1,0]=2;g[0,0,1,3]=4
        learned=(g[...,3][:,:,None,:]*.5).expand(1,1,4,60)
        near=replace_weights(a,v,m,g,learned,'RBFPerHandMass',0)
        far=replace_weights(a,v,m,g,learned,'LearnedGeometryPerHandMass',0)
        self.assertTrue((near[...,0]>near[...,1]).all());self.assertTrue((far[...,1]>far[...,0]).all())
        validate(a,near,v,m,'RBFPerHandMass');validate(a,far,v,m,'LearnedGeometryPerHandMass')

    def test_permute_preserves_multiset_without_reordering_values_or_rng(self):
        a,v,m,g,s=self.fixture();rng=torch.get_rng_state().clone()
        b=replace_weights(a,v,m,g,s,'PermuteWithinHand',0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(b,replace_weights(a,v,m,g,s,'PermuteWithinHand',0)))
        self.assertFalse(torch.equal(a,b))
        for h in range(2):torch.testing.assert_close(a[...,h*30:(h+1)*30].sort(-1).values,b[...,h*30:(h+1)*30].sort(-1).values,rtol=0,atol=0)
        values=torch.arange(60).float();before=(a[...,:60]*values).sum(-1);after=(b[...,:60]*values).sum(-1)
        self.assertFalse(torch.equal(before,after))
        for n in (0,1,2,7,30):
            order=permutation(n,1,3,0,'cpu');self.assertEqual(sorted(order.tolist()),list(range(n)))
            if n>1:self.assertFalse(torch.equal(order,torch.arange(n)))

    def test_no_controllers_selects_only_original_null(self):
        a,v,m,g,s=self.fixture();v.zero_();a.zero_();a[...,-1]=1
        for mode in MODES:
            b=replace_weights(a,v,m,g,s,mode,0);validate(a,b,v,m,mode)
            self.assertEqual(int(b[...,:60].count_nonzero()),0);self.assertTrue(torch.equal(b[...,-1],torch.ones_like(b[...,-1])))

    def test_empty_and_extreme_conditionals_finite(self):
        logits=torch.tensor([[[-1e20,0.,1e20],[float('nan'),float('inf'),-float('inf')]]])
        allowed=torch.tensor([[[True,True,True],[False,False,False]]])
        p=conditional(logits,allowed)
        self.assertTrue(torch.isfinite(p).all());self.assertEqual(float(p[0,0,2]),1.)
        self.assertEqual(int(p[0,1].count_nonzero()),0)


class ModelRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def fixture(self,h=5):
        torch.manual_seed(470);model=G5DirectPredictor(horizon=h)
        with torch.no_grad():
            model.motion[-1].weight.normal_(std=.0002)
            model.global_motion.mapping[-1].weight.normal_(std=.0003)
            model.direct_motion.mapping[-1].weight.normal_(std=.0003)
            for block in model.cross:
                block.q.weight.normal_(std=.02)
                for pair in block.pairwise:pair[-1].weight.normal_(std=.01)
        model.eval().requires_grad_(False);processor=DataProcessor('001-rope')
        batch=prepare_batch(synthetic_batch(55,h,1),processor,model)
        return model,processor,batch

    @torch.inference_mode()
    def test_normal_h15_bit_exact_original_and_score_reconstruction(self):
        model,_,batch=self.fixture(15);before=state_hash(model.state_dict());expected=model(batch['inputs'])
        with intervention(model) as rows:actual,trace=model(batch['inputs'],return_trace=True)
        for key in expected:torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
        for step,row in zip(trace['steps'],rows):
            self.assertEqual(len(row['routing_calls']),2)
            for call in row['routing_calls']:
                self.assertIs(call['alpha_learned'],call['alpha_route'])
                torch.testing.assert_close(call['learned_update'],call['route_update'],rtol=0,atol=0)
            torch.testing.assert_close(step['alpha'],row['routing_calls'][-1]['alpha_learned'],rtol=0,atol=0)
        self.assertEqual(before,state_hash(model.state_dict()))

    @torch.inference_mode()
    def test_all_modes_finite_masks_shared_values_and_unchanged_parameters(self):
        model,_,batch=self.fixture();x=batch['inputs'];digest=state_hash(model.state_dict())
        with intervention(model) as normal_rows:normal=model(x)
        for mode in MODES:
            with intervention(model,mode) as rows:pred,trace=model(x,return_trace=True)
            self.assertTrue(torch.isfinite(pred['xyz']).all())
            self.assertEqual(int(pred['xyz'][~x['gs_mask'][:,None].expand(-1,5,-1)].count_nonzero()),0)
            for k,row in enumerate(rows):
                for l,call in enumerate(row['routing_calls']):
                    torch.testing.assert_close(call['values'],normal_rows[k]['routing_calls'][l]['values'],rtol=0,atol=0)
                    validate(call['alpha_learned'],call['alpha_route'],call['inputs']['tool_valid'],call['inputs']['anchor_valid'],mode)
            for key in pred:
                if key!='xyz':torch.testing.assert_close(pred[key],normal[key],rtol=0,atol=0)
        self.assertEqual(digest,state_hash(model.state_dict()));self.assertTrue(all(p.grad is None for p in model.parameters()))

    @torch.inference_mode()
    def test_motion_reads_learned_not_route_alpha(self):
        model,_,batch=self.fixture();x=batch['inputs']
        with intervention(model,'UniformRaw',only_step=3) as rows:_,trace=model(x,return_trace=True)
        row=rows[3];last=row['routing_calls'][-1];step=trace['steps'][3]
        cp=row['global_call']['kwargs']['cp'];old=row['global_call']['kwargs']['previous_cp']
        adjacent=step['adjacent_valid'];delta=masked(cp-old,adjacent)
        learned=last['alpha_learned'][...,:60];route=last['alpha_route'][...,:60]
        expected=torch.einsum('bahj,bjd->bahd',learned,delta).flatten(-2)
        wrong=torch.einsum('bahj,bjd->bahd',route,delta).flatten(-2)
        torch.testing.assert_close(step['tool_motion'],expected,rtol=0,atol=0)
        torch.testing.assert_close(row['motion_input'][...,128:140],expected,rtol=0,atol=0)
        torch.testing.assert_close(row['motion_input'][...,146:150],(learned*adjacent[:,None,None]).sum(-1),rtol=0,atol=0)
        self.assertFalse(torch.equal(expected,wrong))

    @torch.inference_mode()
    def test_h15_same_history_all_routes_matches_only_step_original_rollout(self):
        model,_,batch=self.fixture(15);x=batch['inputs']
        with intervention(model) as rows:normal,trace=model(x,return_trace=True)
        for k in (0,4,9,14):
            for mode in MODES:
                with self.subTest(k=k,mode=mode):
                    replay=replay_step(model,x,trace,rows,k,mode)
                    with intervention(model,mode,only_step=k):actual=model(x)
                    torch.testing.assert_close(actual['xyz'][:,:k],normal['xyz'][:,:k],rtol=0,atol=0)
                    torch.testing.assert_close(actual['xyz'][:,k],replay['xyz'],rtol=0,atol=0)

    @torch.inference_mode()
    def test_first_layer_learned_alpha_unchanged_second_layer_may_change(self):
        model,_,batch=self.fixture();x=batch['inputs']
        with intervention(model) as a:_,nt=model(x,return_trace=True)
        with intervention(model,'UniformRaw',only_step=3) as b:_,rt=model(x,return_trace=True)
        torch.testing.assert_close(a[3]['routing_calls'][0]['alpha_learned'],b[3]['routing_calls'][0]['alpha_learned'],rtol=0,atol=0)
        self.assertFalse(torch.equal(a[3]['routing_calls'][1]['alpha_learned'],b[3]['routing_calls'][1]['alpha_learned']))

    @torch.inference_mode()
    def test_all_modes_future_causality_and_no_target_input(self):
        model,_,batch=self.fixture();x=batch['inputs'];changed=copy.deepcopy(x)
        changed['controller_points'][:,3:]+=.2;changed['tactile'][:,3:]+=10
        changed['target_xyz']='MUST_NOT_BE_READ'
        for mode in MODES:
            with intervention(model,mode):a=model(x)
            with intervention(model,mode):b=model(changed)
            torch.testing.assert_close(a['xyz'][:,:3],b['xyz'][:,:3],rtol=0,atol=0)

    @torch.inference_mode()
    def test_masked_hand_nan_and_empty_controller_null(self):
        model,_,batch=self.fixture();x={k:v[:1] for k,v in batch['inputs'].items()};changed=copy.deepcopy(x)
        changed['controller_points'].reshape(1,5,60,3)[:,:,:30]=float('nan')
        changed['action_state'][:,:,0]=float('nan');changed['tactile'][:,:,:2]=float('nan')
        for mode in MODES:
            with intervention(model,mode):a=model(x)
            with intervention(model,mode):b=model(changed)
            torch.testing.assert_close(a['xyz'],b['xyz'],rtol=0,atol=0)
        empty=copy.deepcopy(x);empty['controller_mask'].zero_();empty['controller_points'].fill_(float('nan'))
        for mode in MODES:
            with intervention(model,mode) as rows:pred=model(empty)
            self.assertTrue(torch.isfinite(pred['xyz']).all())
            for row in rows:
                for call in row['routing_calls']:self.assertTrue(torch.equal(call['alpha_route'][...,-1],torch.ones_like(call['alpha_route'][...,-1])))

    def test_cleanup_and_unknown_mode_and_non_normal_replay_rejected(self):
        model,_,batch=self.fixture()
        with self.assertRaises(RuntimeError):
            with intervention(model,'UniformRaw'):raise RuntimeError('interrupted')
        for block in model.cross:
            self.assertFalse(block.projection._forward_pre_hooks);self.assertFalse(block.v._forward_hooks)
        self.assertFalse(model.motion._forward_hooks)
        with self.assertRaises(ValueError):
            with intervention(model,'INVALID'):pass
        with torch.inference_mode():
            with intervention(model,'UniformRaw') as rows:_,trace=model(batch['inputs'],return_trace=True)
            with self.assertRaises(ValueError):replay_step(model,batch['inputs'],trace,rows,3,'Normal')

    @torch.inference_mode()
    def test_routing_statistics_and_raw_arrays_on_actual_frozen_trace(self):
        from scripts.experiments.v4_g5_attention_routing_audit.diagnostics import route_rows,capture_routing_arrays
        from scripts.experiments.v4_g5_attention_routing_audit.run import validate_readouts
        model,processor,batch=self.fixture(15);x={k:v[:1] for k,v in batch['inputs'].items()}
        with intervention(model) as original:_,normal_trace=model(x,return_trace=True)
        validate_readouts(original,normal_trace)
        for mode in MODES:
            with intervention(model,mode) as captures:_,trace=model(x,return_trace=True)
            validate_readouts(captures,trace)
            rows,checks=route_rows(x,trace,captures,normal_trace,original,processor.state_dict()['scale'],
                dict(split='SYNTHETIC_ONLY',episode=2,start=0),mode,'full_AR')
            self.assertEqual(len(checks),30)
            self.assertEqual({r['horizon'] for r in rows if r['region']=='all'},set(range(1,16)))
            self.assertEqual({r['horizon'] for r in rows if r['region']!='all'},{1,5,10,15})
            for row in rows:
                self.assertTrue(0<=row['total_variation']<=1.000001)
                self.assertTrue(-1e-7<=row['js_divergence_nats']<=.693148)
                self.assertEqual(row['left_mass_route'],0.)
                if mode=='Normal':
                    self.assertEqual(row['value_aggregate_change_feature_rms'],0.)
                    self.assertEqual(row['total_variation'],0.)
                    self.assertEqual(row['local_branch_change_gs_rms_mm_joint_heads'],0.)
            arrays={};capture_routing_arrays(arrays,mode,captures)
            self.assertEqual(len(arrays),4*2*4)
            json.dumps(rows,allow_nan=False)

    @torch.inference_mode()
    def test_complete_single_history_loss_and_branch_statistics_mock_renderer(self):
        from scripts.training.rendering import GaussianRenderer
        from scripts.training.v4_runtime import v4_objective
        from scripts.experiments.v4_stage1.fixtures import analytic_rasterizer
        from scripts.experiments.v4_g5_attention_routing_audit.run import single_effects
        from scripts.experiments.v4_g5_motion_attribution.run import horizon_rows
        model,processor,batch=self.fixture(15)
        # The formal audit is strictly Batch1; synthetic fixture normally has B2.
        def first(value):
            if isinstance(value,dict):return {k:first(v) for k,v in value.items()}
            if isinstance(value,torch.Tensor):return value[:1]
            if isinstance(value,list):return [first(v) for v in value]
            return value
        batch=first(batch);x=batch['inputs'];renderer=GaussianRenderer(rasterizer=analytic_rasterizer)
        with intervention(model) as captures:pred,trace=model(x,return_trace=True)
        target=renderer.render_targets(batch,processor);loss,metrics=v4_objective(pred,batch,processor,renderer)
        scope=dict(split='SYNTHETIC_ONLY',episode=2,start=0)
        normal_rows=horizon_rows(pred,batch,metrics,pred,processor.state_dict()['scale'],'Normal',scope)
        rows,regions,checks,routing,routing_checks=single_effects(model,batch,pred,trace,captures,normal_rows,
            renderer,target,processor,scope,True)
        self.assertEqual(len(rows),6*4);self.assertEqual(len(checks),6*4)
        self.assertEqual(len(routing_checks),6*4*2)
        self.assertEqual({r['horizon'] for r in rows},{1,5,10,15})
        self.assertEqual({r['region'] for r in regions},{'all','near','middle','far'})
        json.dumps([rows,regions,checks,routing,routing_checks],allow_nan=False)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @torch.inference_mode()
    def test_full_ar_branch_and_spatial_statistics_keep_anchor_gs_distinction(self):
        from scripts.experiments.v4_g5_attention_routing_audit.diagnostics import branch_rows
        from scripts.experiments.v4_g5_attention_routing_audit.run import spatial_rows
        model,processor,batch=self.fixture()
        def first(value):
            if isinstance(value,dict):return {k:first(v) for k,v in value.items()}
            if isinstance(value,torch.Tensor):return value[:1]
            if isinstance(value,list):return [first(v) for v in value]
            return value
        batch=first(batch);x=batch['inputs'];scale=processor.state_dict()['scale']
        with intervention(model) as normal:pred,trace=model(x,return_trace=True)
        rows=branch_rows(x,trace,trace,scale,dict(split='SYNTHETIC_ONLY',episode=2,start=0),'Normal')
        self.assertEqual({r['space'] for r in rows},{'anchor_control','gs_actual'})
        self.assertEqual({r['branch'] for r in rows},{'base','global_','local','final'})
        self.assertTrue(all(r['change_rms_mm']==0 for r in rows))
        with intervention(model,'UniformRaw'):other,changed=model(x,return_trace=True)
        rows=branch_rows(x,changed,trace,scale,dict(split='SYNTHETIC_ONLY',episode=2,start=0),'UniformRaw')
        self.assertTrue(any(r['change_rms_mm']>0 for r in rows if r['branch']=='base'))
        spatial=spatial_rows(other,batch,scale)
        self.assertEqual({r['region'] for r in spatial},{'all','near','middle','far'})
        self.assertTrue(all(r['pred_to_gt_mm']>=0 for r in spatial))
        json.dumps([rows,spatial],allow_nan=False)

    @torch.inference_mode()
    def test_empty_controller_diagnostics_mark_topk_and_conditional_distance_undefined(self):
        from scripts.experiments.v4_g5_attention_routing_audit.diagnostics import route_rows
        model,processor,batch=self.fixture();x={k:v[:1] for k,v in batch['inputs'].items()}
        with intervention(model) as normal:_,nt=model(x,return_trace=True)
        empty=copy.deepcopy(x);empty['controller_mask'].zero_()
        with intervention(model,'UniformRaw') as captures:_,trace=model(empty,return_trace=True)
        # Original valid initial geometry defines regions; score validity comes
        # from the executed call, so no Controller mass must remain undefined.
        rows,_=route_rows(x,trace,captures,nt,normal,processor.state_dict()['scale'],
            dict(split='SYNTHETIC_ONLY',episode=2,start=0),'UniformRaw','full_AR')
        self.assertTrue(all(r['top_k_agreement'] is None for r in rows))
        self.assertTrue(all(r['conditional_distance_route_mm'] is None for r in rows))
        self.assertTrue(all(r['null_route']==1 for r in rows))


class DiagnosticIntegrityTests(unittest.TestCase):
    def test_gpu_render_layout_rejects_wrong_horizon_camera_count_or_resolution(self):
        from scripts.experiments.v4_g5_attention_routing_audit.run import check_render_layout
        zero=torch.zeros(1)
        target=dict(rgb=zero.expand(1,15,3,720,1280,3),alpha=zero.expand(1,15,3,720,1280))
        check_render_layout(target)
        for rgb in (zero.expand(1,15,2,720,1280,3),zero.expand(1,5,3,720,1280,3),zero.expand(1,15,3,360,640,3)):
            with self.assertRaises(ValueError):check_render_layout(dict(target,rgb=rgb))

    def test_entropy_and_js_reference(self):
        from scripts.experiments.v4_g5_attention_routing_audit.diagnostics import entropy,js_divergence
        p=torch.tensor([[1.,0.],[.5,.5]])
        torch.testing.assert_close(entropy(p),torch.tensor([0.,torch.log(torch.tensor(2.))]))
        torch.testing.assert_close(js_divergence(p,p),torch.zeros(2),rtol=0,atol=0)
        torch.testing.assert_close(js_divergence(p[:1],p[:1].flip(-1)),torch.tensor([torch.log(torch.tensor(2.))]))

    def test_no_new_optimizer_backward_or_step_calls(self):
        import ast
        root=Path(__file__).resolve().parents[2]/'experiments'/'v4_g5_attention_routing_audit'
        for path in root.glob('*.py'):
            node=ast.parse(path.read_text(encoding='utf8'))
            for call in (n for n in ast.walk(node) if isinstance(n,ast.Call)):
                if isinstance(call.func,ast.Attribute):
                    self.assertNotIn(call.func.attr,('backward','step','AdamW','Adam','SGD'),str(path))

    def test_report_refuses_incomplete_or_existing_and_preserves_mode_definitions(self):
        from scripts.experiments.v4_g5_attention_routing_audit.report import build
        episode=lambda count:dict(samples=count,geometry_mm=2.,supervised_geometry=.02,rgb=.1,alpha=.1)
        data={name:[] for name in ('windows','horizons','horizons_val_summary','same_history_val_summary',
            'same_history_regions_val_summary','branch_changes_val_summary','structure_val_summary',
            'routing_metrics_val_summary','routing_checks','intervention_checks')}
        data['val_summary']={mode:dict(geometry_mm=2.,rgb=.1,alpha=.1,objective=.13,
            episodes={'3':episode(55),'6':episode(45)}) for mode in (*MODES,'Copy')}
        provenance=dict(training_source_sha='SYNTHETIC_ONLY',diagnostic_source_sha='TEST',checkpoint='NONE',checkpoint_sha256='0'*64,
            job_id='NO_GPU',gpu='NONE',torch='TEST',cuda='NONE',gsplat='MOCK',elapsed_seconds=0.,peak_allocated_bytes=0)
        with tempfile.TemporaryDirectory(prefix='touch4d-routing-audit-') as tmp:
            out=Path(tmp);(out/'results').mkdir()
            with self.assertRaises(RuntimeError):build(out)
            (out/'completed.json').write_text('{}',encoding='utf8')
            for name,value in data.items():(out/'results'/(name+'.json')).write_text(json.dumps(value),encoding='utf8')
            (out/'provenance.json').write_text(json.dumps(provenance),encoding='utf8')
            with patch('scripts.experiments.v4_g5_attention_routing_audit.report.plots',return_value=[]):
                build(out)
                report=(out/'final_report.md').read_text(encoding='utf8')
                for word in ('SYNTHETIC_ONLY','UniformRaw','PermuteWithinHand','alpha_learned','FFN'):
                    self.assertIn(word,report)
                with self.assertRaises(FileExistsError):build(out)

    def test_three_camera_sheets_use_all_conditions_and_do_not_edit_originals(self):
        from PIL import Image
        from scripts.experiments.v4_g5_attention_routing_audit.report import plots
        from scripts.experiments.v4_g5_signed_local.diagnostics import sha256
        with tempfile.TemporaryDirectory(prefix='touch4d-routing-pictures-') as tmp:
            out=Path(tmp);(out/'figures').mkdir();hashes={}
            for key in ('val_ep3_s135','val_ep3_s245','val_ep6_s110','train_ep7_s105'):
                directory=out/'images'/key;directory.mkdir(parents=True)
                for camera in range(3):
                    for channel in ('rgb','alpha'):
                        for i,mode in enumerate(('GT',*MODES,'Copy')):
                            path=directory/f'h15_v{camera}_{channel}_{mode}.png'
                            Image.new('RGB',(16,9),(i*25,camera*80,120)).save(path);hashes[path]=sha256(path)
            import numpy as np
            (out/'arrays').mkdir()
            xyz=np.array([[0.,0.,0.],[.01,.02,.03],[-.01,-.02,-.03]])
            arrays={mode+'_world':np.repeat(xyz[None],15,axis=0) for mode in (*MODES,'Copy')}
            arrays.update({f'gt_h{h:02}_world':xyz for h in range(1,16)})
            np.savez_compressed(out/'arrays'/'val_ep3_s245.npz',**arrays)
            data={name:[] for name in ('horizons','horizons_val_summary','same_history_val_summary',
                'routing_metrics_val_summary','branch_changes_val_summary','structure_val_summary')}
            images=plots(out,data)
            sheets=[p for p in images if p.endswith('_comparison.png')]
            self.assertEqual(len(sheets),24);self.assertEqual(len(images),38)
            self.assertTrue(all(sha256(p)==h for p,h in hashes.items()))
            with Image.open(out/sheets[0]) as image:
                self.assertEqual(image.size,(1920,1212))
                self.assertEqual(image.getpixel((100,100)),(0,0,120))  # GT panel


if __name__=='__main__':unittest.main()
