"""Frozen information-path contracts. No optimizer, backward, or real data."""
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
from scripts.experiments.v4_g5_condition_attribution.interventions import (
    MODES,CONTENT_MODES,DIRECT_MODES,KV_MODES,CHANNELS,features13,content_hooks,
    intervention,replay_step)
from scripts.experiments.v4_g5_condition_attribution.activity import activity_rows,freeze_groups
from scripts.experiments.v4_g5_condition_attribution.diagnostics import assert_preserved_inputs,cross_rows


class ConditionAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def fixture(self,horizon=4):
        torch.manual_seed(618)
        model=G5DirectPredictor(horizon=horizon)
        with torch.no_grad():
            model.motion[-1].weight.normal_(std=.0002)
            model.global_motion.mapping[-1].weight.normal_(std=.0003)
            model.direct_motion.mapping[-1].weight.normal_(std=.0003)
            for block in model.cross:block.q.weight.normal_(std=.03)
        model.eval().requires_grad_(False)
        x=prepare_batch(synthetic_batch(55,horizon,1),DataProcessor('001-rope'),model)['inputs']
        return model,x

    def test_exact_preregistered_modes(self):
        self.assertEqual(len(MODES),9);self.assertEqual(len(KV_MODES),4);self.assertEqual(len(CHANNELS),3)

    def test_pressure_only_two_channels_storage_safe(self):
        a=torch.randn(2,60,13);copy_=a.clone();b=features13(a,'T3PressureZero')
        torch.testing.assert_close(a,copy_,rtol=0,atol=0)
        self.assertEqual(int(b[...,7:9].count_nonzero()),0)
        for sl in (slice(0,7),slice(9,13)):torch.testing.assert_close(b[...,sl],a[...,sl],rtol=0,atol=0)

    def test_all6_off_preserves7(self):
        a=torch.randn(1,60,13);b=features13(a,'T3All6Off')
        torch.testing.assert_close(a[...,:7],b[...,:7],rtol=0,atol=0);self.assertEqual(int(b[...,7:].count_nonzero()),0)

    def test_a0_off_preserves_t3(self):
        a=torch.randn(1,60,13);b=features13(a,'CrossA0GeometryOff')
        torch.testing.assert_close(a[...,7:],b[...,7:],rtol=0,atol=0);self.assertEqual(int(b[...,:7].count_nonzero()),0)

    def test_all13_off_is_not_token_off(self):
        m,x=self.fixture()
        with content_hooks(m.conditions,'CrossA0T3Off') as rows:tools,v,cp,tr=m.conditions(x,2)
        self.assertEqual(int(rows[0]['effective_features13'].count_nonzero()),0)
        self.assertGreater(float(tools[v].abs().sum()),0)

    @torch.inference_mode()
    def test_normal_h15_all_attributes_bit_exact_and_state_unchanged(self):
        m,x=self.fixture(15);before=state_hash(m.state_dict());expected=m(x)
        with intervention(m) as rows:actual,tr=m(x,return_trace=True)
        for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        self.assertEqual(len(rows),15);self.assertEqual(before,state_hash(m.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.parameters()))

    @torch.inference_mode()
    def test_full_modes_finite_padding_and_static_attributes(self):
        m,x=self.fixture();reference=m(x)
        for mode in MODES:
            with intervention(m,mode):p,t=m(x,return_trace=True)
            for key,value in p.items():
                self.assertTrue(torch.isfinite(value).all())
                if key!='xyz':torch.testing.assert_close(value,reference[key],rtol=0,atol=0)
            mask=~x['gs_mask'][:,None].expand(-1,4,-1)
            self.assertEqual(int(p['xyz'][mask].count_nonzero()),0)

    @torch.inference_mode()
    def test_all_interventions_leave_raw_inputs_unchanged(self):
        m,x=self.fixture();original=copy.deepcopy(x)
        for mode in MODES:
            with intervention(m,mode):m(x)
        for key in x:torch.testing.assert_close(x[key],original[key],rtol=0,atol=0)

    @torch.inference_mode()
    def test_content_modes_do_not_close_hand_or_geometry_bypass(self):
        m,x=self.fixture()
        with intervention(m) as normal:p,t=m(x,return_trace=True)
        for mode in CONTENT_MODES:
            with intervention(m,mode,only_step=2) as rows:q,tr=m(x,return_trace=True)
            assert_preserved_inputs(rows[2],normal[2],mode)
            for key in ('hand_basis12','anchor_hand_basis18'):
                torch.testing.assert_close(tr['steps'][2][key],t['steps'][2][key],rtol=0,atol=0)

    @torch.inference_mode()
    def test_cross_hand_off_only_projection_addition(self):
        m,x=self.fixture()
        with content_hooks(m.conditions,'Normal') as a:normal=m.conditions(x,2)
        with content_hooks(m.conditions,'CrossHandFusionOff') as b:changed=m.conditions(x,2)
        torch.testing.assert_close(a[0]['hand_tokens'],b[0]['hand_tokens'],rtol=0,atol=0)
        torch.testing.assert_close(a[0]['effective_features13'],b[0]['effective_features13'],rtol=0,atol=0)
        self.assertEqual(int(b[0]['effective_hand_projection'].count_nonzero()),0)
        self.assertFalse(torch.equal(normal[0],changed[0]))

    @torch.inference_mode()
    def test_direct_hand_off_preserves_cross_fusion(self):
        m,x=self.fixture()
        with intervention(m) as n:normal,nt=m(x,return_trace=True)
        with intervention(m,'DirectHandTokenOff',only_step=2) as r:p,t=m(x,return_trace=True)
        assert_preserved_inputs(r[2],n[2],'DirectHandTokenOff')
        d=r[2]['direct_intervention']
        self.assertEqual(int(d['global_kwargs_effective']['hand_tokens'].count_nonzero()),0)
        self.assertEqual(int(d['local_kwargs_effective']['hand_tokens'].count_nonzero()),0)
        torch.testing.assert_close(d['local_kwargs_effective']['global_hand_contribution'],t['steps'][2]['hand_contribution'],rtol=0,atol=0)
        torch.testing.assert_close(t['steps'][2]['alpha'],nt['steps'][2]['alpha'],rtol=0,atol=0)

    @torch.inference_mode()
    def test_local6_off_mapping_context_and_product_consistent(self):
        m,x=self.fixture()
        with intervention(m,'LocalController6Off') as rows:p,t=m(x,return_trace=True)
        for raw,s in zip(rows,t['steps']):
            d=raw['direct_intervention'];basis=d['local_basis_effective']
            self.assertEqual(int(basis[...,12:].count_nonzero()),0)
            torch.testing.assert_close(d['local_context_effective'][...,260:278],basis,rtol=0,atol=0)
            matrix=d['local_matrix'].reshape(*basis.shape[:-1],3,18)
            expected=masked(torch.einsum('bahij,bahj->bahi',matrix,basis),t['field'].anchor_mask[:,:,None]&s['hand_valid'][:,None])
            torch.testing.assert_close(expected,s['direct_seed_per_hand'],rtol=0,atol=0)

    @torch.inference_mode()
    def test_signed12_off_recomputes_g4_and_g5_context(self):
        m,x=self.fixture()
        with intervention(m,'SignedHand12Off') as rows:p,t=m(x,return_trace=True)
        for raw,s in zip(rows,t['steps']):
            d=raw['direct_intervention']
            for value in (s['hand_basis12'],s['global_delta'],s['hand_contribution'],s['anchor_hand_basis18'][...,:12],d['global_context_effective'][...,-12:]):
                self.assertEqual(int(value.count_nonzero()),0)
            torch.testing.assert_close(d['local_context_effective'][...,260:278],s['anchor_hand_basis18'],rtol=0,atol=0)
            self.assertEqual(int(d['local_context_effective'][...,-3:].count_nonzero()),0)
        self.assertGreater(float(t['steps'][2]['anchor_hand_basis18'][...,12:].abs().sum()),0)

    @torch.inference_mode()
    def test_h1_direct_ablation_equal_because_signed_motion_zero(self):
        m,x=self.fixture();reference=m(x)
        for mode in DIRECT_MODES:
            with intervention(m,mode):p=m(x)
            torch.testing.assert_close(p['xyz'][:,0],reference['xyz'][:,0],rtol=0,atol=0)

    @torch.inference_mode()
    def test_normal_same_history_replay_bit_exact(self):
        m,x=self.fixture()
        with intervention(m) as rows:p,t=m(x,return_trace=True)
        for k in range(4):
            r=replay_step(m,x,t,rows,k)
            torch.testing.assert_close(r['xyz'],p['xyz'][:,k],rtol=0,atol=0)

    @torch.inference_mode()
    def test_every_end_to_end_replay_matches_only_step_ar(self):
        m,x=self.fixture()
        with intervention(m) as rows:p,t=m(x,return_trace=True)
        for mode in MODES[1:]:
            r=replay_step(m,x,t,rows,2,mode)
            with intervention(m,mode,only_step=2):actual=m(x)
            torch.testing.assert_close(actual['xyz'][:,:2],p['xyz'][:,:2],rtol=0,atol=0)
            torch.testing.assert_close(actual['xyz'][:,2],r['xyz'],rtol=0,atol=0)

    @torch.inference_mode()
    def test_fixed_routing_uses_two_normal_layers_and_normal_key(self):
        m,x=self.fixture()
        with intervention(m) as rows:p,t=m(x,return_trace=True)
        for mode in KV_MODES:
            r=replay_step(m,x,t,rows,2,mode,'FixedRoutingChangedValue')
            for call,normal in zip(r['condition_cross_calls'],rows[2]['condition_cross_calls']):
                torch.testing.assert_close(call['alpha_used'],normal['alpha_used'],rtol=0,atol=0)
                torch.testing.assert_close(call['k'],normal['k'],rtol=0,atol=0)
            torch.testing.assert_close(r['alpha'],t['steps'][2]['alpha'],rtol=0,atol=0)
            torch.testing.assert_close(r['tool_motion'],t['steps'][2]['tool_motion'],rtol=0,atol=0)

    @torch.inference_mode()
    def test_changed_key_preserves_normal_value(self):
        m,x=self.fixture()
        with intervention(m) as rows:p,t=m(x,return_trace=True)
        for mode in KV_MODES:
            r=replay_step(m,x,t,rows,2,mode,'ChangedKeyNormalValue')
            for call,normal in zip(r['condition_cross_calls'],rows[2]['condition_cross_calls']):
                torch.testing.assert_close(call['v'],normal['v'],rtol=0,atol=0)
                torch.testing.assert_close(call['geometry'],normal['geometry'],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(a['k'],b['k']) for a,b in zip(r['condition_cross_calls'],rows[2]['condition_cross_calls'])))

    @torch.inference_mode()
    def test_projection_uses_each_own_linear_layer(self):
        m,x=self.fixture()
        with intervention(m) as rows:p,t=m(x,return_trace=True)
        with content_hooks(m.conditions,'T3All6Off'):tools,valid,_,_=m.conditions(x,2)
        for channel,name in [('FixedRoutingChangedValue','v'),('ChangedKeyNormalValue','k')]:
            r=replay_step(m,x,t,rows,2,'T3All6Off',channel)
            for block,call in zip(m.cross,r['condition_cross_calls']):
                expected=torch.nn.functional.linear(block.memory_norm(masked(tools,valid)),getattr(block,name).weight,getattr(block,name).bias)
                torch.testing.assert_close(call[name],expected,rtol=0,atol=0)

    @torch.inference_mode()
    def test_all_invalid_null_only(self):
        m,x=self.fixture();x['controller_mask'].zero_();x['action_valid'].zero_()
        for mode in MODES:
            with intervention(m,mode):p,t=m(x,return_trace=True)
            self.assertTrue(torch.isfinite(p['xyz']).all())
            for s in t['steps']:
                self.assertEqual(int(s['alpha'][...,:60].count_nonzero()),0)
                self.assertTrue(torch.equal(s['alpha'][...,-1],torch.ones_like(s['alpha'][...,-1])))

    @torch.inference_mode()
    def test_masked_hand_nan_cannot_leak(self):
        m,x=self.fixture();other=copy.deepcopy(x)
        other['action_state'][0,:,0]=float('nan');other['controller_points'][0,:,0]=float('nan')
        other['tactile'][0,:,:2]=float('nan')
        for mode in MODES:
            with intervention(m,mode):a=m(x)
            with intervention(m,mode):b=m(other)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    @torch.inference_mode()
    def test_future_conditions_do_not_affect_prefix(self):
        m,x=self.fixture();other=copy.deepcopy(x)
        other['controller_points'][:,3:]+=.1;other['action_state'][:,3:,...,:3]+=.1;other['tactile'][:,3:]+=10
        for mode in MODES:
            with intervention(m,mode):a=m(x)
            with intervention(m,mode):b=m(other)
            torch.testing.assert_close(a['xyz'][:,:3],b['xyz'][:,:3],rtol=0,atol=0)

    def test_hooks_removed_on_exception(self):
        m,x=self.fixture()
        with self.assertRaises(RuntimeError):
            with intervention(m,'T3All6Off'):raise RuntimeError('forced')
        self.assertFalse(any(z._forward_hooks or z._forward_pre_hooks for z in m.modules()))

    def test_mid_forward_failure_keeps_original_error_and_removes_hooks(self):
        m,x=self.fixture()
        with patch.object(m.cross[0].ff,'forward',side_effect=RuntimeError('ORIGINAL_SENTINEL')):
            with self.assertRaisesRegex(RuntimeError,'ORIGINAL_SENTINEL'):
                with intervention(m,'T3All6Off'):m(x)
        self.assertFalse(any(z._forward_hooks or z._forward_pre_hooks for z in m.modules()))

    @torch.inference_mode()
    def test_valid_zero_and_missing_activity_distinct(self):
        m,x=self.fixture();x={k:v[:1] for k,v in x.items()};x['tactile'].zero_()
        rows=activity_rows(m.conditions,x,dict(split='val',episode=3,start=10))
        right=[r for r in rows if r['operation_hand']]
        self.assertTrue(all(r['valid_zero_taxels']>0 and r['mean_valid_points']==15 and r['nonzero_fraction_valid']==0 for r in right))
        x['valid_mask'].zero_();missing=activity_rows(m.conditions,x,dict(split='val',episode=3,start=10))
        self.assertTrue(all(r['effective_valid_taxels']==0 and r['nonzero_fraction_valid'] is None for r in missing))

    @torch.inference_mode()
    def test_tactile_age_roi_and_finite_semantics(self):
        m,x=self.fixture();x={k:v[:1] for k,v in x.items()}
        x['tactile_age_us'][:,1,2]=80000;x['tactile'][:,2,3]=float('nan')
        rows=activity_rows(m.conditions,x,dict(split='val',episode=3,start=10))
        self.assertTrue(all(r['effective_valid_taxels']<=372 for r in rows))
        self.assertEqual([r for r in rows if r['horizon']==2 and r['hand']=='right' and r['finger']==0][0]['effective_valid_taxels'],0)
        self.assertEqual([r for r in rows if r['horizon']==3 and r['hand']=='right' and r['finger']==1][0]['effective_valid_taxels'],0)

    def test_input_only_group_rule_ties_and_no_observation(self):
        rows=[]
        for ep in (3,6):
            for i in range(8):rows.append(dict(split='val',episode=ep,start=i,operation_hand=True,effective_valid_taxels=100,positive_valid_taxels=i*10 if ep==3 else 0))
        rows.append(dict(split='val',episode=3,start=9,operation_hand=True,effective_valid_taxels=0,positive_valid_taxels=0))
        groups=freeze_groups(rows);by={(r['episode'],r['start']):r for r in groups['windows']}
        self.assertEqual(by[3,0]['activity_group'],'LOW');self.assertEqual(by[3,7]['activity_group'],'HIGH')
        self.assertEqual(by[3,9]['activity_group'],'NO_OBSERVATION')
        self.assertTrue(all(by[6,i]['activity_group']=='INDETERMINATE' for i in range(8)))

    @torch.inference_mode()
    def test_cross_diagnostics_valid_counts_and_zero_normal_changes(self):
        m,x=self.fixture();x={k:v[:1] for k,v in x.items()}
        with intervention(m) as rows:p,t=m(x,return_trace=True)
        c=rows[2]['condition_cross_calls'];stats=cross_rows(x,t['field'],c,c,dict(split='val',episode=3,start=10),'Normal','full_AR',3)
        self.assertTrue(stats)
        self.assertTrue(all(r['key_change_feature_rms']==0 and r['value_change_feature_rms']==0 and r['attention_total_variation']==0 for r in stats))

    def test_invalid_mode_step_and_kv_combinations_rejected(self):
        m,x=self.fixture()
        for mode in ('NoAction','typo'):
            with self.assertRaises(ValueError):
                with intervention(m,mode):pass
        with self.assertRaises(ValueError):
            with intervention(m,only_step=-1):pass
        with self.assertRaises(ValueError):replay_step(m,x,{},[],0,'SignedHand12Off','ChangedKeyNormalValue')

    @torch.inference_mode()
    def test_actual_single_step_runner_original_mock_loss_and_reductions(self):
        from scripts.experiments.v4_stage1.fixtures import analytic_rasterizer
        from scripts.training.rendering import GaussianRenderer
        from scripts.training.v4_runtime import v4_objective
        from scripts.experiments.v4_g5_condition_attribution.run import single_effects,CachedRenderer,horizon_rows
        m,_=self.fixture(15);processor=DataProcessor('001-rope');raw=synthetic_batch(55,15,1)
        def take(value):
            if isinstance(value,torch.Tensor):return value[:1]
            if isinstance(value,dict):return {k:take(v) for k,v in value.items()}
            if isinstance(value,list):return value[:1] if all(isinstance(v,str) for v in value) else [take(v) for v in value]
            return value
        single=take(raw)
        # Camera IDs are [batch][camera], unlike the [horizon][batch,N] targets.
        single['meta']['depth_camera_ids']=copy.deepcopy(raw['meta']['depth_camera_ids'][:1])
        batch=prepare_batch(single,processor,m);x=batch['inputs']
        with intervention(m) as captures:p,t=m(x,return_trace=True)
        renderer=GaussianRenderer(analytic_rasterizer);target=renderer.render_targets(batch,processor)
        loss,metrics=v4_objective(p,batch,processor,CachedRenderer(renderer,target))
        scope=dict(split='val',episode=3,start=73,activity_group='SYNTHETIC_ONLY')
        hrows=horizon_rows(p,batch,metrics,p,processor.state_dict()['scale'],'Normal',scope)
        with patch('scripts.experiments.v4_g5_condition_attribution.run.SELECTED',(1,15)):
            records=single_effects(m,batch,p,t,captures,hrows,renderer,target,processor,scope,False)
        self.assertEqual(len(records['kv_attribution']),32)
        self.assertEqual(len(records['kv_regions']),128)
        self.assertTrue(all(torch.isfinite(torch.tensor(r['joint'])) for r in records['kv_attribution']))
        self.assertTrue(all(r['channel'] in CHANNELS for r in records['kv_attribution']))

    def test_experiment_has_no_training_or_parameter_assignment(self):
        import ast
        root=Path(__file__).resolve().parents[2]/'experiments'/'v4_g5_condition_attribution'
        for path in root.glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf8'))
            for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call)):
                if isinstance(call.func,ast.Attribute):self.assertNotIn(call.func.attr,('backward','step','AdamW','Adam','SGD'),str(path))

    def test_allocated_gpu_foreign_process_guard_never_interferes(self):
        import os
        from scripts.experiments.v4_g5_condition_attribution.run import ensure_assigned_gpu_idle
        with patch.dict(os.environ,{'SLURM_JOB_GPUS':'','SLURM_STEP_GPUS':'4','CUDA_VISIBLE_DEVICES':'0'}):
            with patch('scripts.experiments.v4_g5_condition_attribution.run.subprocess.check_output',side_effect=['GPU-FAKE\n',str(os.getpid())]) as query:
                result=ensure_assigned_gpu_idle()
                self.assertEqual(result['foreign_compute_pids'],[])
                self.assertEqual(result['slurm_step_gpus'],'4')
                self.assertIn('GPU-FAKE',query.call_args[0][0])
            with patch('scripts.experiments.v4_g5_condition_attribution.run.subprocess.check_output',side_effect=['GPU-FAKE\n','99999999']):
                with self.assertRaisesRegex(RuntimeError,'preserve it'):ensure_assigned_gpu_idle()
            with patch('scripts.experiments.v4_g5_condition_attribution.run.subprocess.check_output',return_value='GPU-A\nGPU-B\n'):
                with self.assertRaisesRegex(RuntimeError,'unambiguously'):ensure_assigned_gpu_idle()


class ConditionReportTests(unittest.TestCase):
    def test_paired_counts_exclude_train_and_keep_counterexamples(self):
        from scripts.experiments.v4_g5_condition_attribution.report import analyze
        rows=[]
        for split,ep,start in [('val',3,1),('val',6,2),('train',7,105)]:
            rows.append(dict(split=split,episode=ep,start=start,mode='Normal',joint=.1,cd_mm=3.))
            for mode in MODES[1:]:rows.append(dict(split=split,episode=ep,start=start,mode=mode,joint=.09 if ep==3 else .12,cd_mm=2. if ep==3 else 4.))
        counts,examples=analyze(rows)
        self.assertTrue(all(r['joint_improved']==1 and r['joint_degraded']==1 for r in counts if r['episode']=='all'))
        self.assertTrue(all(r['split']=='val' for r in examples))

    def test_report_refuses_incomplete_and_existing_and_answers_ten_questions(self):
        from scripts.experiments.v4_g5_condition_attribution.report import build
        episode=dict(geometry_mm=2.,supervised_geometry=.02,rgb=.1,alpha=.1)
        summary={m:dict(geometry_mm=2.,rgb=.1,alpha=.1,objective=.13,episodes={'3':dict(episode,samples=55),'6':dict(episode,samples=45)}) for m in (*MODES,'Copy')}
        windows=[dict(split='val',episode=ep,start=i,mode=m,joint=.13,cd_mm=2.) for ep,n in ((3,55),(6,45)) for i in range(n) for m in (*MODES,'Copy')]
        data=dict(val_summary=summary,windows=windows,horizons_val_summary=[],kv_attribution_val_summary=[],branch_changes_val_summary=[],
                  condition_activity=[],condition_activity_groups=dict(thresholds={}),activity_group_summary=[])
        provenance=dict(diagnostic_source_sha='TEST',training_source_sha='SYNTHETIC_ONLY',job_id='NONE',step_id='NONE',checkpoint='NONE',checkpoint_sha256='0'*64,model_hash_before='TEST',model_hash_after='TEST')
        with tempfile.TemporaryDirectory(prefix='touch4d-condition-report-') as tmp:
            out=Path(tmp);(out/'results').mkdir()
            with self.assertRaises(RuntimeError):build(out)
            (out/'completed.json').write_text('{}',encoding='utf8')
            for name,value in data.items():(out/'results'/f'{name}.json').write_text(json.dumps(value),encoding='utf8')
            (out/'provenance.json').write_text(json.dumps(provenance),encoding='utf8')
            with patch('scripts.experiments.v4_g5_condition_attribution.report.plots',return_value=[]):
                build(out)
                report=(out/'final_report.md').read_text(encoding='utf8')
                for q in range(1,11):self.assertIn(f'Q{q} ',report)
                with self.assertRaises(FileExistsError):build(out)

    def test_plots_three_camera_sheets_and_fixed_geometry_preserve_sources(self):
        import numpy as np
        from PIL import Image
        from scripts.experiments.v4_g5_condition_attribution.report import plots
        from scripts.experiments.v4_g5_signed_local.diagnostics import sha256
        with tempfile.TemporaryDirectory(prefix='touch4d-condition-plots-') as tmp:
            out=Path(tmp)
            for directory in ('figures','images','arrays'):(out/directory).mkdir()
            source=out/'images'/'val_ep3_s135';source.mkdir();hashes={}
            for camera in range(3):
                for channel in ('rgb','alpha'):
                    for i,mode in enumerate(('GT',*MODES,'Copy')):
                        path=source/f'h15_v{camera}_{channel}_{mode}.png'
                        Image.new('RGB',(16,9),(i*20,camera*70,100)).save(path);hashes[path]=sha256(path)
            points=np.array([[0.,0.,0.],[.01,.02,.03],[-.01,-.02,-.03]])
            arrays=dict(initial_world=points)
            for mode in (*MODES,'Copy'):arrays[mode+'_world']=np.repeat(points[None],15,axis=0)
            arrays.update({f'gt_h{h:02}_world':points for h in range(1,16)})
            np.savez_compressed(out/'arrays'/'val_ep3_s135.npz',**arrays)
            data={k:[] for k in ('horizons_val_summary','kv_attribution_val_summary','branch_changes_val_summary','condition_activity')}
            generated=plots(out,data)
            self.assertEqual(len(generated),16)
            self.assertTrue(all(sha256(p)==h for p,h in hashes.items()))
            with Image.open(out/'figures'/'val_ep3_s135_h15_v0_rgb_comparison.png') as image:self.assertEqual(image.size,(1920,1602))


if __name__=='__main__':unittest.main()
