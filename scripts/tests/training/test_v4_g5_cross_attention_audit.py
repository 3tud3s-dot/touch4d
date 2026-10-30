"""All new tests use frozen synthetic weights: no optimizer or backward."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.training.model import masked
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_cross_attention_audit.interventions import (
    MODES, SELECTED, intervention, replay_step, switches)
from scripts.experiments.v4_g5_cross_attention_audit.diagnostics import attention_rows,same_history_regions
from scripts.experiments.v4_g5_cross_attention_audit.run import isolation_check


class FrozenCrossAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def fixture(self,horizon=5):
        torch.manual_seed(470)
        model=G5DirectPredictor(horizon=horizon)
        with torch.no_grad():
            model.motion[-1].weight.normal_(std=.0002)
            model.global_motion.mapping[-1].weight.normal_(std=.0003)
            model.direct_motion.mapping[-1].weight.normal_(std=.0003)
            for block in model.cross:block.q.weight.normal_(std=.02)
        model.eval().requires_grad_(False);processor=DataProcessor('001-rope')
        batch=prepare_batch(synthetic_batch(55,horizon,1),processor,model)
        return model,processor,batch

    def test_exact_six_modes_and_no_base_deletion(self):
        self.assertEqual(MODES,('Normal','NoCrossValue','NoToolMotion12','NoHeadMass4','NoValue+NoTool12','NoCrossInfo'))
        for mode in MODES:self.assertTrue(switches(mode,0)[0])

    @torch.inference_mode()
    def test_normal_h15_bit_exact_and_state_unchanged(self):
        model,_,batch=self.fixture(15);x=batch['inputs'];before=state_hash(model.state_dict())
        expected=model(x)
        with intervention(model) as rows:actual,trace=model(x,return_trace=True)
        self.assertEqual(len(rows),15)
        for key in expected:torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)
        self.assertEqual(before,state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @torch.inference_mode()
    def test_value_projection_only_ffn_scores_and_history_remain(self):
        model,_,batch=self.fixture()
        with intervention(model,'NoCrossValue') as rows:pred,trace=model(batch['inputs'],return_trace=True)
        raw=rows[3]
        self.assertTrue(any(float(u.abs().max())>0 for u in raw['cross_projection_raw']))
        for block,call in zip(model.cross,raw['cross_calls']):
            q=call['args'][0];mask=call['args'][5]
            expected=masked(q+block.ff(block.ff_norm(q)),mask)
            torch.testing.assert_close(expected,call['output'][0],rtol=0,atol=0)
            torch.testing.assert_close(call['output'][1].sum(-1),torch.ones_like(call['output'][1][...,0]),rtol=1e-6,atol=1e-7)
        self.assertGreater(float(trace['steps'][3]['head_history_mass'].abs().max()),0)
        self.assertGreater(float(trace['steps'][3]['tool_motion'].abs().max()),0)

    @torch.inference_mode()
    def test_tool_and_mass_channels_separate_preserve_other_heads_on_same_history(self):
        model,_,batch=self.fixture();x=batch['inputs']
        with intervention(model):normal,nt=model(x,return_trace=True)
        for mode,zero_slice in [('NoToolMotion12',slice(128,140)),('NoHeadMass4',slice(146,150))]:
            with intervention(model,mode,only_step=3) as rows:pred,trace=model(x,return_trace=True)
            raw=rows[3];original=raw['motion_input'];effective=raw['effective_motion_input']
            self.assertEqual(int(effective[...,zero_slice].count_nonzero()),0)
            allowed=torch.ones(151,dtype=torch.bool);allowed[zero_slice]=False
            torch.testing.assert_close(original[...,allowed],effective[...,allowed],rtol=0,atol=0)
            for key in ('global_delta','local_correction','direct_seed_per_hand','tool_motion','head_history_mass'):
                torch.testing.assert_close(trace['steps'][3][key],nt['steps'][3][key],rtol=0,atol=0)
            torch.testing.assert_close(normal['xyz'][:,:3],pred['xyz'][:,:3],rtol=0,atol=0)

    @torch.inference_mode()
    def test_no_cross_info_kv_scores_isolated_not_no_action(self):
        model,_,batch=self.fixture();x=batch['inputs'];before=state_hash(model.state_dict())
        result=isolation_check(model,x)
        self.assertTrue(result['all_attributes_bit_exact']);self.assertGreater(result['max_attention_weight_change'],0)
        with intervention(model,'NoCrossInfo'):normal=model(x)
        changed=copy.deepcopy(x);changed['controller_points'][:,1:]+=0.03
        with intervention(model,'NoCrossInfo'):other=model(changed)
        torch.testing.assert_close(normal['xyz'][:,0],other['xyz'][:,0],rtol=0,atol=0)
        self.assertFalse(torch.equal(normal['xyz'],other['xyz']))
        self.assertEqual(before,state_hash(model.state_dict()))

    @torch.inference_mode()
    def test_no_cross_info_also_closes_current_tactile_output_route(self):
        model,_,batch=self.fixture();x=batch['inputs'];changed=copy.deepcopy(x)
        changed['tactile']+=10
        with intervention(model,'NoCrossInfo'):a=model(x)
        with intervention(model,'NoCrossInfo'):b=model(changed)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    @torch.inference_mode()
    def test_h15_same_history_replay_matches_only_step_all_modes(self):
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
    def test_replay_recomputes_global_local_after_value_change(self):
        model,_,batch=self.fixture();x=batch['inputs']
        with intervention(model) as rows:_,trace=model(x,return_trace=True)
        a=replay_step(model,x,trace,rows,3,'Normal');b=replay_step(model,x,trace,rows,3,'NoCrossValue')
        for key in ('query','base_cumulative_d','global_delta','local_correction'):
            self.assertFalse(torch.equal(a[key],b[key]),key)

    @torch.inference_mode()
    def test_all_modes_padding_masks_attributes_and_causal_prefix(self):
        model,_,batch=self.fixture();x=batch['inputs'];changed=copy.deepcopy(x)
        changed['controller_points'][:,3:]+=0.2
        for mode in MODES:
            with intervention(model,mode):a=model(x)
            with intervention(model,mode):b=model(changed)
            self.assertTrue(torch.isfinite(b['xyz']).all())
            torch.testing.assert_close(a['xyz'][:,:3],b['xyz'][:,:3],rtol=0,atol=0)
            self.assertEqual(int(b['xyz'][~x['gs_mask'][:,None].expand(-1,5,-1)].count_nonzero()),0)
            for key in a:
                if key!='xyz':torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    @torch.inference_mode()
    def test_h1_not_assumed_invariant_when_value_path_changes(self):
        model,_,batch=self.fixture();x=batch['inputs']
        with intervention(model):a=model(x)
        with intervention(model,'NoCrossValue'):b=model(x)
        self.assertFalse(torch.equal(a['xyz'][:,0],b['xyz'][:,0]))
        for mode in ('NoToolMotion12','NoHeadMass4'):
            with intervention(model,mode):b=model(x)
            torch.testing.assert_close(a['xyz'][:,0],b['xyz'][:,0],rtol=0,atol=0)

    @torch.inference_mode()
    def test_attention_statistics_and_same_history_spatial_fields(self):
        model,processor,batch=self.fixture();x={k:v[:1] for k,v in batch['inputs'].items()}
        with intervention(model) as rows:pred,trace=model(x,return_trace=True)
        scope=dict(split='val',episode=3,start=135)
        table=attention_rows(x,trace,rows,processor.state_dict()['scale'],scope)
        self.assertEqual(len(table),5*2*4*4)
        for row in table:self.assertAlmostEqual(row['controller_mass']+row['null_mass'],1.,places=6)
        for row in table:
            if row['horizon']==1 and row['layer']==2:self.assertEqual(row['final_layer_tool_vector_rms_mm'],0)
        normal=replay_step(model,x,trace,rows,3)
        changed=replay_step(model,x,trace,rows,3,'NoCrossInfo')
        n=dict(trace['steps'][3],_query=normal['query'])
        gt=batch['targets']['target_xyz'][3][0,batch['targets']['target_mask'][3][0]]
        result=same_history_regions(x,trace,n,changed,gt,processor.state_dict()['scale'],dict(**scope,mode='NoCrossInfo',horizon=4))
        self.assertEqual({r['region'] for r in result},{'all','near','middle','far'})
        self.assertTrue(all(r['change_rms_mm']>=0 for r in result))

    @torch.inference_mode()
    def test_all_controllers_invalid_null_finite_all_modes(self):
        model,_,batch=self.fixture();x=copy.deepcopy(batch['inputs'])
        x['controller_mask'].zero_();x['controller_points'].fill_(float('nan'))
        for mode in MODES:
            with intervention(model,mode):pred,trace=model(x,return_trace=True)
            self.assertTrue(torch.isfinite(pred['xyz']).all())
            for s in trace['steps']:
                torch.testing.assert_close(s['alpha'][...,-1],torch.ones_like(s['alpha'][...,-1]),rtol=0,atol=0)

    def test_cleanup_and_wrong_mode_or_history_rejected(self):
        model,_,batch=self.fixture()
        with self.assertRaises(RuntimeError):
            with intervention(model,'NoCrossInfo'):raise RuntimeError('synthetic interruption')
        self.assertFalse(model.motion._forward_pre_hooks)
        for block in model.cross:self.assertFalse(block.projection._forward_hooks)
        with self.assertRaises(ValueError):
            with intervention(model,'NO_SUCH_MODE'):pass
        with self.assertRaises(ValueError):
            with intervention(model,only_step=5):pass

    @torch.inference_mode()
    def test_masked_hand_nan_cannot_leak_through_any_mode(self):
        model,_,batch=self.fixture();x={k:v[:1] for k,v in batch['inputs'].items()}
        self.assertFalse(bool(x['operation_hands'][0,0]))
        changed=copy.deepcopy(x)
        changed['action_state'][:,:,0]=float('nan')
        # Dataset stores [B,H,2,30,3], not an already flattened 60-node axis.
        changed['controller_points'].reshape(1,5,60,3)[:,:,:30]=float('nan')
        torch.testing.assert_close(changed['controller_points'][:,:,1],x['controller_points'][:,:,1],rtol=0,atol=0)
        changed['tactile'][:,:,:2]=float('nan')
        for mode in MODES:
            with intervention(model,mode):a=model(x)
            with intervention(model,mode):b=model(changed)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    @torch.inference_mode()
    def test_mock_renderer_original_joint_loss_all_modes(self):
        from scripts.training.rendering import GaussianRenderer
        from scripts.training.v4_runtime import v4_objective
        from scripts.experiments.v4_stage1.fixtures import analytic_rasterizer
        model,processor,batch=self.fixture();renderer=GaussianRenderer(rasterizer=analytic_rasterizer)
        for mode in MODES:
            with intervention(model,mode):pred=model(batch['inputs'])
            loss,m=v4_objective(pred,batch,processor,renderer)
            self.assertTrue(torch.isfinite(loss))
            torch.testing.assert_close(loss,(m['geometry']+m['rgb']+.1*m['alpha']).mean(),rtol=0,atol=0)

    def test_no_new_optimizer_backward_or_step_calls(self):
        import ast
        root=Path(__file__).resolve().parents[2]/'experiments'/'v4_g5_cross_attention_audit'
        for path in root.glob('*.py'):
            node=ast.parse(path.read_text(encoding='utf8'))
            for call in (n for n in ast.walk(node) if isinstance(n,ast.Call)):
                if isinstance(call.func,ast.Attribute):
                    self.assertNotIn(call.func.attr,('backward','step','AdamW','Adam','SGD'),str(path))

    def test_report_uses_records_and_refuses_overwrite_or_incomplete_run(self):
        from scripts.experiments.v4_g5_cross_attention_audit.report import build
        episode=lambda count:dict(samples=count,geometry_mm=2.,supervised_geometry=.02,rgb=.1,alpha=.1)
        summary={mode:dict(geometry_mm=2.,rgb=.1,alpha=.1,objective=.13,episodes={'3':episode(55),'6':episode(45)}) for mode in (*MODES,'Copy')}
        data={name:[] for name in ('horizons','horizons_val_summary','same_history_val_summary','same_history_regions','attention_trace_val_summary','windows','structure','intervention_checks')}
        data['val_summary']=summary
        provenance=dict(training_source_sha='SYNTHETIC_ONLY',diagnostic_source_sha='TEST',checkpoint='NONE',checkpoint_sha256='0'*64,
                        job_id='NO_GPU',gpu='NONE',torch='TEST',cuda='NONE',gsplat='MOCK',elapsed_seconds=0.,peak_allocated_bytes=0)
        with tempfile.TemporaryDirectory(prefix='touch4d-cross-audit-') as tmp:
            out=Path(tmp);(out/'results').mkdir()
            with self.assertRaises(RuntimeError):build(out)
            (out/'completed.json').write_text('{}',encoding='utf8')
            for name,value in data.items():(out/'results'/(name+'.json')).write_text(json.dumps(value),encoding='utf8')
            (out/'provenance.json').write_text(json.dumps(provenance),encoding='utf8')
            with patch('scripts.experiments.v4_g5_cross_attention_audit.report.plots',return_value=[]):
                build(out)
                report=(out/'final_report.md').read_text(encoding='utf8')
                for word in ('SYNTHETIC_ONLY','NoCrossInfo','NoHeadMass4','FFN'):
                    self.assertIn(word,report)
                with self.assertRaises(FileExistsError):build(out)


if __name__=='__main__':unittest.main()
