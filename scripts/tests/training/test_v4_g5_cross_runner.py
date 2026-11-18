"""CPU integration of real predictor/renderer/loss, without training or CUDA."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.data_process import DataProcessor
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.experiments.v4_stage1.fixtures import synthetic_batch, analytic_rasterizer
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_cross_necessity.interventions import MODES, intervention, one_frame
from scripts.experiments.v4_g5_cross_necessity.metrics import (
    projection_stats, aggregate, decompose)
from scripts.experiments.v4_g5_cross_necessity.run import (
    CachedRenderer, horizon_rows, structure_rows, single_effects, response_effects, write_json, csv_write)


def first_sample(tree):
    if isinstance(tree, dict):
        return {k: first_sample(v) for k, v in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree[:1] if tree.ndim and tree.shape[0] == 2 else tree
    if isinstance(tree, list):
        return [first_sample(v) for v in tree]
    return tree


class CrossRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        torch.manual_seed(461)
        model = G5DirectPredictor()
        with torch.no_grad():
            model.motion[-1].weight.normal_(std=.0002)
            model.global_motion.mapping[-1].weight.normal_(std=.0003)
            model.direct_motion.mapping[-1].weight.normal_(std=.0003)
        model.eval().requires_grad_(False)
        processor = DataProcessor('001-rope')
        batch = first_sample(prepare_batch(synthetic_batch(55,15,1),processor,model))
        renderer = GaussianRenderer(rasterizer=analytic_rasterizer)
        return model, processor, batch, renderer

    @torch.inference_mode()
    def test_current_step_pipeline_uses_original_renderer_loss_and_real_replay(self):
        model, processor, batch, renderer = self.fixture()
        digest = state_hash(model.state_dict())
        with intervention(model) as raw:
            normal, trace = model(batch['inputs'], return_trace=True)
        target = renderer.render_targets(batch, processor)
        _, metrics = v4_objective(normal,batch,processor,CachedRenderer(renderer,target))
        scope = dict(split='val',episode=3,start=135)
        rows = horizon_rows(normal,batch,metrics,normal,processor.state_dict()['scale'],'Normal',scope)
        effects, checks, regions = single_effects(model,batch,normal,trace,raw,rows,
            renderer,target,processor,scope,verify=False)
        self.assertEqual(len(effects),4*(len(MODES)-1))
        self.assertEqual(checks,[])
        self.assertEqual(len(regions),4*4*(len(MODES)-1))
        self.assertEqual({r['history'] for r in effects},{'fixed_Normal'})
        for r in effects:
            self.assertTrue(all(torch.isfinite(torch.tensor(r[k])) for k in
                ('cd_mm','rgb','alpha','joint','change_rms_mm')))
        self.assertEqual(digest,state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    @torch.inference_mode()
    def test_response_pipeline_never_claims_counterfactual_gt(self):
        model,processor,batch,renderer=self.fixture()
        normal=model(batch['inputs'])
        target=renderer.render_targets(batch,processor)
        digest=state_hash(model.state_dict())
        rows,arrays=response_effects(model,batch,normal,renderer,target,processor,
                                    dict(split='val',episode=3,start=135))
        self.assertEqual(len(rows),2*2*15)
        self.assertEqual(len(arrays),4)
        self.assertTrue(all(r['counterfactual_ground_truth_available'] is False for r in rows))
        self.assertTrue(all(r['output_change_rms_mm']==0 for r in rows if r['horizon']==1))
        self.assertEqual(digest,state_hash(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @torch.inference_mode()
    def test_unsupported_response_is_not_a_zero_response_or_h1_relaxation(self):
        model,processor,batch,renderer=self.fixture()
        batch['inputs']['action_valid'][0,0,1]=False
        normal=model(batch['inputs'])
        target=renderer.render_targets(batch,processor)
        rows,arrays=response_effects(model,batch,normal,renderer,target,processor,
                                    dict(split='val',episode=3,start=135))
        unavailable=[r for r in rows if not r['available']]
        self.assertEqual(len(unavailable),30)
        self.assertTrue(all(r['response']=='HalfRootTranslation' for r in unavailable))
        self.assertTrue(all(r['output_change_rms_mm'] is None for r in unavailable))
        self.assertTrue(all(not r['measurable_above_numeric_floor'] for r in unavailable))
        self.assertEqual(len(arrays),2)

    @torch.inference_mode()
    def test_regional_error_is_one_way_to_full_independent_gt(self):
        model,processor,batch,_=self.fixture()
        prediction=model(batch['inputs'])
        rows=structure_rows(prediction,batch,processor.state_dict()['scale'],
                            dict(split='val',episode=3,start=135),'Normal')
        self.assertEqual(len(rows),60)
        self.assertEqual({r['region'] for r in rows},{'all','near','middle','far'})
        self.assertTrue(all(r['pred_region_to_full_gt_mm']>=0 for r in rows))
        self.assertTrue(all('neighbor_distortion_p95' in r for r in rows))

    @torch.inference_mode()
    def test_branch_projection_and_field_sum_on_actual_predictor(self):
        model,processor,batch,_=self.fixture()
        _,trace=model(batch['inputs'],return_trace=True)
        branches,cancel,checks=decompose(batch['inputs'],trace,processor.state_dict()['scale'])
        self.assertEqual(len(checks),15)
        self.assertTrue(cancel)
        self.assertEqual({r['space'] for r in branches},{'anchor_control','gs_actual'})
        self.assertTrue(all('signed_projection_mean_mm' in r for r in branches))
        self.assertTrue(all(r['xyz_bit_exact'] for r in checks))

    def test_projection_is_signed_and_not_clamped_to_unit_fraction(self):
        got=projection_stats(torch.tensor([[3.,0.,0.]]),torch.tensor([[2.,0.,0.]]))
        self.assertAlmostEqual(got['signed_projection_mean_mm'],3.)
        self.assertAlmostEqual(got['final_fraction_mean'],1.5)
        got=projection_stats(torch.tensor([[-3.,0.,0.]]),torch.tensor([[2.,0.,0.]]))
        self.assertAlmostEqual(got['signed_projection_mean_mm'],-3.)
        self.assertAlmostEqual(got['final_fraction_mean'],-1.5)
        got=projection_stats(torch.ones(2,3),torch.zeros(2,3))
        self.assertEqual(got['projection_undefined'],2)
        self.assertIsNone(got['final_fraction_mean'])

    def test_episode_equal_aggregation_and_undefined_count(self):
        rows=[dict(split='val',episode=3,mode='A',value=1.) for _ in range(55)]
        rows += [dict(split='val',episode=6,mode='A',value=3.) for _ in range(45)]
        rows += [dict(split='train',episode=7,mode='A',value=1000.)]
        got=aggregate(rows,('mode',),('value',))[0]
        self.assertEqual(got['value'],2.)
        self.assertEqual(got['episode_window_counts'],{'3':55,'6':45})

    @torch.inference_mode()
    def test_one_frame_uses_selected_real_target_not_future_as_input(self):
        model,processor,batch,renderer=self.fixture()
        normal=model(batch['inputs']);original=copy.deepcopy(batch['inputs'])
        for index in (0,4,9,14):
            view,pred=one_frame(batch,normal,index)
            self.assertIs(view['inputs'],batch['inputs'])
            self.assertIs(view['targets']['target_xyz'][0],batch['targets']['target_xyz'][index])
            self.assertEqual(pred['xyz'].shape[1],1)
            loss,_=v4_objective(pred,view,processor,renderer)
            self.assertTrue(torch.isfinite(loss))
        for key in original:
            if isinstance(original[key],torch.Tensor):
                torch.testing.assert_close(original[key],batch['inputs'][key],rtol=0,atol=0)

    def test_output_helpers_refuse_overwrite_and_nonfinite(self):
        with tempfile.TemporaryDirectory(prefix='touch4d-g5-necessity-') as tmp:
            p=Path(tmp)/'result.json';write_json(p,{'synthetic_test_only':True})
            with self.assertRaises(FileExistsError):write_json(p,{})
            with self.assertRaises(ValueError):write_json(Path(tmp)/'invalid.json',{'x':float('nan')})
            with self.assertRaises(ValueError):csv_write(Path(tmp)/'empty.csv',[])

    def test_runner_has_no_optimizer_backward_or_step_call(self):
        import scripts.experiments.v4_g5_cross_necessity.run as run
        tree=ast.parse(Path(run.__file__).read_text(encoding='utf8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and
            n.func.attr in ('backward','step','Adam','AdamW','SGD') for n in calls))


if __name__=='__main__':
    unittest.main()
