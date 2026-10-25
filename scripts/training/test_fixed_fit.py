"""Disk-free CPU acceptance: python -B -m unittest scripts.training.test_fixed_fit -v."""

import copy
from io import StringIO
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ..data_process import DataProcessor
from ..data_process.test_data_process import make_batch, add_depth_targets
from . import fixed_fit as fit
from .model import PositionPredictor
from .rendering import GaussianRenderer
from .test_training import fake_gaussian_rasterizer
from .train import checkpoint_state, copy_initial, restore_training, parser, main as train_main
from .visualize import ellipsoid_axes, geometry_data, geometry_html
from .evaluate import main as evaluate_main


def contract(scene="001-rope"):
    # Deliberately loose synthetic fixtures, NOT proposed experimental thresholds.
    return dict(version="p0-fit-v1", scene=scene, supervision_review="synthetic unit fixture",
                tie_cd_mm=.01, tie_h5_mm=.01, tie_render=.001,
                windows={fit.window_key(*w): dict(rgb_max=[1.]*5, alpha_max=[1.]*5,
                    cd_mm_max=[1e6]*5, motion_rms_mm_max=[1e6]*5, neighbor_rms_mm_max=[1e6]*5,
                    axis_ratio_range=[.001,1000.], volume_ratio_range=[.001,1000.],
                    scale_mm_range=[1e-6,1e6], cov_condition_max=1e12) for w in fit.WINDOWS[scene]})


def single_raw(scene="001-rope", window=None):
    def first(v):
        if isinstance(v, torch.Tensor):
            return v[:1].clone()
        if isinstance(v, dict):
            return {key: first(value) for key, value in v.items()}
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            return [first(value) for value in v]
        return copy.deepcopy(v[:1])
    raw = first(add_depth_targets(make_batch(scene)))
    episode, start = window or fit.WINDOWS[scene][0]
    raw["meta"]["episode_id"].fill_(episode)
    raw["meta"]["frame_ids"] = torch.arange(start,start+6)[None]
    return raw


class FixedFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.raw = single_raw()
        self.processor = DataProcessor("001-rope")
        self.batch = self.processor(self.raw)
        self.model = PositionPredictor(width=16, heads=4, layers=1, controller_features=7,
                                       prediction_mode="autoregressive", output_mode="gs_geometry",
                                       gs_local_attention="knn", gs_local_k=16)
        self.config = dict(self.model.config, scene="001-rope", depth_weight=0., chamfer_chunk=2,
                           **fit.WEIGHTS, protocol="fixed400", fit_contract=contract())
        self.renderer = fit.TargetCache(GaussianRenderer(fake_gaussian_rasterizer))

    def test_exact_windows_validate_actual_metadata_not_just_indices(self):
        for scene, windows in fit.WINDOWS.items():
            raws = [single_raw(scene, w) for w in windows]
            dataset = SimpleNamespace(windows=[(99,0), *windows])
            class Dataset:
                windows = dataset.windows
                def __getitem__(self, index):
                    return raws[index-1]
            with patch.object(fit, "collate_fn", side_effect=lambda samples: samples[0]):
                selected = fit.select_windows(Dataset(), scene)
                self.assertEqual([int(r["meta"]["frame_ids"][0,0]) for r in selected], [w[1] for w in windows])
                raws[0]["meta"]["frame_ids"][0,-1] += 1
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    fit.select_windows(Dataset(), scene)

    def test_missing_duplicate_requests_and_dataset_duplicates_fail(self):
        for windows, requested in [([],None), ([(1,85),(1,85),(2,73)],None),
                                   ([(1,85)],[(1,85),(1,85)])]:
            with self.assertRaises(ValueError):
                fit.select_windows(SimpleNamespace(windows=windows), "001-rope", requested)

    def test_manifest_freezes_data_and_positive_ignore_without_negative(self):
        cfg = dict(depth_cameras=["cam_a","cam_b"])
        rows = fit.manifest([self.raw], cfg)
        self.assertEqual(rows[0]["negative_pixels"],0)
        self.assertEqual(rows[0]["positive_pixels"], [[11,11]]*5)
        self.assertEqual(rows[0]["ignore_pixels"], [[1,1]]*5)
        changed = copy.deepcopy(self.raw)
        changed["inputs"]["controller_points"][0,0,0,0,0] += .001
        self.assertNotEqual(rows[0]["data_sha256"],fit.manifest([changed],cfg)[0]["data_sha256"])
        changed["targets"]["target_depth_valid"][:,2,0] = False
        with self.assertRaisesRegex(ValueError,"Positive"):
            fit.manifest([changed],cfg)
        with self.assertRaisesRegex(ValueError,"camera"):
            fit.manifest([self.raw],dict(depth_cameras=["cam_b","cam_a"]))

    def test_contract_requires_explicit_scene_limits_and_evidence(self):
        for scene in fit.WINDOWS:
            self.assertEqual(fit.validate_contract(contract(scene),scene),contract(scene))
        for mutation in (lambda c:c.pop("supervision_review"),
                         lambda c:c.update(tie_cd_mm=float("nan")),
                         lambda c:c["windows"]["1:85"].update(rgb_max=[1.]*4),
                         lambda c:c["windows"]["1:85"].update(axis_ratio_range=[1,0]),
                         lambda c:c["windows"].pop("2:73")):
            c=contract(); mutation(c)
            with self.assertRaises(ValueError):
                fit.validate_contract(c,"001-rope")

    def test_translation_rms_neighbor_difference_units_padding_and_covariance(self):
        pred={k:v.clone() for k,v in copy_initial(self.batch["inputs"],"gs_geometry").items()}
        scale=self.processor.state_dict()["scale"]
        pred["xyz"] += torch.tensor([.001,.002,-.003])/scale
        keep=self.batch["inputs"]["gs_mask"]
        for value in pred.values():
            value[~keep[:,None].expand(-1,5,-1)] = float("nan")
        rows=fit.geometry_diagnostics(pred,self.batch,scale)
        for row in rows:
            np.testing.assert_allclose(row["mean_displacement_mm"],[1,2,-3],atol=1e-4)
            self.assertLess(row["translation_rms_mm"],1e-4)
            self.assertLess(row["neighbor_rms_mm"],1e-4)
            self.assertEqual(row["axis_ratio"],[1.,1.])
            self.assertEqual(row["volume_ratio"],[1.,1.])
            self.assertTrue(row["covariance_positive"])
        pred["xyz"][0,4,0,0] += .01/scale
        pred["log_scales"][0,4,keep[0]] += np.log(2)
        row=fit.geometry_diagnostics(pred,self.batch,scale)[-1]
        self.assertGreater(row["translation_rms_mm"],0)
        self.assertGreater(row["neighbor_rms_mm"],0)
        np.testing.assert_allclose(row["volume_ratio"],[8,8],rtol=1e-6)

    def test_nonfinite_scales_and_zero_quaternion_rejected(self):
        for field,value in (("xyz",float("nan")),("log_scales",1000.),("quaternions",0.)):
            pred={k:v.clone() for k,v in copy_initial(self.batch["inputs"],"gs_geometry").items()}
            pred[field][0,2,0]=value
            with self.assertRaises(FloatingPointError):
                fit.geometry_diagnostics(pred,self.batch,self.processor.state_dict()["scale"])

    def test_metrics_ignore_gt_row_order_and_restore_training_mode(self):
        batches=[self.batch,self.processor(single_raw(window=(2,73)))]
        self.model.train()
        result=fit.evaluate_fixed({"model":self.model},batches,self.processor,self.renderer,self.config)
        self.assertTrue(self.model.training)
        self.assertEqual(set(result["model"]["windows"]),{"1:85","2:73"})
        self.assertEqual(result["model"]["cd_mm"],result["copy_initial"]["cd_mm"])
        self.assertEqual(len(result["model"]["rgb"]),5)
        changed=copy.deepcopy(self.batch)
        for h in range(5):
            mask=changed["targets"]["target_mask"][h][0]
            for field in ("target_xyz","target_attr"):
                t=changed["targets"][field][h]
                t[0,mask]=t[0,mask].flip(0)
        pred=copy_initial(self.batch["inputs"],"gs_geometry")
        a=fit.prediction_metrics(pred,self.batch,self.processor,self.renderer,2)
        b=fit.prediction_metrics(pred,changed,self.processor,self.renderer,2)
        np.testing.assert_allclose(a["cd_mm"],b["cd_mm"],atol=1e-6)

    def test_mask_only_targets_no_predicted_alpha_mask_and_target_cache(self):
        renderer=GaussianRenderer(fake_gaussian_rasterizer)
        with patch.object(renderer,"render_targets",wraps=renderer.render_targets) as render:
            cached=fit.TargetCache(renderer)
            first=cached.render_targets(self.batch,self.processor)
            self.assertIs(first,cached.render_targets(self.batch,self.processor))
            self.assertEqual(render.call_count,1)
            self.assertTrue(all(not v.requires_grad for v in first.values()))
        mask=self.batch["targets"]["target_depth_valid"].clone()
        fit.evaluate_fixed({"model":self.model},[self.batch],self.processor,self.renderer,self.config)
        self.assertTrue(torch.equal(mask,self.batch["targets"]["target_depth_valid"]))

    def fixture_metrics(self):
        return fit.evaluate_fixed({"model":self.model},
            [self.batch,self.processor(single_raw(window=(2,73)))],self.processor,self.renderer,self.config)

    def test_candidate_render_shape_gates_ties_and_manual_pending(self):
        metrics=self.fixture_metrics()
        row=fit.candidate(metrics,contract(),20)
        self.assertTrue(row["eligible"])
        self.assertFalse(row["automatic_fit_checks"]) # zero head equals copy, not an improvement
        self.assertEqual(row["manual_structure_review"],"pending")
        self.assertTrue(fit.better(row,None,contract()))
        same=copy.deepcopy(row);same["update"]=40
        self.assertFalse(fit.better(same,row,contract()))
        improved=copy.deepcopy(row);improved["rank"][1]-=.1
        self.assertTrue(fit.better(improved,row,contract()))
        bad=copy.deepcopy(metrics);bad["model"]["windows"]["1:85"]["rgb"][3]=2.
        rejected=fit.candidate(bad,contract(),40)
        self.assertFalse(rejected["eligible"])
        self.assertFalse(fit.better(rejected,None,contract()))
        bad=copy.deepcopy(metrics);bad["model"]["windows"]["1:85"]["diagnostics"][2]["axis_ratio"]=[1.,2000.]
        self.assertFalse(fit.candidate(bad,contract(),40)["eligible"])

    def test_400_actual_optimizer_steps_alternate_windows_and_evaluate_both(self):
        # Tiny scalar optimization validates scheduling; no scene training or CUDA.
        model=torch.nn.Linear(1,1,bias=False)
        optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        seen,evaluations=[],[]
        batches=[dict(inputs=torch.tensor([[float(i+1)]]),window=i) for i in range(2)]
        def loss(prediction,batch,*args,**kwargs):
            seen.append(batch["window"])
            self.assertEqual(kwargs,fit.WEIGHTS)
            return prediction.square().sum(),dict(render_time_available=torch.ones(1,5,dtype=torch.bool))
        with patch.object(fit,"objective",side_effect=loss), patch.object(optimizer,"step",wraps=optimizer.step) as step:
            fit.update_loop(model,optimizer,batches,None,None,dict(chamfer_chunk=2),
                            lambda n,*rest:evaluations.append(n),evaluate=lambda:{"both_windows":True})
            self.assertEqual(step.call_count,400)
        self.assertEqual(seen,[0,1]*200)
        self.assertEqual(evaluations,list(range(20,401,20)))

    def test_resume_count_and_empty_supervision_does_not_count_a_step(self):
        model=torch.nn.Linear(1,1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        batches=[dict(inputs=torch.ones(1,1))]*2
        with patch.object(fit,"objective",return_value=(torch.tensor(0.,requires_grad=True),
                           dict(render_time_available=torch.zeros(1,5,dtype=torch.bool)))), \
             patch.object(optimizer,"step") as step:
            with self.assertRaisesRegex(ValueError,"unobserved"):
                fit.update_loop(model,optimizer,batches,None,None,dict(chamfer_chunk=2),lambda *a:None,start=380)
            step.assert_not_called()
        seen=[]
        with patch.object(fit,"objective",side_effect=lambda p,*a,**kw:
              (p.square().sum(),dict(render_time_available=torch.ones(1,5,dtype=torch.bool)))):
            fit.update_loop(model,optimizer,batches,None,None,dict(chamfer_chunk=2),
                            lambda n,*a:seen.append(n),start=380,evaluate=lambda:{})
        self.assertEqual(seen,[400])
        with self.assertRaises(ValueError):
            fit.update_loop(model,optimizer,batches,None,None,{},lambda *a:None,start=381)

    def checkpoint(self,update=20):
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        row=fit.candidate(self.fixture_metrics(),contract(),update)
        state=checkpoint_state(self.model,optimizer,self.processor,self.config,update//2-1,row["rank"][0],[])
        state["fit"]=dict(completed_updates=update,best_candidate=row,manifest=["synthetic"])
        return state

    def test_checkpoint_publish_order_last400_and_no_legacy_best(self):
        for update,expected in ((20,["best_fit.pt","last.pt"]),(400,["best_fit.pt","last.pt","last_400.pt"])):
            state=self.checkpoint(update)
            with patch.object(fit,"atomic_checkpoint") as save:
                fit.publish_fit(Path("memory"),state,True)
                self.assertEqual([call.args[0].name for call in save.call_args_list],expected)
            with patch.object(fit,"atomic_checkpoint") as save:
                fit.publish_fit(Path("memory"),state,False)
                self.assertNotIn("best_fit.pt",[call.args[0].name for call in save.call_args_list])

    def test_diagnostic_mode_retains_checkpoints_without_acceptance_or_thresholds(self):
        metrics=self.fixture_metrics()
        row=fit.candidate(metrics,None,20)
        self.assertIsNone(row["eligible"])
        self.assertIsNone(row["automatic_fit_checks"])
        self.assertTrue(fit.better(row,None,None))
        state=self.checkpoint()
        state["config"].update(fit_diagnostic=True,fit_contract=None)
        state["fit"]["best_candidate"]=row
        with patch.object(fit,"atomic_checkpoint") as save:
            fit.publish_fit(Path("memory"),state,True)
            self.assertEqual([c.args[0].name for c in save.call_args_list],
                             ["step_000020.pt","lowest_cd.pt","last.pt"])
        with patch("pathlib.Path.is_file",return_value=True),patch.object(fit,"read_checkpoint",return_value=state):
            self.assertEqual(fit.recover_fit_best(state,"memory"),row)
        args=parser().parse_args(["--scene","001-rope","--protocol","fixed400","--fit-diagnostic",
                                 "--fit-contract","memory/reviewed.json"])
        with patch("pathlib.Path.read_text",return_value=json.dumps(contract())):
            with self.assertRaisesRegex(ValueError,"must not"):
                fit.run(args)

    def test_resume_preserves_better_best_after_publish_interruption(self):
        last=self.checkpoint(20); saved=copy.deepcopy(last)
        saved["fit"]["completed_updates"]=40
        saved["fit"]["best_candidate"]["update"]=40
        saved["fit"]["best_candidate"]["rank"][0]-=1.
        with patch("pathlib.Path.is_file",return_value=True),patch.object(fit,"read_checkpoint",return_value=saved):
            self.assertEqual(fit.recover_fit_best(last,"memory")["update"],40)
            saved["fit"]["best_candidate"]["rank"][0]+=2.
            with self.assertRaisesRegex(ValueError,"regressed"):
                fit.recover_fit_best(last,"memory")
        with patch("pathlib.Path.is_file",return_value=False):
            with self.assertRaisesRegex(ValueError,"missing"):
                fit.recover_fit_best(last,"memory")
            last["fit"]["best_candidate"]=None
            self.assertIsNone(fit.recover_fit_best(last,"memory"))

    def test_restore_new_metadata_keeps_exact_model_and_optimizer(self):
        state=self.checkpoint()
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        before=self.model(self.batch["inputs"])
        restore_training(state,self.model,optimizer)
        after=self.model(self.batch["inputs"])
        for key in before:
            torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
        changed=copy.deepcopy(state);changed["config"]["fit_contract"]["tie_cd_mm"]=99
        with patch("pathlib.Path.is_file",return_value=True),patch.object(fit,"read_checkpoint",return_value=changed):
            with self.assertRaisesRegex(ValueError,"protocol"):
                fit.recover_fit_best(state,"memory")

    def test_train_dispatch_is_opt_in_and_does_not_start_training(self):
        args=parser().parse_args(["--scene","001-rope"])
        self.assertEqual(args.protocol,"epochs")
        self.assertIsNone(args.chamfer_weight) # legacy resolution remains separate
        with patch.object(fit,"run",return_value=0) as run:
            self.assertEqual(train_main(["--scene","001-rope","--protocol","fixed400"]),0)
            run.assert_called_once()
        with self.assertRaisesRegex(ValueError,"reviewed"):
            fit.run(parser().parse_args(["--scene","001-rope","--protocol","fixed400"]))

    def test_run_orchestration_publishes_final_evaluates_and_exports_both_checkpoints(self, diagnostic=False):
        # All persistence/CUDA/training are mocked. Exercise real orchestration,
        # checkpoint schema, exact manifest, fit selection and final exports.
        model=PositionPredictor(output_mode="gs_geometry",prediction_mode="autoregressive",
                                controller_features=7,gs_local_attention="knn",gs_local_k=16)
        raws=[single_raw(window=w) for w in fit.WINDOWS["001-rope"]]
        for raw in raws:
            raw["meta"]["depth_camera_ids"]=[fit.GS_CAMERAS["001-rope"]]
            for key in ("target_depth","target_depth_valid"):
                raw["targets"][key]=raw["targets"][key][:,:,[0,1,0]]
            for key in ("camera_intrinsics","camera_to_world"):
                raw["targets"][key]=raw["targets"][key][:,[0,1,0]]
        metrics=self.fixture_metrics()
        checkpoints,outputs={},{}
        def save_checkpoint(path,value): checkpoints[Path(path).name]=copy.deepcopy(value)
        def save_json(path,value): outputs[Path(path).name]=copy.deepcopy(value)
        def simulate_updates(*args,**kwargs):
            callback=args[6]
            for update in range(20,401,20):callback(update,metrics,.5,.1)
        arguments=parser().parse_args(["--scene","001-rope","--protocol","fixed400",
                                     "--fit-contract","memory/reviewed.json","--run-id","unit-test"])
        if diagnostic:
            arguments.fit_contract=None
            arguments.fit_diagnostic=True
        with patch.object(fit,"fixed_batches",return_value=raws), \
             patch.object(fit,"choose_cameras",return_value=fit.GS_CAMERAS["001-rope"]), \
             patch.object(fit,"PositionPredictor",return_value=model), \
             patch.object(model,"to",return_value=model), \
             patch.object(fit,"to_device",side_effect=lambda raw,device:raw), \
             patch.object(fit.torch.cuda,"is_available",return_value=True), \
             patch.object(fit.torch.cuda,"get_rng_state_all",return_value=[]), \
             patch.object(fit.torch.cuda,"manual_seed_all"), \
             patch.dict("sys.modules",gsplat=SimpleNamespace(__version__="unit-test")), \
             patch.object(fit,"update_loop",side_effect=simulate_updates), \
             patch.object(fit,"evaluate_fixed",return_value=metrics) as evaluate, \
             patch.object(fit,"atomic_checkpoint",side_effect=save_checkpoint), \
             patch.object(fit,"atomic_json",side_effect=save_json), \
             patch.object(fit,"read_checkpoint",side_effect=lambda p:copy.deepcopy(checkpoints[Path(p).name])), \
             patch("scripts.training.visualize.export_gaussians") as export, \
             patch("pathlib.Path.read_text",return_value=json.dumps(contract())), \
             patch("pathlib.Path.is_file",side_effect=lambda p:p.name in checkpoints,autospec=True), \
             patch("pathlib.Path.exists",return_value=False),patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.write_text"),redirect_stdout(StringIO()):
            self.assertEqual(fit.run(arguments),0)
        name="lowest_cd.pt" if diagnostic else "best_fit.pt"
        expected={name,"last.pt","last_400.pt"}
        if diagnostic:
            expected.update(f"step_{update:06d}.pt" for update in range(20,401,20))
        self.assertEqual(set(checkpoints),expected)
        self.assertEqual(checkpoints["last_400.pt"]["fit"]["completed_updates"],400)
        self.assertEqual(checkpoints[name]["fit"]["completed_updates"],20)
        self.assertEqual(len(outputs["metrics.json"]),20)
        self.assertEqual(export.call_count,4)
        self.assertEqual(evaluate.call_count,2)
        self.assertEqual(outputs["fit_summary.json"]["acceptance"],
                         "not assessed: tolerances not frozen" if diagnostic else "manual structural review pending")

    def test_diagnostic_run_exports_lowest_cd_and_retains_all_twenty_states(self):
        self.test_run_orchestration_publishes_final_evaluates_and_exports_both_checkpoints(True)

    def test_fit_evaluation_cli_reuses_exact_windows_and_rejects_changed_manifest(self):
        from . import evaluate as evaluation
        cfg=dict(self.config,depth_cameras=["cam_a","cam_b"])
        raws=[self.raw,single_raw(window=(2,73))]
        evidence=fit.manifest(raws,cfg)
        saved=dict(fit=dict(completed_updates=400,manifest=evidence))
        with patch.object(evaluation,"load_model",return_value=(self.model,self.processor,cfg)), \
             patch.object(evaluation,"read_checkpoint",return_value=saved), \
             patch.object(evaluation,"make_renderer",return_value=GaussianRenderer(fake_gaussian_rasterizer)), \
             patch.object(fit,"fixed_batches",return_value=raws),redirect_stdout(StringIO()) as output:
            self.assertEqual(evaluate_main(["--checkpoint","memory/last_400.pt","--split","fit","--device","cpu"]),0)
            result=json.loads(output.getvalue())
            self.assertFalse(result["held_out_validation"])
            self.assertEqual(result["candidate"]["update"],400)
            self.assertEqual(len(result["metrics"]["model"]["windows"]),2)
            saved["fit"]["manifest"]=[]
            with self.assertRaisesRegex(ValueError,"manifest"):
                evaluate_main(["--checkpoint","memory/last.pt","--split","fit","--device","cpu"])

    def test_ellipsoid_axes_are_world_wxyz_and_covariance_correct(self):
        log=np.log([[1.,2.,3.]])
        axes=ellipsoid_axes(log,np.array([[1.,0.,0.,0.]]))[0]
        np.testing.assert_allclose(axes,np.diag([1,2,3]))
        q=np.array([[np.sqrt(.5),0.,0.,np.sqrt(.5)]])
        axes=ellipsoid_axes(log,q*2)[0]
        np.testing.assert_allclose(axes.T@axes,np.diag([4,1,9]),atol=1e-12)
        with self.assertRaises(ValueError):
            ellipsoid_axes(log,np.zeros((1,4)))

    def test_geometry_exports_distinct_centers_ellipsoids_template_paths(self):
        xyz=np.array([[0.,0.,0.],[.01,0.,0.],[.02,0.,0.]],dtype=np.float32)
        attr=np.zeros((3,11),dtype=np.float32);attr[:,:3]=-5;attr[:,3]=1
        arrays=dict(initial_xyz=xyz,initial_attributes=attr,frame_ids=np.arange(6),
                    predicted_xyz=np.stack([xyz+.001*h for h in range(5)]),
                    predicted_log_scales=np.repeat(attr[None,:,:3],5,axis=0),
                    predicted_quaternions=np.repeat(attr[None,:,3:7],5,axis=0))
        for h in range(1,6):
            arrays[f"target_xyz_{h}"]=xyz[:2]
            arrays[f"target_attr_{h}"]=attr[:2]
        data=geometry_data(arrays,limit=2)
        self.assertEqual(len(data["initial"]["centers"]),3)
        self.assertEqual(data["initial"]["ids"],[0,2])
        self.assertEqual([f["ids"] for f in data["prediction"]],[[0,2]]*5)
        self.assertEqual(len(data["target"][0]["centers"]),2)
        page=geometry_html(data,"test <window>")
        self.assertIn("test &lt;window&gt;",page)
        self.assertIn('id="c1"',page);self.assertIn('id="e1"',page)
        self.assertIn("D.prediction.slice",page)
        self.assertNotIn("https://",page)
        if shutil.which("node"):
            script=re.search(r"<script>(.*?)</script>",page,re.S).group(1)
            result=subprocess.run(["node","--check"],input=script,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            dom='''const devicePixelRatio=1,window={},elements={};
const context={beginPath(){},stroke(){},fillRect(...a){if(a.some(v=>!Number.isFinite(v)))throw Error('invalid rectangle')},moveTo(...a){if(a.some(v=>!Number.isFinite(v)))throw Error('invalid point')},lineTo(...a){if(a.some(v=>!Number.isFinite(v)))throw Error('invalid point')}};
const document={getElementById(id){return elements[id]??=( {value:id==='view'?'orbit':0,checked:true,getBoundingClientRect(){return {width:400,height:250}},getContext(){return context},setPointerCapture(){}} )},querySelectorAll(){return ['c0','c1','c2','e0','e1','e2'].map(id=>document.getElementById(id))}};
'''
            exercise="for(let h=0;h<=5;h++){elements.frame.value=h;for(let view of ['orbit','xy','xz','yz']){elements.view.value=view;elements.frame.oninput()}}"
            result=subprocess.run(["node"],input=dom+script+exercise,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__ == "__main__":
    unittest.main()
