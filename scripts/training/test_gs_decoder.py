"""Disk-free P1a tests: python -B -m unittest scripts.training.test_gs_decoder -v."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from ..data_process import DataProcessor
from ..data_process.test_data_process import make_batch, add_depth_targets
from .gs_decoder import (build_initial_anchor_field, pool_anchor_features,
                         interpolate_anchor_displacements)
from .model import PositionPredictor
from .losses import objective, nearest_distances
from .rendering import GaussianRenderer
from .test_training import fake_gaussian_rasterizer
from .evaluate import load_model
from .train import (VERSION, GS_JOINT_VERSION, DECODER_DEFAULTS, checkpoint_state,
                    checkpoint_config, restore_training, resolve_decoder,
                    recover_best_score, publish_checkpoints, loss_options)


def model_config(**changes):
    return dict(dict(width=32, heads=4, layers=2, controller_features=7,
                     prediction_mode="autoregressive", output_mode="gs_geometry",
                     gs_local_attention="knn", gs_local_k=16, gs_decoder="anchors",
                     anchor_count=128, anchor_pool_k=16, anchor_bandwidth=1.), **changes)


class AnchorDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.processor = DataProcessor("001-rope")
        self.batch = self.processor(add_depth_targets(make_batch()))
        self.model = PositionPredictor(**model_config())

    def activate(self):
        torch.nn.init.normal_(self.model.position_head[-1].weight, std=.02)

    def checkpoint(self, model=None):
        model = model or self.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        config = dict(model.config, scene="001-rope", depth_weight=0., chamfer_chunk=2,
                      rgb_weight=1., alpha_weight=.1, scale_weight=.01, chamfer_weight=1.)
        return checkpoint_state(model, optimizer, self.processor, config, 2, .4, [])

    def test_fps_determinism_counts_duplicates_and_padding(self):
        points = torch.tensor([[[0.,0,0], [1.,0,0], [2.,0,0], [3.,0,0], [float("nan"),0,0]],
                               [[1.,1,1], [1.,1,1], [1.,1,1], [float("nan"),0,0], [0.,0,0]],
                               [[0.,0,0]]*5, [[float("nan"),0,0]]*5])
        keep = torch.tensor([[1,1,1,1,0], [1,1,1,0,0], [1,0,0,0,0], [0,0,0,0,0]], dtype=torch.bool)
        field = build_initial_anchor_field(points, keep)
        again = build_initial_anchor_field(points, keep)
        self.assertEqual(field.anchor_mask.sum(-1).tolist(), [4,3,1,0])
        self.assertEqual(field.anchor_indices[0,:4].tolist(), [0,3,1,2])
        self.assertEqual(field.anchor_indices[1,:3].tolist(), [0,1,2])
        for key in field.__dataclass_fields__:
            torch.testing.assert_close(getattr(field,key), getattr(again,key), rtol=0, atol=0)
        for b in range(4):
            selected = field.anchor_indices[b,field.anchor_mask[b]]
            self.assertEqual(selected.unique().numel(), len(selected))
            self.assertTrue(keep[b,selected].all())
            self.assertTrue(keep[b,field.pool_indices[b][field.pool_mask[b]]].all())
        self.assertTrue(torch.isfinite(field.weights).all())
        self.assertTrue(torch.all(field.bandwidth > 0))
        torch.testing.assert_close(field.weights.sum(-1), keep.float())
        self.assertTrue(torch.all(field.weights[~keep] == 0))

    def test_rbf_constant_translation_partition_unity_and_dense_support(self):
        points = torch.tensor([[[0.,0,0], [.01,.01,0], [.02,0,0], [.03,.01,0]]], dtype=torch.float64)
        keep = torch.ones(1,4,dtype=torch.bool)
        field = build_initial_anchor_field(points,keep,anchor_count=3,pool_k=2)
        self.assertTrue(torch.all(field.weights > 0))
        torch.testing.assert_close(field.weights.sum(-1),torch.ones(1,4,dtype=torch.float64),atol=1e-15,rtol=0)
        translation = torch.tensor([.01,-.02,.03],dtype=torch.float64)
        displacement = translation.expand(1,3,3).clone().requires_grad_()
        output = interpolate_anchor_displacements(displacement,field,keep)
        torch.testing.assert_close(output,translation.expand_as(output),atol=1e-15,rtol=0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(displacement.grad).all())
        self.assertGreater(displacement.grad.abs().sum().item(),0)
        wider = build_initial_anchor_field(points,keep,3,2,2.)
        torch.testing.assert_close(wider.bandwidth,field.bandwidth*2)
        self.assertFalse(torch.equal(wider.weights,field.weights))

    def test_anchor_cap_and_point_reordering_preserve_a_generic_field(self):
        xyz=torch.randn(1,143,3)
        keep=torch.ones(1,143,dtype=torch.bool)
        field=build_initial_anchor_field(xyz,keep,128,16)
        self.assertEqual(field.anchor_mask.sum().item(),128)
        self.assertEqual(field.anchor_indices.unique().numel(),128)
        self.assertEqual(field.pool_mask.sum(-1).unique().tolist(),[16])
        order=torch.randperm(143)
        reordered=build_initial_anchor_field(xyz[:,order],keep[:,order],128,16)
        # Random points avoid distance ties: FPS and pools must depend on geometry,
        # rather than the unrelated ordering of independently reconstructed GS.
        torch.testing.assert_close(xyz[0,field.anchor_indices[0]],
                                   xyz[0,order[reordered.anchor_indices[0]]],atol=0,rtol=0)
        torch.testing.assert_close(field.weights[:,order],reordered.weights,atol=1e-7,rtol=1e-6)

    def test_pool_mean_excludes_padding_and_invalid_anchor_values(self):
        xyz = torch.tensor([[[0.,0,0],[1.,0,0],[2.,0,0],[float("nan"),0,0]]])
        keep = torch.tensor([[1,1,1,0]],dtype=torch.bool)
        field = build_initial_anchor_field(xyz,keep,anchor_count=4,pool_k=2)
        features = torch.tensor([[[1.,2.],[3.,4.],[9.,10.],[float("nan"),float("nan")]]],requires_grad=True)
        pooled = pool_anchor_features(features,field)
        for i in range(3):
            expected=features[0,field.pool_indices[0,i,field.pool_mask[0,i]]].mean(0)
            torch.testing.assert_close(pooled[0,i],expected)
        self.assertTrue(torch.equal(pooled[0,3],torch.zeros(2)))
        pooled.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.all(features.grad[~keep] == 0))
        d = torch.ones(1,4,3); d[~field.anchor_mask]=float("nan")
        out=interpolate_anchor_displacements(d,field,keep)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out[keep],torch.ones_like(out[keep]))
        self.assertTrue(torch.all(out[~keep] == 0))

    def test_empty_geometry_and_invalid_parameters(self):
        xyz=torch.empty(2,0,3);keep=torch.empty(2,0,dtype=torch.bool)
        field=build_initial_anchor_field(xyz,keep)
        self.assertEqual(tuple(field.weights.shape),(2,0,0))
        self.assertEqual(tuple(pool_anchor_features(torch.empty(2,0,8),field).shape),(2,0,8))
        self.assertEqual(tuple(interpolate_anchor_displacements(torch.empty(2,0,3),field,keep).shape),(2,0,3))
        xyz=torch.zeros(1,2,3);keep=torch.ones(1,2,dtype=torch.bool)
        for kwargs in (dict(anchor_count=0),dict(anchor_count=True),dict(pool_k=0),
                       dict(bandwidth_factor=0),dict(bandwidth_factor=float("nan"))):
            with self.assertRaises(ValueError):build_initial_anchor_field(xyz,keep,**kwargs)
        xyz[0,0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            build_initial_anchor_field(xyz,keep)

    def test_paired_initial_shared_weights_and_no_free_shape_heads(self):
        torch.manual_seed(42); free=PositionPredictor(**model_config(gs_decoder="free"))
        torch.manual_seed(42); anchors=PositionPredictor(**model_config())
        self.assertFalse(hasattr(anchors,"scale_head"))
        self.assertFalse(hasattr(anchors,"rotation_head"))
        self.assertEqual(set(free.state_dict())-set(anchors.state_dict()),
                         {key for key in free.state_dict() if key.startswith(("scale_head.","rotation_head."))})
        for key,value in anchors.state_dict().items():
            torch.testing.assert_close(value,free.state_dict()[key],atol=0,rtol=0)

    def test_zero_and_constant_head_are_initial_relative_not_accumulated(self):
        x=self.batch["inputs"];keep=x["gs_mask"][:,None].expand(-1,5,-1)
        prediction=self.model(x)
        initial=x["gs_xyz"][:,None].expand_as(prediction["xyz"])
        torch.testing.assert_close(prediction["xyz"],initial,atol=0,rtol=0)
        d=torch.tensor([.02,-.03,.01])
        with torch.no_grad():self.model.position_head[-1].bias.copy_(d)
        prediction=self.model(x)
        torch.testing.assert_close(prediction["xyz"][keep],(initial+d)[keep],atol=1e-7,rtol=1e-6)
        self.assertTrue(torch.all(prediction["xyz"][~keep] == 0))
        for key,part in (("log_scales",slice(0,3)),("quaternions",slice(3,7)),
                         ("opacity_logits",slice(7,8)),("sh_dc",slice(8,11))):
            inherited=x["gs_attr"][...,part][:,None].expand_as(prediction[key])
            torch.testing.assert_close(prediction[key],inherited,atol=0,rtol=0)
        # Same scales/quaternions imply exactly the same Gaussian covariances.
        self.assertTrue(torch.equal(prediction["log_scales"][:,0],prediction["log_scales"][:,-1]))
        self.assertTrue(torch.equal(prediction["quaternions"][:,0],prediction["quaternions"][:,-1]))

    def test_fixed_initial_field_five_steps_live_reencoded_centers(self):
        self.activate(); x=self.batch["inputs"];fields=[];encoded=[]
        real_interpolate=interpolate_anchor_displacements
        def inspect(delta,field,keep):
            fields.append(field)
            return real_interpolate(delta,field,keep)
        hook=self.model.gs_encoder.register_forward_pre_hook(lambda module,args:encoded.append(args[0]))
        try:
            with patch("scripts.training.model.build_initial_anchor_field",wraps=build_initial_anchor_field) as build, \
                 patch("scripts.training.model.interpolate_anchor_displacements",side_effect=inspect):
                prediction=self.model(x)
            self.assertEqual(build.call_count,1)
        finally:hook.remove()
        self.assertEqual(len(fields),5)
        self.assertTrue(all(field is fields[0] for field in fields))
        torch.testing.assert_close(build.call_args.args[0],x["gs_xyz"],atol=0,rtol=0)
        for k,state in enumerate(encoded):
            torch.testing.assert_close(state[...,:3],x["gs_xyz"] if k==0 else prediction["xyz"][:,k-1],atol=0,rtol=0)
            torch.testing.assert_close(state[...,3:],x["gs_attr"],atol=0,rtol=0)

    def test_five_step_causality_and_invalid_nan_masks(self):
        self.activate();x=self.batch["inputs"];original=self.model(x)
        for k in range(4):
            other=copy.deepcopy(x)
            for key in ("action_state","controller_points","tactile"):
                other[key][:,k+1:]+=torch.randn_like(other[key][:,k+1:])*20
            for key in ("action_valid","controller_mask","valid_mask","tactile_spatial_mask"):
                other[key][:,k+1:]=~other[key][:,k+1:]
            other["tactile_age_us"][:,k+1:]+=10000
            result=self.model(other)
            for key in result:torch.testing.assert_close(result[key][:,:k+1],original[key][:,:k+1],atol=0,rtol=0)
        other=copy.deepcopy(x)
        for key,mask in (("gs_xyz","gs_mask"),("gs_attr","gs_mask"),("action_state","action_valid"),
                         ("controller_points","controller_mask"),("tactile","valid_mask")):
            other[key][~other[mask]]=float("nan")
        result=self.model(other)
        for key in result:torch.testing.assert_close(result[key],original[key],atol=0,rtol=0)
        result["xyz"].square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_all_missing_sample_remains_finite_and_does_not_mix_batch_geometry(self):
        self.activate(); x=copy.deepcopy(self.batch["inputs"])
        before=self.model(x)
        x["gs_mask"][1].zero_()
        x["gs_xyz"][1]=float("nan");x["gs_attr"][1]=float("nan")
        for key in ("action_valid","controller_mask","valid_mask","tactile_spatial_mask"):
            x[key][1].zero_()
        after=self.model(x)
        for key in after:
            self.assertTrue(torch.isfinite(after[key]).all())
            self.assertTrue(torch.all(after[key][1]==0))
            torch.testing.assert_close(after[key][0],before[key][0],atol=0,rtol=0)

    def test_fifth_chamfer_reaches_every_previous_anchor_head_and_condition(self):
        self.activate();x=copy.deepcopy(self.batch["inputs"])
        x["action_state"].requires_grad_();x["controller_points"].requires_grad_()
        deltas=[];encoded=[]
        hooks=[self.model.position_head.register_forward_hook(lambda module,args,out:deltas.append(out)),
               self.model.gs_encoder.register_forward_pre_hook(lambda module,args:encoded.append(args[0]))]
        try:prediction=self.model(x)
        finally:
            for hook in hooks:hook.remove()
        points=prediction["xyz"][0,-1,x["gs_mask"][0]]
        target=self.batch["targets"]["target_xyz"][-1][0,self.batch["targets"]["target_mask"][-1][0]]
        loss=(nearest_distances(points,target).mean()+nearest_distances(target,points).mean())*.5
        gradients=torch.autograd.grad(loss,deltas+encoded[1:],retain_graph=True,allow_unused=True)
        self.assertEqual(len(deltas),5)
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)
        loss.backward()
        for key in ("action_state","controller_points"):
            self.assertTrue(torch.isfinite(x[key].grad).all())
            self.assertGreater(x[key].grad[:,:1].abs().sum().item(),0)

    def test_joint_loss_has_zero_scale_regularizer_and_finite_anchor_gradients(self):
        self.activate()
        prediction=self.model(self.batch["inputs"])
        loss,metrics=objective(prediction,self.batch,self.processor,GaussianRenderer(fake_gaussian_rasterizer),
                               0.,2,rgb_weight=1.,alpha_weight=.1,scale_weight=.01,chamfer_weight=1.)
        self.assertTrue(torch.all(metrics["scale_regularization"]==0))
        loss.backward()
        self.assertGreater(self.model.position_head[-1].weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_v4_checkpoint_exact_roundtrip_and_architecture_conflicts(self):
        self.activate();state=copy.deepcopy(self.checkpoint())
        self.assertEqual(state["version"],VERSION)
        with patch("scripts.training.train.torch.load",return_value=state):
            loaded,_,config=load_model("memory/anchors.pt","cpu")
        before=self.model(self.batch["inputs"]);after=loaded(self.batch["inputs"])
        for key in before:torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
        optimizer=torch.optim.AdamW(loaded.parameters(),lr=1e-4)
        restore_training(state,loaded,optimizer)
        for change in (dict(gs_decoder="free"),dict(anchor_count=64),dict(anchor_pool_k=8),dict(anchor_bandwidth=2.)):
            other=PositionPredictor(**model_config(**change))
            snapshot=copy.deepcopy(other.state_dict())
            with self.assertRaises(ValueError):restore_training(state,other,torch.optim.AdamW(other.parameters()))
            for key in snapshot:torch.testing.assert_close(snapshot[key],other.state_dict()[key],atol=0,rtol=0)
            args={key:None for key in DECODER_DEFAULTS};args.update(change)
            with self.assertRaises(ValueError):
                resolve_decoder(args["gs_decoder"],args["anchor_count"],args["anchor_pool_k"],args["anchor_bandwidth"],state)
        for key in DECODER_DEFAULTS:
            missing=copy.deepcopy(state);del missing["config"][key]
            with self.assertRaises(ValueError):checkpoint_config(missing)
        forged=copy.deepcopy(state);forged["version"]=GS_JOINT_VERSION
        with self.assertRaises(ValueError):checkpoint_config(forged)
        forged=copy.deepcopy(state);forged["config"]["gs_decoder"]="free"
        with self.assertRaises(ValueError):checkpoint_config(forged)
        forged=self.checkpoint(PositionPredictor(**model_config(gs_decoder="free")))
        forged["config"]["gs_decoder"]="anchors"
        with self.assertRaises(ValueError):checkpoint_config(forged)

    def test_old_v3_free_checkpoint_keeps_joint_loss_optimizer_and_best(self):
        free=PositionPredictor(**model_config(gs_decoder="free"))
        for head in (free.position_head,free.scale_head,free.rotation_head):
            torch.nn.init.normal_(head[-1].weight,std=.02)
        original_optimizer=torch.optim.AdamW(free.parameters(),lr=1e-4)
        def update(model,optimizer):
            optimizer.zero_grad(set_to_none=True)
            model(self.batch["inputs"])["xyz"].square().mean().backward()
            optimizer.step()
        update(free,original_optimizer)
        state=copy.deepcopy(checkpoint_state(free,original_optimizer,self.processor,
                            self.checkpoint(free)["config"],2,.4,[]))
        state["version"]=GS_JOINT_VERSION
        for key in DECODER_DEFAULTS:del state["config"][key]
        with patch("scripts.training.train.torch.load",return_value=state):
            loaded,_,config=load_model("memory/v3.pt","cpu")
        self.assertEqual(config["gs_decoder"],"free")
        self.assertEqual(config["chamfer_weight"],1.)
        expected=free(self.batch["inputs"]);actual=loaded(self.batch["inputs"])
        for key in expected:torch.testing.assert_close(expected[key],actual[key],atol=0,rtol=0)
        renderer=GaussianRenderer(fake_gaussian_rasterizer)
        l0,_=objective(expected,self.batch,self.processor,renderer,0.,2,**loss_options(config))
        l1,_=objective(actual,self.batch,self.processor,renderer,0.,2,**loss_options(config))
        torch.testing.assert_close(l0,l1,atol=0,rtol=0)
        opt=torch.optim.AdamW(loaded.parameters(),lr=1e-4);restore_training(state,loaded,opt)
        update(free,original_optimizer);update(loaded,opt)
        for name,value in free.state_dict().items():
            torch.testing.assert_close(value,loaded.state_dict()[name],atol=0,rtol=0)
        resaved=checkpoint_state(loaded,opt,self.processor,config,3,.6,[])
        canonical=copy.deepcopy(state);canonical["config"]=checkpoint_config(state)
        with patch("pathlib.Path.is_file",return_value=True),patch("scripts.training.train.read_checkpoint",return_value=canonical):
            best=recover_best_score(resaved,"memory")
        self.assertEqual(best,.4)
        with patch("scripts.training.train.atomic_checkpoint") as save:
            publish_checkpoints(Path("memory"),dict(resaved,best=best),.5 < best)
        self.assertEqual([call.args[0].name for call in save.call_args_list],["last.pt"])


if __name__ == "__main__":
    unittest.main()
