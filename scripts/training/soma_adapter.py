"""Frozen SoMA E10 inference adapter. No training or fitted statistics.

The rollout/parity functions below are moved verbatim from the accepted comparison
implementation. Their fixed scene/horizon are part of the current demo protocol.
Third-party code and runtime assets are supplied explicitly, outside this package.
"""
import ast
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

scene = "config_a_2cam"
horizon = 15


def require(value, message):
    if not value:
        raise ValueError(message)


def pairs_from_centers(centers, controller_levels, object_levels):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import NearestCentroid
    labels_by_level = []
    for threshold in object_levels:
        labels = AgglomerativeClustering(n_clusters=None, linkage="complete",
                                         distance_threshold=threshold).fit_predict(centers)
        labels_by_level.append(labels)
        centers = NearestCentroid().fit(centers, labels).centroids_
    labels_by_level.append(np.zeros(len(centers), dtype=np.int64))
    require(len(controller_levels) == len(labels_by_level), "Invalid hierarchy depth")
    pairs = [(ctl, obj + ctl.max() + 1) for ctl, obj in zip(controller_levels, labels_by_level)]
    return pairs + [(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))]

def prepare_soma_rollout(model, initial, covariance, pairs, actions, gravity, volume):
    p2c = [(torch.as_tensor(c, device="cuda"), torch.as_tensor(o, device="cuda")) for c, o in pairs]
    ctl_count = actions.shape[1]
    pin = initial["xyz"].new_zeros((ctl_count + len(initial["xyz"]), 1))
    pin[:ctl_count] = 1
    # Only scene opacity is read after decoding. The render callback below leaves
    # native motion and covariance computation intact. No frame labels are passed in.
    gaussian = SimpleNamespace(get_opacity=initial["opacity_logits"].sigmoid())
    attr = model.scene_attr[model._scene_attr_dict[scene]].expand(len(initial["xyz"]), -1)
    diag_volume = (initial["log_scales"].exp() * volume).prod(-1, keepdim=True)
    return p2c, pin, gaussian, attr, diag_volume

def soma_predict(model, initial, covariance, pairs, actions, gravity, volume):
    p2c, pin, gaussian, attr, diag_volume = prepare_soma_rollout(
        model, initial, covariance, pairs, actions, gravity, volume)
    current = previous = rest = initial["xyz"]
    rest_cov = covariance
    anchor = actions[0]
    positions, covariances = [], []
    for k in range(horizon):
        graph, connections = model._preprocess(
            previous, current, rest, actions[k], actions[k + 1], anchor,
            attr, diag_volume, rest_cov, gravity, p2c_mapping=p2c, pin_mask=pin)
        xyz, cov, *_ = model.encode_decode(
            graph, connections, gaussian, [None], rest_cov.clone(), rest.clone(),
            actions.shape[1], register_norm=False, gt_label=None, cln_gaussian=gaussian)
        predicted, pred_cov = xyz[actions.shape[1]:], cov[actions.shape[1]:]
        positions.append(predicted)
        covariances.append(pred_cov)
        previous, current = current, predicted
        if (k + 1) % model.frame_gap == 0:
            # Native continuous inference refreshes from its OWN prediction and
            # sets same-boundary previous/current equal. No data read occurs here.
            rest, rest_cov, previous, anchor = current, pred_cov, current, actions[k + 1]
    xyz, cov = torch.stack(positions), torch.stack(covariances)
    require(torch.isfinite(xyz).all().item() and torch.isfinite(cov).all().item(),
            "Nonfinite model rollout")
    return xyz, cov

def native_parity(model, canonical, covariance, pairs, actions, gravity, volume, candidate):
    initial = canonical["xyz"]
    model.scene_init_pos = {scene + "_frame_0": initial}
    model.scene_init_prev_pos = {scene + "_frame_0": initial}
    model.scene_init_cov = {scene + "_frame_0": covariance}
    p2c = [(torch.as_tensor(c, device="cuda")[None], torch.as_tensor(o, device="cuda")[None])
           for c, o in pairs]
    inputs = {
        "img": [initial[:1]], "cam": [SimpleNamespace(to_device=lambda *a: None)],
        "controller_trajectory": actions[None],
        "external": gravity[None], "volume_scalar": initial.new_tensor([[volume]]),
        "p2c_mapping": p2c, "gs_aligned_frame": 0,
    }
    zeros = initial.new_zeros((1, horizon+1, 3, 1, 1))
    model.encode_decode_render_only = lambda *a, **k: []
    model._encode_decode_test = lambda *a, **k: {"acc": {}}
    previous = current = None
    for k in range(horizon):
        result = model.simple_test(
            inputs, [zeros], prev_state=previous, cur_state=current, pred_frame_idx=k,
            meta={"seq_idx": torch.zeros((1, 1), dtype=torch.long), "scene_name": scene})
        torch.testing.assert_close(torch.as_tensor(result["pred_pos"], device="cuda"), candidate[0][k])
        torch.testing.assert_close(torch.as_tensor(result["pred_cov"], device="cuda"), candidate[1][k])
        previous = torch.as_tensor(result["cur_state"], device="cuda")
        current = torch.as_tensor(result["pred_pos"], device="cuda")
    model.scene_init_pos.clear()
    model.scene_init_prev_pos.clear()
    model.scene_init_cov.clear()
    del model.encode_decode_render_only
    del model._encode_decode_test

def load_pkl(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def controller_partition():
    """Original dis_split partition of the ordered 30 right-hand control points."""
    return [np.repeat(np.arange(10), 3), np.repeat(np.arange(2), 5), np.zeros(2)]


def configuration_files(config):
    """List the frozen local mmcv config and its literal _base_ file closure."""
    found, visiting = [], set()

    def visit(path):
        path = Path(path).resolve()
        require(path not in visiting, 'Cyclic SoMA configuration inheritance')
        if path in found:
            return
        require(path.is_file(), f'Missing SoMA configuration asset: {path}')
        visiting.add(path)
        module = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in module.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_base_' for t in node.targets):
                bases = ast.literal_eval(node.value)
                if isinstance(bases, str):
                    bases = [bases]
                require(isinstance(bases, (list, tuple)) and all(isinstance(b, str) for b in bases),
                        'Expected local literal SoMA config inheritance')
                for base in bases:
                    visit(path.parent/base)
        visiting.remove(path)
        found.append(path)

    visit(config)
    return tuple(found)


def relocate_runtime_config(cfg, package, geometry):
    """Set paths/mode before constructing SoMA; do not change model math.

    The native Stage-2 continuous constructor initializes from its observed PLY
    and returns before the segmented-mode pred_stage1 cache loader. Both this
    mode and the relocated paths are explicit so deleting historical runs does
    not leave a dormant dependency on their Stage-1 predictions.
    """
    package, geometry = Path(package).resolve(), Path(geometry).resolve()
    cfg.model.test_rollout_mode = 'continuous'
    cfg.model.data_dir = str(package.parent)
    require(len(cfg.model.gs_scene) == 1 and cfg.model.gs_scene[0]['name'] == scene,
            'Expected the single original SoMA scene template')
    cfg.model.gs_scene[0]['model_path'] = str(geometry)

    def field(obj, name):
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    relocated = []
    for split in ('train', 'val', 'test'):
        node = field(cfg.data, split)
        # The original training config wraps EmbodiedDataset in RepeatDataset.
        while node is not None:
            env = field(node, 'env_cfg')
            if env is not None:
                if isinstance(env, dict):
                    env['data_dir'] = str(package.parent)
                else:
                    env.data_dir = str(package.parent)
                relocated.append(split)
                break
            node = field(node, 'dataset')
    require('test' in relocated, 'Missing native SoMA test environment configuration')
    return dict(test_rollout_mode='continuous', stage1_prediction_cache_required=False,
                model_data_dir=str(package.parent), scene_geometry=str(geometry),
                dataset_root=str(package.parent), relocated_dataset_splits=relocated)


class SomaAdapter:
    """Load the supplied SoMA source/config/package and frozen E10 checkpoint.

    Source/config are the recorded SoMA revision, not a reimplementation. Paths
    are replaced to relocate assets; the original time, gravity and model
    configuration must pass the assertions before execution.
    """

    def __init__(self, *, source, config, package, geometry, checkpoint,
                 canonical, canonical_controller, quat_scale_to_covar_preci):
        source, package, geometry = map(Path, (source, package, geometry))
        for directory in (source, source / "gaussian-splatting"):
            sys.path.insert(0, str(directory))
        from mmcv import Config
        from mmgs.models import build_simulator
        from mmgs.utils.transformation_utils import get_mat_from_upper

        cfg = Config.fromfile(str(config))
        self.runtime_overrides = relocate_runtime_config(cfg, package, geometry)
        require(cfg.model.frame_gap == 10 and not cfg.model.use_rotation,
                "SoMA must retain the native H10 refresh and disabled rotation")
        require(not cfg.model.flag_save_gaussian and not cfg.model.flag_update_gaussian
                and not cfg.model.flag_update_gaussian_train,
                "Inference must not update or save SoMA Gaussian training state")
        env = cfg.data.test.env_cfg
        require(env.rot_est[scene] == [0., 0., 0., 1.], "Unexpected SoMA world rotation")
        require(abs(float(env.real_dt[scene]) - 1/15) < 1e-12
                and abs(float(env.dt) - 1/30) < 1e-12,
                "Original SoMA timing must remain explicit; no recalibration")
        iface = __import__('json').loads((package / 'scene_interface.json').read_text())
        require(iface['initial_gaussian']['source_frame'] == 113
                and iface['score_local_half_open'] == [155, 194],
                "SoMA package is not the original Ep0 interface")
        trajectory = load_pkl(package / env.tracking_pcd)['controller_points']
        np.testing.assert_array_equal(trajectory, canonical_controller)
        require(trajectory.shape == (194, 30, 3), "Expected original 194-frame control track")
        control_cache, = (package / 'controller_mask/dis_split').glob('controller_dis_split*.pkl')
        object_cache, = (package / 'cluster_mask/dis_split').glob('dis_split*.pkl')
        self.controller_levels = load_pkl(control_cache)['p2c']
        require(len(self.controller_levels) == 3, 'Unexpected controller hierarchy depth')
        for actual, expected in zip(self.controller_levels, controller_partition()):
            np.testing.assert_array_equal(actual, expected)
        self.thresholds = [level['downsample_rate'] for level in env.cluster_cfg[scene]]
        self.model = build_simulator(cfg.model).cuda().eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(state['meta']['epoch'] == 10 and state['meta']['iter'] == 6000,
                'SoMA checkpoint is not Stage 2 E10 / 6000 updates')
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.initial_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.trajectory = torch.as_tensor(trajectory, device='cuda', dtype=torch.float32)
        self.model.decode_head.pre_render = lambda *a, **k: (self.trajectory[:1, :1, :1], self.trajectory[:1, :1, :1])
        self.gravity = self.trajectory.new_tensor(env.gravity).reshape(1, 3) * (env.real_dt[scene] / env.dt)**2
        self.volume = env.volume_scalar[scene]
        self.get_mat_from_upper = get_mat_from_upper
        self.quat_scale_to_covar_preci = quat_scale_to_covar_preci
        self.dependencies = [*configuration_files(config), geometry, package/'scene_interface.json',
                             package/env.tracking_pcd, control_cache, object_cache]
        # Native, observed starting geometry and hierarchy must agree before any
        # cross-episode predictions are compared.
        template = self.model.gs_scene_dict[scene]
        torch.testing.assert_close(template.get_xyz, canonical['xyz'], rtol=0, atol=0)
        saved = [(c, o+c.max()+1) for c, o in zip(self.controller_levels, load_pkl(object_cache)['p2c'])]
        saved.append((np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)))
        rebuilt = pairs_from_centers(canonical['xyz'].cpu().numpy(), self.controller_levels, self.thresholds)
        for actual, expected in zip(rebuilt, saved):
            for x, y in zip(actual, expected):
                np.testing.assert_array_equal(x, y)
        cov = template.get_covariance()
        from_v1, _ = quat_scale_to_covar_preci(canonical['quaternions'], canonical['log_scales'].exp(),
                                             compute_preci=False, triu=True)
        torch.testing.assert_close(cov, from_v1, atol=1e-9, rtol=2e-5)
        initial = dict(canonical, xyz=template.get_xyz, log_scales=template.get_scaling.log())
        candidate = soma_predict(self.model, initial, cov, saved, self.trajectory[:16], self.gravity, self.volume)
        native_parity(self.model, initial, cov, saved, self.trajectory[:16], self.gravity, self.volume, candidate)

    def predict(self, initial, actions):
        """Initial GT only; future object geometry never enters model state."""
        covariance, _ = self.quat_scale_to_covar_preci(
            initial['quaternions'], initial['log_scales'].exp(), compute_preci=False, triu=True)
        pairs = pairs_from_centers(initial['xyz'].cpu().numpy(), self.controller_levels, self.thresholds)
        result = soma_predict(self.model, initial, covariance, pairs, actions, self.gravity, self.volume)
        template = self.model.gs_scene_dict[scene]
        self.model.gs_scene_dict[scene] = SimpleNamespace(
            get_scaling=initial['log_scales'].exp(), get_opacity=initial['opacity_logits'].sigmoid(),
            get_xyz=initial['xyz'])
        try:
            native_parity(self.model, initial, covariance, pairs, actions, self.gravity, self.volume, result)
        finally:
            self.model.gs_scene_dict[scene] = template
        positions, covariance = result
        matrices = self.get_mat_from_upper(covariance.flatten(0, 1)).reshape(horizon, -1, 3, 3)
        return positions, covariance, matrices

    def verify_unchanged(self):
        actual = self.model.state_dict()
        require(actual.keys() == self.initial_state.keys(), 'SoMA state keys changed')
        require(all(torch.equal(v.detach().cpu(), self.initial_state[k]) for k, v in actual.items()),
                'SoMA weights or normalizers changed during frozen inference')
