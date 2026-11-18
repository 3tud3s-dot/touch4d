"""Retained compatibility/world-geometry assertions extracted before legacy feature retirement."""
import copy
import pickle
import unittest
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.tests.data_process.test_data_process import make_batch, add_depth_targets
from scripts.training.cloth_geometry import ellipsoid_axes, geometry_data


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
    episode, start = window or {'001-rope': (1, 85), '051-cube': (0, 109)}[scene]
    raw["meta"]["episode_id"].fill_(episode)
    raw["meta"]["frame_ids"] = torch.arange(start,start+6)[None]
    return raw


class CoreGeometryCompatibilityTests(unittest.TestCase):
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

    def test_position_predictor_pickle_path_is_retained(self):
        from scripts.training.model import PositionPredictor
        restored = pickle.loads(b'cscripts.training.model\nPositionPredictor\n.')
        self.assertIs(restored, PositionPredictor)
        self.assertEqual(PositionPredictor.__module__, 'scripts.training.model')


if __name__ == '__main__':
    unittest.main()
