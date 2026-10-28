import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.geometry import hand_slots,action_conditions
from scripts.rebuild_dataset.rebuild import controller_arrays
from scripts.training.v4_conditions import V4ConditionEncoder
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_signed_motion.core import deltas,tool_aggregation,cosine,frame_contract


class SignedMotionTests(unittest.TestCase):
    def test_coordinate_roundtrip_and_signed_displacement(self):
        p=DataProcessor('001-rope');mask=torch.tensor([[True,False]])
        xyz=torch.tensor([[[.2,-.01,.3],[0.,0.,0.]]]);v=torch.tensor([[[.001,-.002,.003],[0.,0.,0.]]])
        n=p._xyz(xyz,mask);n2=p._xyz(xyz+v,mask)
        torch.testing.assert_close(p.denormalize_xyz(n,mask),xyz)
        torch.testing.assert_close(p.denormalize_displacement(n2-n,mask),v,atol=3e-8,rtol=1e-5)
        self.assertEqual(torch.sign((n2-n)[0,0]).tolist(),[1.,-1.,1.])

    def test_adjacent_and_cumulative_are_distinct(self):
        cp=torch.zeros(3,60,3);cp[1,:,1]=1;cp[2,:,1]=-1;m=torch.ones(3,60,dtype=torch.bool)
        cu,cv,ad,am=deltas(cp,m,2)
        self.assertTrue((cu[:,1]==-1).all());self.assertTrue((ad[:,1]==-2).all())
        self.assertFalse(deltas(cp,m,0)[3].any())

    def test_invalid_endpoint_does_not_create_motion(self):
        cp=torch.randn(3,60,3);m=torch.ones(3,60,dtype=torch.bool);m[0,0]=False;m[1,1]=False
        cu,cv,ad,am=deltas(cp,m,2)
        self.assertFalse(cv[0]);self.assertEqual(cu[0].abs().sum(),0)
        self.assertFalse(am[1]);self.assertEqual(ad[1].abs().sum(),0)

    def test_four_heads_null_and_signed_aggregation(self):
        a=torch.zeros(2,4,61);a[:,:,60]=1;a[:,0,60]=.25;a[:,0,31]=.75
        d=torch.zeros(60,3);d[31]=torch.tensor([1.,-2.,3.]);out=tool_aggregation(a,d)
        torch.testing.assert_close(out[:,0],d[31][None].expand(2,3)*.75)
        self.assertEqual(out[:,1:].abs().sum(),0)

    def test_time_protocol_and_shift_rejection(self):
        c,t=frame_contract(np.arange(135,151),135,15,1)
        self.assertEqual(c[-1],149);self.assertEqual(t[-1],150)
        with self.assertRaises(ValueError):frame_contract(np.arange(136,152),135,15,1)

    def test_hand_feature_mask_and_future_locality(self):
        raw=synthetic_batch(points=16,horizon=5,prediction_interval=1)
        x=prepare_batch(raw,DataProcessor('001-rope'),SimpleNamespace(config={'horizon':5,'prediction_interval':1}))['inputs']
        encoder=V4ConditionEncoder(5).eval()
        with torch.no_grad():
            before=encoder.features(x,1)[2];alter=copy.deepcopy(x)
            alter['controller_points'][0,:,0]=100;alter['action_state'][0,:,0]=100;alter['tactile'][0,:,:2]=100
            alter['controller_points'][:,2:]+=10
            after=encoder.features(alter,1)[2]
        torch.testing.assert_close(before,after)
        self.assertTrue((after[0,:30]==0).all())

    def test_slot_order_and_rigid_world_translation(self):
        meta={'hand_order':['left','right'],'stream_order':['tactilel_left','tactilel_right','tactiler_left','tactiler_right'],
              'robot_slot_hands':['right','left'],'physical_hand_to_robot_slot':{'left':1,'right':0}}
        self.assertEqual(hand_slots(meta),(1,0))
        t=np.tile(np.eye(4),(2,2,1,1));t[1,0,:3,3]=[.001,-.002,.003];opening=np.full((2,2),.08)
        points,valid,_=controller_arrays(t,opening,meta['robot_slot_hands'])
        # Two independently rounded float32 world positions: verify one-ULP position
        # agreement instead of subtracting them then assuming double-precision delta.
        expected=(points[0,1].astype(np.float64)+[.001,-.002,.003]).astype(np.float32)
        np.testing.assert_array_max_ulp(points[1,1],expected,maxulp=1)
        np.testing.assert_array_equal(np.sign(points[1,1]-points[0,1]),np.tile([1,-1,1],(30,1)))
        np.testing.assert_array_equal(points[1,0],points[0,0]);self.assertTrue(valid.all())

    def test_zero_direction_undefined_not_wrong(self):
        self.assertIsNone(cosine([0,0,0],[0,1,0]));self.assertEqual(cosine([0,-1,0],[0,1,0]),-1)


if __name__=='__main__':unittest.main()
