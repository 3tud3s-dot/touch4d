import copy
import unittest
import torch
from scripts.data_process import DataProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.training.v4_runtime import prepare_batch
from scripts.training.v4_model import V4Stage1Predictor
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from scripts.experiments.v4_g4_attribution.interventions import intervention,direct_positions,model_digest,effective_hands
from scripts.experiments.v4_g4_attribution.run import direct_effects


class FrozenAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.grad=torch.is_grad_enabled();torch.set_grad_enabled(False)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads);torch.set_grad_enabled(cls.grad)
    def setUp(self):
        torch.manual_seed(112)
        self.model=G4GlobalMotionPredictor(horizon=15).eval().requires_grad_(False)
        self.model.motion[-1].weight.normal_(std=.002)
        self.model.global_motion.mapping[-1].weight.normal_(std=.02)
        self.x=prepare_batch(synthetic_batch(27,15),DataProcessor('001-rope'),self.model)['inputs']

    def test_no_global_is_shared_retrained_weights_not_original_g0(self):
        shared=V4Stage1Predictor(horizon=15).eval().requires_grad_(False)
        shared.load_state_dict({k:v for k,v in self.model.state_dict().items() if not k.startswith('global_motion.')})
        with intervention(self.model,'NoGlobal'):pred=self.model(self.x)
        original=shared(self.x)
        for k in pred:torch.testing.assert_close(pred[k],original[k],rtol=0,atol=0)

    def test_missing_left_invariance_and_right_equals_all_off(self):
        normal=self.model(self.x)
        with intervention(self.model,'NoLeft'):left=self.model(self.x)
        with intervention(self.model,'NoRight'):right=self.model(self.x)
        with intervention(self.model,'NoGlobal'):neither=self.model(self.x)
        for k in normal:
            torch.testing.assert_close(normal[k][0],left[k][0],rtol=0,atol=0)
            torch.testing.assert_close(right[k][0],neither[k][0],rtol=0,atol=0)
        self.assertGreater(float((normal['xyz'][1]-left['xyz'][1]).abs().max()),0)

    def test_frozen_history_direct_effect_verified_with_actual_step_intervention(self):
        normal,trace=self.model(self.x,return_trace=True)
        rows,checks=direct_effects(self.model,self.x,normal,trace,.157,2,73,verify=True)
        self.assertEqual(len(rows),12);self.assertEqual(len(checks),12)
        self.assertTrue(all(r['on_minus_off_rms_mm']==0 for r in rows if r['horizon']==1))

    def test_full_ar_difference_not_assumed_equal_to_direct_shift(self):
        normal,trace=self.model(self.x,return_trace=True)
        with intervention(self.model,'NoGlobal'):off=self.model(self.x)
        direct=direct_positions(self.x,trace,14,'NoGlobal')
        self.assertGreater(float((direct-off['xyz'][:,14]).abs().max()),1e-10)
        torch.testing.assert_close(normal['xyz'][:,0],off['xyz'][:,0],rtol=0,atol=0)

    def test_hooks_restore_weights_and_inputs_even_on_exception(self):
        before=model_digest(self.model);inputs=copy.deepcopy(self.x)
        count=len(self.model.global_motion._forward_hooks)
        with self.assertRaises(RuntimeError):
            with intervention(self.model,'NoLeft'):raise RuntimeError('diagnostic error')
        self.assertEqual(len(self.model.global_motion._forward_hooks),count)
        for mode in ('Normal','NoGlobal','NoLeft','NoRight'):
            with intervention(self.model,mode):self.model(self.x)
        self.assertEqual(before,model_digest(self.model))
        for k,v in inputs.items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(v,self.x[k],rtol=0,atol=0,equal_nan=True)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))

    def test_output_only_hand_mask_does_not_average_or_flip(self):
        hands=torch.tensor([[[1.,-2.,3.],[-4.,5.,-6.]]])
        self.assertTrue(torch.equal(effective_hands(hands,'NoLeft').sum(1),hands[:,1]))
        self.assertTrue(torch.equal(effective_hands(hands,'NoRight').sum(1),hands[:,0]))
        self.assertTrue(torch.equal(effective_hands(hands,'NoGlobal'),torch.zeros_like(hands)))
        self.assertTrue(torch.equal(hands,torch.tensor([[[1.,-2.,3.],[-4.,5.,-6.]]])))

    def test_reporting_episode_equal_and_unbounded_retention(self):
        from scripts.experiments.v4_g4_attribution.report import retained,aggregate_curve
        self.assertEqual(retained(5,3,4),50)
        self.assertEqual(retained(5,3,6),-50)
        self.assertEqual(retained(5,3,2),150)
        self.assertIsNone(retained(3,3,4))
        rows=[]
        for h in range(1,16):
            rows.extend([dict(mode='Normal',episode=3,horizon=h,value=1)]*2)
            rows.append(dict(mode='Normal',episode=6,horizon=h,value=4))
        self.assertEqual(aggregate_curve(rows,'Normal','value'),[2.5]*15)


if __name__=='__main__':unittest.main()
