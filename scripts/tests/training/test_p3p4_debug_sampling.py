"""CPU-only sample coverage arithmetic and frozen-list safety checks."""
import copy
import unittest

import torch

from scripts.data_process import DataProcessor
from scripts.tests.training.test_fixed_fit import single_raw
from scripts.experiments.debug_audit.sampling_audit import frozen_rows, quantiles, window_statistics


class DebugSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_manifest_is_24_unique_training_rows_without_reselection(self):
        rows=[dict(episode=e,start=s,frame_ids=list(range(s,s+6))) for e in (1,2,4,5,7,8) for s in (0,5,10,15)]
        value=dict(windows={"5":dict(train=rows)})
        before=copy.deepcopy(value)
        self.assertIs(frozen_rows(value),rows)
        self.assertEqual(value,before)
        broken=copy.deepcopy(value);broken["windows"]["5"]["train"][0]=dict(rows[1])
        with self.assertRaises(ValueError):frozen_rows(broken)
        broken=copy.deepcopy(value);broken["windows"]["5"]["train"][0]["episode"]=3
        with self.assertRaises(ValueError):frozen_rows(broken)

    def test_world_mm_chamfer_and_operation_hand_controller_motion(self):
        raw=single_raw(window=(2,73))
        raw["inputs"]["controller_mask"].fill_(True)
        initial=raw["inputs"]["controller_points"][:,:1].clone()
        shifts=torch.arange(5)*.001
        raw["inputs"]["controller_points"]=initial.expand_as(raw["inputs"]["controller_points"]).clone()
        raw["inputs"]["controller_points"][:,:,1,: ,0]+=shifts[None,:,None]
        # Deliberate inactive-left noise must not alter active-right summaries.
        raw["inputs"]["controller_points"][:,:,0,: ,0]+=100*shifts[None,:,None]
        processor=DataProcessor("001-rope")
        result=window_statistics(processor(raw),processor,centered_cd=True)
        self.assertEqual(result["operation_hands"],[False,True])
        self.assertEqual(result["controller_cumulative_by_hand"][-1][0]["count"],0)
        self.assertIsNone(result["controller_cumulative_by_hand"][-1][0]["rms_mm"])
        self.assertAlmostEqual(result["condition_t4_controller_rms_mm"],4.,places=3)
        self.assertEqual(len(result["copy_cd_mm"]),5)
        self.assertEqual(len(result["centered_copy_cd_mm"]),5)
        self.assertEqual(result["condition_frames"],[73,74,75,76,77])
        self.assertEqual(result["target_frames"],[74,75,76,77,78])

    def test_quantiles_do_not_invent_activity_thresholds(self):
        result=quantiles([0.,1.,2.,None])
        self.assertEqual(result["count"],3)
        self.assertEqual(result["values"]["median"],1.)
        self.assertEqual(quantiles([None])["count"],0)
        with self.assertRaises(ValueError):quantiles([float("nan")])


if __name__ == "__main__":
    unittest.main()
