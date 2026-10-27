"""The audit must isolate objectives and preserve the one-window protocol."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.experiments.debug_audit.runner import weights,parameter_stats,CHECKPOINTS,BASE
from scripts.experiments.debug_audit.common import export_evidence
from scripts.training.train import copy_initial
from scripts.data_process import DataProcessor


class DebugRunnerTests(unittest.TestCase):
    def test_only_loss_changes(self):
        self.assertEqual(weights('E1'),dict(chamfer_weight=1.,rgb_weight=0.,alpha_weight=0.,scale_weight=0.))
        self.assertEqual(weights('E2'),dict(chamfer_weight=1.,rgb_weight=1.,alpha_weight=.1,scale_weight=.01))
        with self.assertRaises(ValueError):weights('E3')

    def test_historical_paths_frozen(self):
        self.assertEqual(len(CHECKPOINTS),4)
        self.assertTrue(all('p3-p4-spatial-dynamics-v1/runs/' in p for p in CHECKPOINTS.values()))
        self.assertEqual(BASE,'400b4c389325efe18d4d9a54eb5412e722ab22cc')

    def test_actual_parameter_change(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,1,bias=False))
        with torch.no_grad():model[0].weight.fill_(2)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        model(torch.ones(1,2)).sum().backward()
        with torch.no_grad():model[0].weight.add_(.25)
        result=parameter_stats(model,before)['0']
        self.assertAlmostEqual(result['gradient_norm'],2**.5)
        self.assertAlmostEqual(result['update_norm'],.25*2**.5)

    def test_export_without_optional_image_libraries(self):
        x=dict(gs_xyz=torch.zeros(1,3,3),gs_attr=torch.zeros(1,3,11),gs_mask=torch.ones(1,3,dtype=torch.bool),
               action_state=torch.zeros(1,5,2,13))
        batch=dict(inputs=x,targets=dict(target_xyz=[torch.zeros(1,4,3) for _ in range(5)],
            target_mask=[torch.ones(1,4,dtype=torch.bool) for _ in range(5)]),meta=dict(frame_ids=torch.arange(6)[None]))
        class Renderer:
            def __call__(self,*args):return dict(rgb=torch.zeros(1,5,3,4,6,3),alpha=torch.zeros(1,5,3,4,6))
            render_targets=__call__
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',{'PIL':None,'matplotlib':None}):
            out=Path(folder)/'evidence'
            evidence=export_evidence(out,copy_initial(x,'gs_geometry'),batch,DataProcessor('001-rope'),Renderer())
            self.assertEqual(len(list(out.glob('*.png'))),90)
            self.assertEqual(evidence['cameras'],3)
            self.assertTrue((out/'geometry.npz').exists())
            self.assertTrue((out/'h05-c2-alpha-pred.png').read_bytes().startswith(b'\x89PNG'))


if __name__=='__main__':unittest.main()
