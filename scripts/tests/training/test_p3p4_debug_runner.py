"""The audit must isolate objectives and preserve the one-window protocol."""
import unittest
import torch
from scripts.experiments.debug_audit.runner import weights,parameter_stats,CHECKPOINTS,BASE


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


if __name__=='__main__':unittest.main()
