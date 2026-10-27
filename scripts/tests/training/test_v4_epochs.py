import unittest
from unittest.mock import patch
import torch
from scripts.experiments.full_training.v4_epochs import configuration,loader,PreparedLoader,PreparedProcessor
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.training.v4_runtime import build_model,prepare_batch
from scripts.data_process import DataProcessor


class V4EpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_original_epoch_loop_and_joint_validation_with_real_v4(self):
        from scripts.training.train import train_epoch,evaluate_loader
        from scripts.training.rendering import GaussianRenderer
        from scripts.experiments.v4_stage1.fixtures import analytic_rasterizer
        p=DataProcessor('001-rope');m=build_model('v4_stage1');raw=synthetic_batch(32,10)
        stream=PreparedLoader([raw],p,m);adapter=PreparedProcessor(p)
        renderer=GaussianRenderer(analytic_rasterizer)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
        before=m.motion[-1].weight.detach().clone()
        c=configuration('data','depth','sha')
        training=train_epoch(m,opt,stream,adapter,renderer,c,'cpu')
        self.assertTrue(torch.isfinite(torch.tensor(training['objective'])))
        self.assertFalse(torch.equal(before,m.motion[-1].weight))
        val=evaluate_loader(m,stream,adapter,renderer,c,'cpu')
        self.assertEqual(set(val),{'model','copy_initial'})
        self.assertAlmostEqual(val['model']['objective'],val['model']['supervised_geometry']+val['model']['rgb']+.1*val['model']['alpha'])

    def test_frozen_full_training_contract(self):
        c=configuration('data','depth','sha')
        self.assertEqual((c['epochs'],c['horizon'],c['prediction_interval'],c['single_epochs']),(8,10,1,0))
        self.assertEqual([c[k] for k in ('chamfer_weight','rgb_weight','alpha_weight','scale_weight','depth_weight')],[1.,1.,.1,0.,0.])
        for split in ('train','val'):
            with patch('scripts.experiments.full_training.v4_epochs.make_dataloader') as make:
                loader(c,split,3)
                self.assertEqual(make.call_args.args[1],split)
                self.assertEqual(make.call_args.kwargs['stage'],'all')
                self.assertEqual(make.call_args.kwargs['seed'],45)
                self.assertEqual(make.call_args.kwargs['horizon'],10)
                self.assertEqual(make.call_args.kwargs['prediction_interval'],1)
                self.assertTrue(make.call_args.kwargs['with_depth'])
        with self.assertRaises(ValueError):loader(c,'test')

    def test_prepare_once_and_delegate_denormalization(self):
        raw=synthetic_batch(32,10);p=DataProcessor('001-rope');m=build_model('v4_stage1')
        expected=prepare_batch(raw,p,m)
        actual=next(iter(PreparedLoader([raw],p,m)));adapter=PreparedProcessor(p)
        self.assertIs(adapter(actual),actual)
        torch.testing.assert_close(actual['inputs']['gs_xyz'],expected['inputs']['gs_xyz'],rtol=0,atol=0)
        torch.testing.assert_close(adapter.denormalize_batch(actual)['inputs']['gs_xyz'],p.denormalize_batch(expected)['inputs']['gs_xyz'])
        self.assertEqual(adapter.state_dict(),p.state_dict())
        with self.assertRaises(ValueError):adapter(raw)
