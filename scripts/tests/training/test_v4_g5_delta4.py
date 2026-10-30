"""Independent sparse-time protocol; does not alter historical delta1 tests."""
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g5_signed_local import protocol as p
from scripts.experiments.v4_g5_signed_local import formal_checkpoint as cp
from scripts.dataloader.splits import window_starts

class Delta4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_frozen_configuration(self):
        c=p.configuration('data','depth','sha','direct');p.validate_configuration(c)
        self.assertEqual((c['horizon'],c['prediction_interval'],c['epochs']),(15,4,8))
        self.assertEqual((c['lr'],c['weight_decay'],c['scale_weight']),(1e-4,1e-4,0.))
        c['prediction_interval']=1
        with self.assertRaises(ValueError):p.validate_configuration(c)
    def test_loader_interval_and_split(self):
        c=p.configuration('data','depth','sha','direct')
        with patch.object(p,'make_dataloader') as build:
            p.loader(c,'train',3)
            self.assertEqual(build.call_args.kwargs['prediction_interval'],4)
            self.assertEqual(build.call_args.kwargs['seed'],45)
            self.assertEqual(build.call_args.kwargs['horizon'],15)
        with self.assertRaises(ValueError):p.loader(c,'test')
    def test_window_span(self):
        self.assertEqual(list(window_starts(61,'train',15,4)),[0])
        self.assertEqual(list(window_starts(60,'train',15,4)),[])
        self.assertEqual(list(window_starts(71,'val',15,4)),[0,5,10])
    def test_initialization_h15_equivalence(self):
        r=p.paired_initialization()
        self.assertTrue(r['shared_exact']);self.assertTrue(r['direct_mapping_exact'])
        self.assertTrue(all(all(x.values()) for x in r['h15_all_attributes_exact'].values()))
    def test_real_frame_semantics_and_causality(self):
        model=p.fresh_model('direct');raw=synthetic_batch(32,15,4)
        batch=prepare_batch(raw,DataProcessor('001-rope'),model);x=batch['inputs']
        self.assertTrue(torch.equal(x['condition_frame_ids'],raw['meta']['frame_ids'][:,:-1]))
        self.assertTrue(torch.equal(raw['meta']['frame_ids'][:,1:]-raw['meta']['frame_ids'][:,:-1],torch.full_like(raw['meta']['frame_ids'][:,1:],4)))
        with torch.no_grad():
            out=model(x)
            y={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in x.items()}
            y['controller_points'][:,10:]+=1
            other=model(y)
        self.assertEqual(out['xyz'].shape[1],15)
        self.assertTrue(torch.isfinite(out['xyz']).all())
        self.assertTrue(torch.equal(out['xyz'][:,:10],other['xyz'][:,:10]))
    def test_checkpoint_namespace(self):
        self.assertIn('delta4',cp.VERSIONS['direct'])
        self.assertIn('delta4',p.BUCKETS['direct'])
        self.assertEqual(p.fresh_model('direct').config['prediction_interval'],4)

    def test_formal_runner_delta4_loss_update_accounting(self):
        from scripts.tests.training import test_v4_g5_training as original
        with patch.object(original,'synthetic_batch',side_effect=lambda n,h:synthetic_batch(n,h,4)):
            original.G5FormalTrainingTests().test_real_g4_training_loop_with_g5_skip_update_accounting()
