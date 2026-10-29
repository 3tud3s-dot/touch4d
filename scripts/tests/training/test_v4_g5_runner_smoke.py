"""Bounded integration instrumentation, real runner CPU fixtures, no GPU jobs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts.data_process import DataProcessor
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_stage1.fixtures import synthetic_batch,analytic_rasterizer
from scripts.tests.training.test_v4_g5_training import first
from scripts.experiments.v4_g5_signed_local.protocol import fresh_model,configuration
from scripts.experiments.v4_g5_signed_local import runner_smoke as smoke


class RunnerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_smoke_namespace_rejects_formal_or_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            good=root/'v4-g5-runner-integration-smoke-v1'/'smoke-one'
            self.assertEqual(smoke.smoke_output(good),good.resolve())
            for bad in (root/'v4-g5-direct-v1-training'/'run-001',good.parent/'run-001'):
                with self.assertRaises(ValueError):smoke.smoke_output(bad)
            good.mkdir(parents=True)
            with self.assertRaises(FileExistsError):smoke.smoke_output(good)

    def test_formal_and_model_files_guard_rejects_changes(self):
        with patch.object(smoke,'source_preflight',return_value=('sha',{})),patch('subprocess.check_output',side_effect=[b'a',b'b']):
            with self.assertRaises(ValueError):smoke.guard_source('sha')
        self.assertTrue({'train.py','monitor.py','formal_checkpoint.py','model.py','motion.py','propagation.py'}<=set(smoke.FROZEN))

    def test_active_observer_real_loss_gradient_equivalence_both_variants(self):
        processor=DataProcessor('001-rope');renderer=GaussianRenderer(analytic_rasterizer)
        for variant in ('direct','propagate'):
            model=fresh_model(variant)
            with torch.no_grad():model.direct_motion.mapping[-1].weight.normal_(std=.001)
            batch=prepare_batch(first(synthetic_batch(32,15)),processor,model)
            result=smoke.equivalence(model,batch,processor,renderer)
            self.assertTrue(result['passed']);self.assertTrue(result['gradient_close'])
            self.assertTrue(result['active_local_projection']);self.assertEqual(result['optimizer_updates'],0)

    def test_audit_observes_real_formal_runner_two_updates_and_frozen_val(self):
        processor=DataProcessor('001-rope');model=fresh_model('propagate')
        raw=first(synthetic_batch(32,15));v3=copy.deepcopy(raw);v6=copy.deepcopy(raw)
        v3['meta']['episode_id'].fill_(3);v6['meta']['episode_id'].fill_(6);v6['inputs']['operation_hands'].fill_(True)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        audit=smoke.UpdateAudit(model,opt);pre=opt.register_step_pre_hook(audit.before);post=opt.register_step_post_hook(audit.after)
        original=torch.nn.utils.clip_grad_norm_
        try:
            with patch('torch.nn.utils.clip_grad_norm_',side_effect=lambda *a,**k:audit.clip(original,*a,**k)):
                row=smoke.perform_epoch(model,opt,processor,GaussianRenderer(analytic_rasterizer),
                    configuration('data','depth','sha','propagate'),'cpu',[raw,raw],[v3,v6],[[2,73],[2,73]],[[3,73],[6,73]])
        finally:pre.remove();post.remove()
        self.assertEqual(row['optimizer_update_flags'],[True,True]);self.assertEqual(row['successful_updates'],2)
        self.assertEqual(len(audit.records),2);self.assertEqual(audit.clips,2)
        self.assertGreater(audit.records[0]['groups']['direct_output']['raw_gradient_norm'],0)
        self.assertGreater(audit.records[1]['groups']['propagation_score']['raw_gradient_norm'],0)
        self.assertEqual(set(row['g5_diagnostics']['phases']),{'train','val'})
        with self.assertRaises(AssertionError):audit.before(opt,(),{})


if __name__=='__main__':unittest.main()
