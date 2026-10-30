"""Strict A/B full-state checkpoint, original-Joint selection and epoch rollback."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from scripts.data_process import DataProcessor
from scripts.experiments.v4_g4_global_motion.protocol import digest
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.tests.training.test_v4_g5_training import manifest, row
from scripts.experiments.v4_g5_struct_state.model import VARIANTS, fresh_model, paired_initialization
from scripts.experiments.v4_g5_struct_state.protocol import configuration
from scripts.experiments.v4_g5_struct_state import checkpoint as cp


def same_snapshot(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys()==right.keys() and all(same_snapshot(left[k],right[k]) for k in left)
    if isinstance(left, (list,tuple)):
        return len(left)==len(right) and all(same_snapshot(a,b) for a,b in zip(left,right))
    return same_tree(left,right)


def identity(directory, model, initialization):
    m=manifest();orders=[[0,1]]*8
    return dict(run_id=str(Path(directory).resolve()),source_sha='test-source',variant=model.variant,
        training=configuration('data','depth','test-source',model.variant,.01),model=copy.deepcopy(model.config),
        processor=DataProcessor('001-rope').state_dict(),environment=dict(device='cpu-test'),
        initialization=copy.deepcopy(initialization),calibration_sha256='f'*64,
        data=dict(manifest=m,manifest_hash=digest(m),epoch_orders=orders,
            epoch_order_hashes=[digest(v) for v in orders],observed_epoch_order_hashes=[digest(v) for v in orders]))


class StructStateCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.initialization=paired_initialization(verify_outputs=False)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def model_opt(self, variant=VARIANTS[0]):
        model=fresh_model(variant)
        return model,torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)

    def step(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        factor=random.random()+float(np.random.rand())+float(torch.rand(()))
        sum(p.square().sum()*factor for p in model.parameters()).backward()
        optimizer.step()

    def test_each_variant_saves_fresh_complete_initial_and_last(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                model,opt=self.model_opt(variant);ident=identity(d,model,self.initialization)
                state=cp.save_boundary(d,model,opt,ident,cp.empty_ledger())
                self.assertEqual(state['version'],cp.VERSIONS[variant])
                self.assertEqual(state_hash(state['model']),self.initialization['variant_model_hashes'][variant])
                self.assertTrue(same_tree(model.state_dict(),cp.read(Path(d)/'initial.pt',ident)['model']))
                self.assertTrue((Path(d)/'last.pt').is_file());self.assertFalse((Path(d)/'best.pt').exists())
                self.assertEqual(len(state['parameter_signature']),len(list(model.parameters())))

    def test_returned_snapshot_does_not_follow_later_model_or_optimizer_changes(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            returned=cp.save_boundary(d,model,opt,ident,cp.empty_ledger());frozen=copy.deepcopy(returned)
            self.step(model,opt)
            self.assertTrue(same_snapshot(returned,frozen))
            self.assertTrue(same_snapshot(returned,cp.read(Path(d)/'initial.pt',ident)))

    def test_restore_replays_python_numpy_torch_and_adamw_exactly(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                model,opt=self.model_opt(variant);ident=identity(d,model,self.initialization)
                cp.save_boundary(d,model,opt,ident,cp.empty_ledger());self.step(model,opt)
                ledger=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
                cp.save_boundary(d,model,opt,ident,ledger)
                self.step(model,opt);expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
                report={};actual=cp.restore(d,model,opt,ident,rollback_info=report)
                self.assertEqual(actual,ledger);self.step(model,opt)
                self.assertTrue(same_tree(expected,model.state_dict()));self.assertTrue(same_tree(expected_opt,opt.state_dict()))
                self.assertEqual(report['restored_completed_epochs'],1)
                self.assertFalse(report['uncommitted_update_count_inferred_from_checkpoint'])

    def test_uncommitted_updates_rollback_without_accumulating_success_counts(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            cp.save_boundary(d,model,opt,ident,cp.empty_ledger());initial=copy.deepcopy(model.state_dict())
            self.step(model,opt);self.step(model,opt)
            report=dict(discarded_attempt_updates=2)
            ledger=cp.restore(d,model,opt,ident,rollback_info=report)
            self.assertEqual(ledger,cp.empty_ledger());self.assertTrue(same_tree(initial,model.state_dict()))
            self.assertEqual(report['discarded_attempt_updates'],2)
            self.assertEqual(report['restored_successful_updates'],0)

    def test_alias_repair_recovers_latest_durable_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            initial=cp.save_boundary(d,model,opt,ident,cp.empty_ledger());self.step(model,opt)
            ledger=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
            cp.save_boundary(d,model,opt,ident,ledger)
            torch.save(initial,Path(d)/'last.pt')
            cp.restore(d,model,opt,ident)
            self.assertEqual(cp.read(Path(d)/'last.pt',ident)['ledger'],ledger)
            self.assertEqual(cp.read(Path(d)/'best.pt',ident)['ledger'],ledger)

    def test_epochs_two_four_are_retained_and_best_uses_original_joint_only(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization);ledger=cp.empty_ledger()
            cp.save_boundary(d,model,opt,ident,ledger)
            for epoch,score,total in ((1,.3,.3),(2,.1,5.),(3,.2,.2),(4,.25,.25)):
                self.step(model,opt);sample=row(epoch,score);sample['training']['loss_total']=total
                sample['training']['loss_struct_weighted']=total-score
                ledger=cp.commit_epoch(ledger,sample,ident['data']['manifest'])
                cp.save_boundary(d,model,opt,ident,ledger)
            self.assertEqual(cp.read(Path(d)/'best.pt',ident)['ledger']['completed_epochs'],2)
            self.assertEqual(cp.read(Path(d)/'last.pt',ident)['ledger']['completed_epochs'],4)
            self.assertTrue((Path(d)/'epoch_02.pt').is_file());self.assertTrue((Path(d)/'epoch_04.pt').is_file())
            with self.assertRaises(FileExistsError):cp.save_boundary(d,model,opt,ident,ledger)

    def test_wrong_version_variant_model_and_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            state=cp.save_boundary(d,model,opt,ident,cp.empty_ledger())
            for version in ('touch4d-g5-direct-formal-h15-epochs-v1','touch4d-g4-formal-h15-epochs-v1','smoke',cp.VERSIONS[VARIANTS[1]]):
                with self.subTest(version=version),self.assertRaises(ValueError):cp.validate_state(dict(state,version=version),ident)
            for key in ('source_sha','run_id','calibration_sha256'):
                with self.subTest(key=key),self.assertRaises(ValueError):cp.restore(d,model,opt,dict(ident,**{key:'wrong'}))
            other,other_opt=self.model_opt(VARIANTS[1])
            with self.assertRaises(ValueError):cp.restore(d,other,other_opt,ident)

    def test_wrong_lambda_manifest_environment_processor_and_seed_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            cp.save_boundary(d,model,opt,ident,cp.empty_ledger())
            wrong=[]
            for key in ('data','environment','processor','initialization'):
                value=copy.deepcopy(ident);value[key]=dict(value[key],unexpected=True);wrong.append(value)
            value=copy.deepcopy(ident);value['training']['lambda_struct']=.005;wrong.append(value)
            for value in wrong:
                with self.assertRaises(ValueError):cp.restore(d,model,opt,value)

    def test_incomplete_optimizer_reordered_optimizer_frozen_model_rejected(self):
        for mode in ('partial','reversed','frozen'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                model,opt=self.model_opt();ident=identity(d,model,self.initialization)
                if mode=='partial':opt=torch.optim.AdamW(list(model.parameters())[:-1],lr=1e-4,weight_decay=1e-4)
                if mode=='reversed':opt=torch.optim.AdamW(list(model.parameters())[::-1],lr=1e-4,weight_decay=1e-4)
                if mode=='frozen':next(model.parameters()).requires_grad_(False)
                with self.assertRaises(ValueError):cp.save_boundary(d,model,opt,ident,cp.empty_ledger())

    def test_wrong_adamw_hyperparameters_and_parameter_signature_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            state=cp.save_boundary(d,model,opt,ident,cp.empty_ledger())
            for key,value in (('lr',2e-4),('weight_decay',0.),('betas',(.8,.99)),('foreach',True),('fused',True)):
                bad=copy.deepcopy(state);bad['optimizer']['param_groups'][0][key]=value
                with self.assertRaises(ValueError):cp.validate_state(bad,ident)
            bad=copy.deepcopy(state);bad['parameter_signature'][0]['shape']=[999]
            with self.assertRaises(ValueError):cp.validate_state(bad,ident)

    def test_fresh_weights_hash_requires_common_initialization_not_trained_weights(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            self.step(model,opt)
            with self.assertRaises(ValueError):cp.save_boundary(d,model,opt,ident,cp.empty_ledger())
            self.assertFalse(list(Path(d).glob('*.pt')))

    def test_partial_epoch_wrong_order_update_flags_or_validation_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            state=cp.save_boundary(d,model,opt,ident,cp.empty_ledger())
            ledger=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
            for key,value in (('partial_train',True),('partial_validation',True),('successful_updates',2),
                    ('optimizer_update_flags',[True,True]),('visited_val_windows',1),('window_order',[[2,74],[2,73]])):
                bad=copy.deepcopy(ledger);bad['history'][0][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):cp.validate_state(dict(state,ledger=bad),ident)
            bad=copy.deepcopy(ledger);bad['history'][0]['validation']['model']['episodes']['3']['samples']=0
            with self.assertRaises(ValueError):cp.validate_state(dict(state,ledger=bad),ident)

    def test_epoch_gap_newer_alias_and_forked_alias_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            state=cp.save_boundary(d,model,opt,ident,cp.empty_ledger())
            torch.save(state,Path(d)/'epoch_02.pt')
            with self.assertRaises(ValueError):cp.restore(d,model,opt,ident)
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            cp.save_boundary(d,model,opt,ident,cp.empty_ledger());self.step(model,opt)
            one=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest']);cp.save_boundary(d,model,opt,ident,one)
            self.step(model,opt);two=cp.commit_epoch(one,row(2),ident['data']['manifest']);cp.save_boundary(d,model,opt,ident,two)
            old=cp.read(Path(d)/'epoch_01.pt',ident)
            with self.assertRaises(ValueError):cp.check_aliases(Path(d),old)
            fork=copy.deepcopy(two);fork['history'][0]['training']['foreign']=True
            current=cp.read(Path(d)/'epoch_02.pt',ident)
            with self.assertRaises(ValueError):cp.check_aliases(Path(d),dict(current,ledger=fork))

    def test_read_only_evaluation_checks_source_processor_manifest_and_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            model,opt=self.model_opt();ident=identity(d,model,self.initialization)
            cp.save_boundary(d,model,opt,ident,cp.empty_ledger());path=Path(d)/'initial.pt'
            target=fresh_model(model.variant).eval().requires_grad_(False)
            before=state_hash(target.state_dict())
            state=cp.load_for_evaluation(path,target,ident['processor'],source_sha=ident['source_sha'],
                manifest=ident['data']['manifest'],calibration_sha256=ident['calibration_sha256'])
            self.assertEqual(state_hash(target.state_dict()),before);self.assertEqual(state['ledger'],cp.empty_ledger())
            with self.assertRaises(ValueError):cp.load_for_evaluation(path,target,ident['processor'],source_sha='wrong',manifest=ident['data']['manifest'])
            with self.assertRaises(ValueError):cp.load_for_evaluation(path,target,ident['processor'],source_sha=ident['source_sha'],manifest=ident['data']['manifest'],calibration_sha256='a'*64)


if __name__=='__main__':unittest.main()
