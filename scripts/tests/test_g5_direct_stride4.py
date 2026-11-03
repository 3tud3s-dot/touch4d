"""Index-only sampling and synthetic checkpoint tests; no real data/GPU training."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, Dataset
from scripts.dataloader.splits import STRIDES
from scripts.experiments.v4_g5_direct_stride4 import protocol as p, checkpoint as ck
from scripts.experiments.v4_g5_signed_local import formal_checkpoint as original_ck
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash


def keys():
    return [[ep,s] for ep,n in ((1,176),(2,277),(4,349),(5,292),(7,210),(8,210)) for s in range(n)]


def data():
    train=keys();val=[[ep,s] for ep,n in ((3,273),(6,222)) for s in range(0,n,5)]
    manifest=dict(train=dict(window_keys=train,windows=1514,episodes={'1':176,'2':277,'4':349,'5':292,'7':210,'8':210}),
                  val=dict(window_keys=val,windows=100,episodes={'3':55,'6':45}))
    orders=[p.selected_indices(train,e) for e in range(8)]
    hashes=[p.digest(o) for o in orders]
    return dict(manifest=manifest,manifest_hash=p.digest(manifest),epoch_orders=orders,
                epoch_order_hashes=hashes,observed_epoch_order_hashes=hashes.copy(),sampling=copy.deepcopy(p.SAMPLING))


class Indices(Dataset):
    def __init__(self):self.windows=keys()
    def __len__(self):return len(self.windows)
    def __getitem__(self,i):return i


class SamplingTests(unittest.TestCase):
    def test_coverage_and_counts(self):
        d=data();p.validate_data(d)
        self.assertEqual([len(o) for o in d['epoch_orders']],list(p.COUNTS))
        self.assertEqual(sum(map(len,d['epoch_orders'])),3028)
        self.assertEqual(set(Counter(i for o in d['epoch_orders'] for i in o).values()),{2})

    def test_real_sampler_order_is_reproducible_and_does_not_touch_global_rng(self):
        raw=DataLoader(Indices(),batch_size=1,num_workers=0)
        for epoch in range(8):
            before=torch.get_rng_state().clone()
            a,indices=p.sampled_loader(raw,epoch)
            predicted=[indices[i] for i in p.original.probe_order(a)]
            actual=[int(v[0]) for v in a]
            self.assertEqual(predicted,actual)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            b,_=p.sampled_loader(raw,epoch)
            self.assertEqual(actual,[int(v[0]) for v in b])
            self.assertTrue(all(keys()[i][1]%4==epoch%4 for i in actual))

    def test_global_stride_is_untouched(self):
        self.assertEqual(STRIDES,{'train':1,'val':5,'test':5})

    def test_invalid_epoch_rejected(self):
        for epoch in (-1,8,True,1.5):
            with self.assertRaises(ValueError):p.selected_indices(keys(),epoch)

    def test_wrong_phase_duplicate_and_order_hash_are_rejected(self):
        for kind in ('phase','duplicate','hash'):
            d=data()
            if kind=='phase':d['epoch_orders'][0][0]=1
            elif kind=='duplicate':d['epoch_orders'][0][1]=d['epoch_orders'][0][0]
            else:d['epoch_order_hashes'][0]='wrong'
            with self.assertRaises(ValueError):p.validate_data(d)

    def test_config_matches_baseline_and_sampling_copy_is_independent(self):
        c=p.configuration('data','depth','a'*40)
        self.assertEqual(c['horizon'],15);self.assertEqual(c['prediction_interval'],1)
        self.assertEqual(c['batch_size'],1);self.assertEqual(c['gradient_clip_norm'],1.)
        p.validate_configuration(c)
        c['sampling']['phase_by_epoch'][0]=3
        self.assertEqual(p.SAMPLING['phase_by_epoch'][0],0)
        with self.assertRaises(ValueError):p.validate_configuration(c)

    def test_changed_runtime_json_rejected(self):
        with patch.object(p.json,'loads',return_value={}):
            with self.assertRaises(ValueError):p.configuration('data','depth','a'*40)

    def test_original_and_stride_formats_are_distinct(self):
        self.assertNotEqual(ck.VERSIONS['direct'],original_ck.VERSIONS['direct'])

    def test_output_is_new_separate_and_resume_requires_initial(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'v4-g5-direct-stride4-v1'/'run-001'
            self.assertEqual(p.output_check(out),out.resolve())
            with self.assertRaises(ValueError):p.output_check(out,resume=True)
            out.mkdir(parents=True)
            with self.assertRaises(FileExistsError):p.output_check(out)
            with self.assertRaises(ValueError):p.output_check(out,resume=True,protected=(tmp,))


def validation(score):
    episodes={str(ep):dict(samples=n,render_samples=n,supervised_geometry=0.,rgb=score,alpha=0.) for ep,n in ((3,55),(6,45))}
    item=dict(episodes=episodes,objective=score)
    return dict(model=copy.deepcopy(item),copy_initial=copy.deepcopy(item))


class Tiny(torch.nn.Linear):
    def __init__(self):
        super().__init__(2,1)
        self.config=dict(architecture='v4_g5_signed_local_direct_v1',horizon=15,prediction_interval=1)


class CheckpointTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp)/'v4-g5-direct-stride4-v1'/'run-001';root.mkdir(parents=True)
        model=Tiny();opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        identity=dict(run_id=str(root.resolve()),variant='direct',source_sha='a'*40,
            training=p.configuration('data','depth','a'*40),model=model.config,data=data(),
            initialization=dict(variant_model_hashes={'direct':state_hash(model.state_dict())}))
        return root,model,opt,identity

    def row(self,identity,epoch,score):
        d=identity['data'];order=d['epoch_orders'][epoch-1];manifest=d['manifest']
        return dict(epoch=epoch,sampling_phase=(epoch-1)%4,successful_updates=len(order),
            visited_train_windows=len(order),visited_val_windows=100,partial_train=False,
            partial_validation=False,optimizer_update_flags=[True]*len(order),
            window_order=[manifest['train']['window_keys'][i] for i in order],
            validation_window_order=manifest['val']['window_keys'],validation=validation(score))

    def test_strict_roundtrip_and_resume_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,model,opt,identity=self.fixture(tmp);ledger=ck.empty_ledger()
            ck.save_boundary(root,model,opt,identity,ledger)
            for ep,score in ((1,.2),(2,.3)):
                opt.zero_grad();model(torch.ones(1,2)).sum().backward();opt.step()
                ledger=ck.commit_epoch(ledger,self.row(identity,ep,score),identity['data']['manifest'])
                ck.save_boundary(root,model,opt,identity,ledger)
            other=Tiny();other_opt=torch.optim.AdamW(other.parameters(),lr=1e-4)
            restored=ck.restore(root,other,other_opt,identity)
            self.assertEqual(restored,ledger);self.assertEqual(restored['best_epoch'],1)
            self.assertEqual(restored['completed_epochs']%4,2)
            self.assertEqual(state_hash(model.state_dict()),state_hash(other.state_dict()))
            torch.testing.assert_close(opt.state_dict(),other_opt.state_dict(),rtol=0,atol=0)
            self.assertEqual(ck.read(root/'best.pt',identity)['ledger']['completed_epochs'],1)
            self.assertEqual(ck.read(root/'last.pt',identity)['ledger']['completed_epochs'],2)

    def test_partial_flags_accounting_and_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,model,opt,identity=self.fixture(tmp)
            initial=ck.save_boundary(root,model,opt,identity,ck.empty_ledger())
            for kind in ('partial','flags','phase','source','version'):
                s=copy.deepcopy(initial)
                if kind in ('source','version'):
                    if kind=='source':s['identity']['source_sha']='b'*40
                    else:s['version']=original_ck.VERSIONS['direct']
                else:
                    row=self.row(identity,1,.2)
                    if kind=='partial':row['partial_train']=True
                    elif kind=='flags':row['optimizer_update_flags'][0]=False
                    else:row['sampling_phase']=2
                    s['ledger']=ck.commit_epoch(s['ledger'],row,identity['data']['manifest'])
                with self.assertRaises(ValueError,msg=kind):ck.validate_state(s,identity)

    def test_missing_epoch_chain_and_stale_alias_rollback_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,model,opt,identity=self.fixture(tmp);initial=ck.save_boundary(root,model,opt,identity,ck.empty_ledger())
            ledger=ck.commit_epoch(ck.empty_ledger(),self.row(identity,1,.2),identity['data']['manifest'])
            ck.save_boundary(root,model,opt,identity,ledger)
            with self.assertRaises(ValueError):ck.publish_aliases(root,initial)
            (root/'epoch_01.pt').rename(root/'epoch_02.pt')
            with self.assertRaises(ValueError):ck.restore(root,model,opt,identity)


if __name__=='__main__':unittest.main()
