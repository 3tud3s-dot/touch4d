"""CPU continuation lineage, frozen protocol, optimizer/RNG and Epoch replay."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.tests.training.test_v4_g5_training import identity as old_identity,row
from scripts.experiments.v4_g5_signed_local import continuation_checkpoint as cp
from scripts.experiments.v4_g5_signed_local import formal_checkpoint as old
from scripts.experiments.v4_g5_signed_local.protocol import fresh_model,probe_order
from scripts.experiments.v4_g4_global_motion.protocol import epoch_order,digest
from scripts.experiments.v4_g4_global_motion.checkpoint import rng_state
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash,sha256


def fixture(root):
    model=fresh_model('direct');opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    old_id=old_identity(root,model,'direct');old_id['source_sha']=cp.PARENT_SHA
    old_id['training']['source_sha']=cp.PARENT_SHA
    ledger=old.empty_ledger()
    for ep in range(1,9):ledger=old.commit_epoch(ledger,row(ep,2.-ep*.1),old_id['data']['manifest'])
    parent=dict(version=old.VERSIONS['direct'],identity=old_id,ledger=ledger,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state())
    data=copy.deepcopy(old_id['data']);data['epoch_orders']*=2
    data['epoch_order_hashes']*=2;data['observed_epoch_order_hashes']*=2
    new_id=dict(run_id=str(Path(root).resolve()),variant='direct',source_sha='continuation-sha',
        training=cp.training_config('data','depth','continuation-sha'),model=model.config,processor=old_id['processor'],
        environment=old_id['environment'],data=data,parent_source_sha=cp.PARENT_SHA,
        parent_ledger=copy.deepcopy(ledger),parent_model_hash=state_hash(model.state_dict()),parent_sha256='fixture-hash')
    return model,opt,parent,new_id


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_frozen_configuration_only_changes_budget_and_source(self):
        c=cp.training_config('data','depth','sha')
        old_c=cp.configuration('data','depth','sha','direct')
        self.assertEqual(c,dict(old_c,epochs=16));self.assertEqual(c['horizon'],15)
        self.assertEqual(c['prediction_interval'],1)
        self.assertEqual(c['lr'],1e-4);self.assertEqual(c['chamfer_weight'],1.)
        self.assertEqual(c['max_train_batches'],0)

    def test_parent_hash_direct_format_and_exact_last8(self):
        with tempfile.TemporaryDirectory() as folder:
            model,opt,parent,ident=fixture(folder);path=Path(folder)/'parent.pt';torch.save(parent,path)
            s=cp.load_parent(path,sha256(path),ident['processor'],ident['data']['manifest'])
            self.assertTrue(same_tree(s['model'],model.state_dict()))
            with self.assertRaises(ValueError):cp.load_parent(path,'0'*64,ident['processor'],ident['data']['manifest'])
            parent['version']='touch4d-g5-direct-architecture-v1';torch.save(parent,path)
            with self.assertRaises(ValueError):cp.load_parent(path,sha256(path),ident['processor'],ident['data']['manifest'])

    def test_resume_epoch9_rolls_back_interruption_with_optimizer_rng_and_aliases(self):
        def step(model,opt):
            opt.zero_grad(set_to_none=True)
            factor=random.random()+float(np.random.rand())+float(torch.rand(()))
            (model.direct_motion.mapping[-1].weight.sum()*factor).backward();opt.step()
        with tempfile.TemporaryDirectory() as folder:
            model,opt,parent,ident=fixture(folder);ledger=parent['ledger']
            cp.save(folder,model,opt,ident,ledger)
            step(model,opt);ledger=old.commit_epoch(ledger,row(9,.9),ident['data']['manifest']);cp.save(folder,model,opt,ident,ledger)
            checkpoint=copy.deepcopy(torch.load(Path(folder)/'epoch_09.pt',weights_only=False))
            step(model,opt);expected_model=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
            (Path(folder)/'epoch_10.pt.interrupted.tmp').write_bytes(b'preserved')
            # Deliberately stale same-Run alias is repaired from immutable chain.
            (Path(folder)/'last.pt').write_bytes((Path(folder)/'initial.pt').read_bytes())
            restored=cp.restore(folder,model,opt,ident)
            self.assertEqual(restored,ledger);self.assertTrue(same_tree(opt.state_dict(),checkpoint['optimizer']))
            step(model,opt)
            self.assertTrue(same_tree(expected_model,model.state_dict()));self.assertTrue(same_tree(expected_opt,opt.state_dict()))
            self.assertTrue((Path(folder)/'epoch_10.pt.interrupted.tmp').exists())
            self.assertEqual(torch.load(Path(folder)/'last.pt',weights_only=False)['ledger']['completed_epochs'],9)
            with self.assertRaises(FileExistsError):cp.save(folder,model,opt,ident,ledger)

    def test_all_additional_eight_boundaries_and_best_keep_parent_history(self):
        with tempfile.TemporaryDirectory() as folder:
            model,opt,parent,ident=fixture(folder);ledger=parent['ledger'];cp.save(folder,model,opt,ident,ledger)
            for ep in range(9,17):
                ledger=old.commit_epoch(ledger,row(ep,.8+abs(ep-12)*.01),ident['data']['manifest'])
                cp.save(folder,model,opt,ident,ledger)
            restored=cp.restore(folder,model,opt,ident)
            self.assertEqual(restored['completed_epochs'],16);self.assertEqual(restored['best_epoch'],12)
            self.assertEqual(restored['history'][:8],parent['ledger']['history'])
            self.assertEqual(len(list(Path(folder).glob('epoch_*.pt'))),8)
            with self.assertRaises(ValueError):old.validate_state(torch.load(Path(folder)/'last.pt',weights_only=False),ident)
            bad=copy.deepcopy(torch.load(Path(folder)/'last.pt',weights_only=False));bad['ledger']['completed_epochs']=17
            with self.assertRaises(ValueError):cp.validate(bad,ident)

    def test_mismatched_identity_and_incomplete_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model,opt,parent,ident=fixture(folder);cp.save(folder,model,opt,ident,parent['ledger'])
            state=torch.load(Path(folder)/'initial.pt',weights_only=False)
            for key,value in [('source_sha','other'),('run_id',str(Path(folder)/'other'))]:
                changed=copy.deepcopy(ident);changed[key]=value
                with self.assertRaises(ValueError):cp.validate(state,changed)
            changed=copy.deepcopy(state);changed['ledger']['history'][0]['successful_updates']+=1
            with self.assertRaises(ValueError):cp.validate(changed,ident)
            other=fresh_model('direct');other.config=dict(other.config,prediction_interval=2)
            with self.assertRaises(ValueError):cp.restore(folder,other,opt,ident)
            newrow=row(9,.5);newrow['visited_train_windows']=1
            ledger=old.commit_epoch(parent['ledger'],newrow,ident['data']['manifest'])
            with self.assertRaises(ValueError):cp.save(folder,model,opt,ident,ledger)
            self.assertFalse((Path(folder)/'epoch_09.pt').exists())

    def test_epochs9_16_real_loader_shuffle_uses_seed50_through57(self):
        for index in range(8,16):
            dl=DataLoader(range(31),batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42+index))
            self.assertEqual(probe_order(dl),epoch_order(31,index))


if __name__=='__main__':unittest.main()
