"""Head-only formal runner/checkpoint contracts, using original CPU loop."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from scripts.tests.training.test_v4_g5_training import first,manifest,validation,row
from scripts.data_process import DataProcessor
from scripts.experiments.v4_g5_signed_local.model import G5DirectPredictor
from scripts.experiments.v4_g5_signed_local.protocol import configuration
from scripts.experiments.v4_g4_global_motion.protocol import digest
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch
from scripts.experiments.v4_g5_geometry import formal_checkpoint as cp
from scripts.experiments.v4_g5_geometry import checkpoint as smoke_cp
from scripts.experiments.v4_g5_geometry.train import initialize,perform_epoch


def identity(root,model):
    m=manifest();orders=[[0,1]]*8
    return dict(run_id=str(Path(root).resolve()),source_sha='test-source',
        training=configuration('data','depth','test-source','direct'),geometry_config=model.geometry_config,
        backbone_state_hash=state_hash(model.backbone.state_dict()),
        data=dict(manifest=m,manifest_hash=digest(m),epoch_orders=orders,epoch_order_hashes=[digest(o) for o in orders]))


class GeometryFormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(42);self.model=initialize(G5DirectPredictor(horizon=15,prediction_interval=1))
        self.opt=torch.optim.AdamW(self.model.head.parameters(),lr=1e-4,weight_decay=1e-4)

    def step(self):
        self.opt.zero_grad(set_to_none=True)
        value=random.random()+float(np.random.rand())+float(torch.rand(()))
        (self.model.head[-1].weight.sum()*value).backward();self.opt.step()

    def test_fresh_head_pair_and_frozen_backbone(self):
        other=initialize(copy.deepcopy(self.model.backbone))
        self.assertTrue(same_tree(self.model.head.state_dict(),other.head.state_dict()))
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
        self.assertEqual(sum(p.numel() for p in self.model.head.parameters()),25542)
        self.assertEqual({id(p) for g in self.opt.param_groups for p in g['params']},{id(p) for p in self.model.head.parameters()})

    def test_h15_equivalence_with_original(self):
        x=prepare_batch(synthetic_batch(50,15,1),DataProcessor('001-rope'),self.model)['inputs']
        with torch.no_grad():self.assertTrue(same_tree(self.model(x),self.model.backbone(x)))

    def test_epoch_resume_replays_rng_and_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);initial=cp.save_boundary(d,self.model,self.opt,ident,cp.empty_ledger())
            self.step();ledger=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
            cp.save_boundary(d,self.model,self.opt,ident,ledger)
            torch.save(initial,Path(d)/'last.pt')  # Simulated interrupted alias publication.
            self.step();expected=copy.deepcopy(self.model.head.state_dict());opt=copy.deepcopy(self.opt.state_dict())
            restored=cp.restore(d,self.model,self.opt,ident);self.assertEqual(restored,ledger)
            self.step();self.assertTrue(same_tree(expected,self.model.head.state_dict()));self.assertTrue(same_tree(opt,self.opt.state_dict()))
            self.assertEqual(cp.read(Path(d)/'last.pt',ident)['ledger'],ledger)

    def test_uncommitted_epoch_rollback_and_immutable_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);original=copy.deepcopy(self.model.head.state_dict())
            cp.save_boundary(d,self.model,self.opt,ident,cp.empty_ledger())
            self.step();ledger=cp.restore(d,self.model,self.opt,ident)
            self.assertEqual(ledger['successful_updates'],0);self.assertTrue(same_tree(original,self.model.head.state_dict()))
            self.step();ledger=cp.commit_epoch(ledger,row(1),ident['data']['manifest']);cp.save_boundary(d,self.model,self.opt,ident,ledger)
            with self.assertRaises(FileExistsError):cp.save_boundary(d,self.model,self.opt,ident,ledger)

    def test_strict_format_run_source_architecture_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);s=cp.save_boundary(d,self.model,self.opt,ident,cp.empty_ledger())
            for version in (smoke_cp.VERSION,'touch4d-g5-direct-formal-h15-epochs-v1','touch4d-g4-formal-h15-epochs-v1'):
                with self.assertRaises(ValueError):cp.validate(dict(s,version=version),ident)
            for key in ('source_sha','run_id','backbone_state_hash'):
                wrong=dict(ident,**{key:'wrong'})
                with self.assertRaises(ValueError):cp.restore(d,self.model,self.opt,wrong)
            wrong=copy.deepcopy(ident);wrong['geometry_config']['ar_feedback']=True
            with self.assertRaises(ValueError):cp.validate(dict(s,identity=wrong),wrong)

    def test_wrong_backbone_or_optimizer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);cp.save_boundary(d,self.model,self.opt,ident,cp.empty_ledger())
            with torch.no_grad():next(self.model.backbone.parameters()).add_(.01)
            with self.assertRaises(ValueError):cp.restore(d,self.model,self.opt,ident)
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model)
            opt=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
            with self.assertRaises(ValueError):cp.save_boundary(d,self.model,opt,ident,cp.empty_ledger())

    def test_partial_validation_bad_order_and_counts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);state=cp.save_boundary(d,self.model,self.opt,ident,cp.empty_ledger())
            good=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
            for key,value in (('window_order',[[2,74],[2,73]]),('successful_updates',2),('partial_train',True),('visited_train_windows',1)):
                bad=copy.deepcopy(good);bad['history'][0][key]=value
                with self.assertRaises(ValueError):cp.validate(dict(state,ledger=bad),ident)
            bad=copy.deepcopy(good);bad['history'][0]['validation']['model']['episodes']['3']['samples']=0
            with self.assertRaises(ValueError):cp.validate(dict(state,ledger=bad),ident)

    def test_best_uses_full_joint_not_latest(self):
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);ledger=cp.empty_ledger();cp.save_boundary(d,self.model,self.opt,ident,ledger)
            for e,score in ((1,.1),(2,.2)):
                self.step();ledger=cp.commit_epoch(ledger,row(e,score),ident['data']['manifest']);cp.save_boundary(d,self.model,self.opt,ident,ledger)
            self.assertEqual(cp.read(Path(d)/'best.pt',ident)['ledger']['completed_epochs'],1)
            self.assertEqual(cp.read(Path(d)/'last.pt',ident)['ledger']['completed_epochs'],2)

    def test_missing_durable_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ident=identity(d,self.model);s=cp.save_boundary(d,self.model,self.opt,ident,cp.empty_ledger())
            torch.save(s,Path(d)/'epoch_02.pt')
            with self.assertRaises(ValueError):cp.restore(d,self.model,self.opt,ident)

    def test_actual_original_runner_head_update_and_frozen_validation(self):
        def raster(**args):
            signal=args['means'].mean()+args['scales'].mean()*100
            rgb=torch.sigmoid(signal).expand(1,args['height'],args['width'],3)
            alpha=torch.sigmoid(signal).expand(1,args['height'],args['width'],1)
            return torch.cat((rgb,alpha*.5),-1),alpha,{}
        train=first(synthetic_batch(50,15,1));a=copy.deepcopy(train);a['meta']['episode_id'][0]=3
        b=copy.deepcopy(train);b['meta']['episode_id'][0]=6;b['inputs']['operation_hands'][:]=True
        before=state_hash(self.model.backbone.state_dict());head=state_hash(self.model.head.state_dict())
        result=perform_epoch(self.model,self.opt,DataProcessor('001-rope'),GaussianRenderer(raster),
            configuration('data','depth','test','direct'),'cpu',[train],[a,b],[[2,73]],[[3,73],[6,73]])
        self.assertEqual(result['successful_updates'],1);self.assertEqual(result['optimizer_update_flags'],[True])
        self.assertEqual(result['visited_val_windows'],2)
        self.assertEqual(before,state_hash(self.model.backbone.state_dict()))
        self.assertNotEqual(head,state_hash(self.model.head.state_dict()))
        self.assertEqual(len(result['head_updates']),1)
        self.assertEqual(len(result['geometry_diagnostics']['val']),2)


if __name__=='__main__':unittest.main()
