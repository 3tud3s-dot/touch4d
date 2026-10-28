import copy
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.data_process import DataProcessor
from scripts.experiments.full_training.v4_epochs import configuration as original,loader
from scripts.experiments.v4_g4_global_motion.protocol import configuration,epoch_order,fresh_model,full_validation_score,CheckedStream
from scripts.experiments.v4_g4_global_motion.checkpoint import save_boundary,restore,validate_state,load_for_evaluation
from scripts.experiments.v4_g4_global_motion.train import ContributionLog,perform_epoch
from scripts.experiments.v4_g4_global_motion.evaluate import motion_metrics
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_stage1.fixtures import synthetic_batch,analytic_rasterizer
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch
from scripts.training.train import copy_initial


def val(score=1.):
    return {k:dict(objective=score,episodes={'3':dict(samples=1),'6':dict(samples=1)}) for k in ('model','copy_initial')}


def identity(model):
    return dict(run_id='test-run',source_sha='test',model=model.config,processor={'scale':1},
        data=dict(manifest=dict(train=dict(window_keys=[[1,0]],episodes={'1':1}),
                  val=dict(window_keys=[[3,0],[6,0]],episodes={'3':1,'6':1})),epoch_orders=[[0]]*8))


def empty():return dict(completed_epochs=0,successful_updates=0,best_epoch=None,best_score=None,history=[])


def completed(old,score):
    s=copy.deepcopy(old);n=s['completed_epochs']+1
    s['history'].append(dict(epoch=n,successful_updates=1,window_order=[[1,0]],
                       validation_window_order=[[3,0],[6,0]],validation=val(score)))
    s['completed_epochs']=n;s['successful_updates']+=1
    if s['best_score'] is None or score<s['best_score']:s['best_score']=score;s['best_epoch']=n
    return s


class G4TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_protocol_exact_and_no_test_split(self):
        c=configuration('data','depth','sha');expected=original('data','depth','sha',15)
        expected['architecture']='v4_g4_global_motion_v1';self.assertEqual(c,expected)
        for e in range(8):
            with patch('scripts.experiments.full_training.v4_epochs.make_dataloader') as m:
                loader(c,'train',e);self.assertEqual(m.call_args.kwargs['seed'],42+e)
                self.assertEqual(m.call_args.kwargs['horizon'],15)
                self.assertEqual(m.call_args.kwargs['stage'],'all')
        with self.assertRaises(ValueError):loader(c,'test')

    def test_shuffle_matches_actual_dataloader_including_base_seed(self):
        for e in range(8):
            actual=[int(i) for b in DataLoader(range(31),batch_size=1,shuffle=True,num_workers=0,
                       generator=torch.Generator().manual_seed(42+e)) for i in b]
            self.assertEqual(epoch_order(31,e),actual)
            self.assertEqual(sorted(actual),list(range(31)))
        self.assertNotEqual(epoch_order(31,0),torch.randperm(31,generator=torch.Generator().manual_seed(42)).tolist())

    def test_checked_stream_rejects_skip_reorder(self):
        def batch(ep,start):return {'meta':{'episode_id':torch.tensor([ep]),'frame_ids':torch.tensor([[start,start+1]])}}
        with self.assertRaises(ValueError):list(CheckedStream([batch(1,1)],[[1,0]]))
        with self.assertRaises(ValueError):list(CheckedStream([batch(1,0)],[[1,0],[1,1]]))
        self.assertEqual(len(list(CheckedStream([batch(1,0)],[[1,0]]))),1)

    def test_fresh_optimizer_includes_new_parameters(self):
        model,evidence=fresh_model();self.assertTrue(evidence['shared_exact']);self.assertTrue(evidence['new_output_zero'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        ids={id(p) for g in optimizer.param_groups for p in g['params']}
        self.assertTrue(all(id(p) in ids for p in model.global_motion.parameters()))

    def test_full_validation_required_for_selection(self):
        manifest={'val':{'episodes':{'3':1,'6':1}}}
        self.assertEqual(full_validation_score(val(2),manifest),2)
        bad=val();bad['model']['episodes'].pop('6')
        with self.assertRaises(ValueError):full_validation_score(bad,manifest)
        bad=val();bad['copy_initial']['episodes']['3']['samples']=0
        with self.assertRaises(ValueError):full_validation_score(bad,manifest)

    def test_checkpoint_resume_rng_optimizer_and_crash_alias_recovery(self):
        model,_=fresh_model();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);ident=identity(model)
        def step():
            opt.zero_grad(set_to_none=True)
            value=random.random()+float(np.random.rand())+float(torch.rand(()))
            (model.global_motion.mapping[-1].weight.sum()*value).backward();opt.step()
        with tempfile.TemporaryDirectory() as d:
            initial=save_boundary(d,model,opt,ident,empty())
            step();state=completed(empty(),1);save_boundary(d,model,opt,ident,state)
            # Simulate publication interrupted after durable epoch, before updating aliases.
            torch.save(initial,Path(d)/'last.pt')
            step();expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
            restored=restore(d,model,opt,ident);self.assertEqual(restored['completed_epochs'],1)
            step();self.assertTrue(same_tree(expected,model.state_dict()));self.assertTrue(same_tree(expected_opt,opt.state_dict()))
            save_boundary(d,model,opt,ident,completed(state,2))
            best=torch.load(Path(d)/'best.pt',weights_only=False)
            self.assertEqual(best['ledger']['completed_epochs'],1)
            with self.assertRaises(FileExistsError):save_boundary(d,model,opt,ident,completed(state,2))
            wrong=dict(ident,source_sha='wrong')
            with self.assertRaises(ValueError):restore(d,model,opt,wrong)
            with self.assertRaises(ValueError):load_for_evaluation(Path(d)/'best.pt',model,{'scale':2})

    def test_checkpoint_rejects_smoke_g0_partial_and_bad_best(self):
        model,_=fresh_model();opt=torch.optim.AdamW(model.parameters());ident=identity(model)
        with tempfile.TemporaryDirectory() as d:
            s=save_boundary(d,model,opt,ident,empty())
            for version in ('touch4d-g4-cuda-smoke-v1','touch4d-v4-stage1-smoke-v1'):
                bad=dict(s,version=version)
                with self.assertRaises(ValueError):validate_state(bad,ident)
            bad=copy.deepcopy(s);bad['ledger']['completed_epochs']=1
            with self.assertRaises(ValueError):validate_state(bad,ident)
            bad=copy.deepcopy(s);bad['ledger']=completed(empty(),1);bad['ledger']['best_score']=.1
            with self.assertRaises(ValueError):validate_state(bad,ident)

    def test_contribution_units_and_mask(self):
        log=ContributionLog(.2)
        hands=torch.tensor([[[.01,0,0],[0,.02,0]]]);delta=hands.sum(1)
        log(None,(None,0),(delta,dict(hand_contribution=hands,hand_valid=torch.ones(1,2,dtype=torch.bool))))
        row=log.summary()['1'];self.assertAlmostEqual(row['hand_mean_xyz_mm'][0][0],2,places=5)
        self.assertAlmostEqual(row['global_mean_xyz_mm'][1],4,places=5)
        with self.assertRaises(ValueError):log(None,(None,0),(delta,dict(hand_contribution=hands,hand_valid=torch.zeros(1,2,dtype=torch.bool))))

    def test_geometry_translation_and_copy_diagnostics(self):
        model,_=fresh_model();processor=DataProcessor('001-rope')
        x=prepare_batch(synthetic_batch(32,15),processor,model)['inputs']
        # Evaluate first sample (the second is padded); helper handles valid indices only.
        x={k:(v[:1] if isinstance(v,torch.Tensor) else v) for k,v in x.items()}
        pred=copy_initial(x,'gs_geometry');rows=motion_metrics(pred,x,processor.state_dict()['scale'])
        self.assertTrue(all(r['displacement_rms_mm']==0 for r in rows))
        pred['xyz']=pred['xyz'].clone();pred['xyz'][...,1]+=.01
        rows=motion_metrics(pred,x,processor.state_dict()['scale'])
        for r in rows:
            self.assertLess(r['global_translation_removed_rms_mm'],1e-4)
            if r['neighbor_distortion_p95'] is not None:self.assertLess(r['neighbor_distortion_p95'],1e-4)

    def test_real_g4_original_train_val_loops_with_mock_renderer(self):
        model,_=fresh_model();processor=DataProcessor('001-rope');c=configuration('data','depth','sha')
        raw=synthetic_batch(32,15)
        # Keep the first sample as a true batch-size-one protocol fixture.
        def first(v):
            if isinstance(v,torch.Tensor):return v[:1]
            if isinstance(v,dict):return {k:first(x) for k,x in v.items()}
            if isinstance(v,list):return v[:1] if v and isinstance(v[0],(str,int)) else [first(x) for x in v]
            return v
        cameras=raw['meta']['depth_camera_ids'][:1]
        raw=first(raw);raw['meta']['depth_camera_ids']=cameras
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        before=model.global_motion.mapping[-1].weight.detach().clone()
        row=perform_epoch(model,opt,processor,GaussianRenderer(analytic_rasterizer),c,'cpu',[raw],[raw],[[2,73]],[[2,73]])
        self.assertEqual(row['successful_updates'],1)
        self.assertFalse(torch.equal(before,model.global_motion.mapping[-1].weight))
        self.assertEqual(set(row['contributions']),{'train','val'})
        self.assertAlmostEqual(row['validation']['model']['objective'],row['validation']['model']['supervised_geometry']+
                               row['validation']['model']['rgb']+.1*row['validation']['model']['alpha'])


if __name__=='__main__':unittest.main()
