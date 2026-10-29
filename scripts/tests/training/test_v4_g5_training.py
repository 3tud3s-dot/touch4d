"""CPU formal protocol/resume/orchestration tests; no formal or GPU training."""
import copy
import json
import random
import tempfile
from pathlib import Path
import unittest
import sys
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.data_process import DataProcessor
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch,v4_objective
from scripts.experiments.v4_stage1.fixtures import synthetic_batch,analytic_rasterizer
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local import protocol as p
from scripts.experiments.v4_g5_signed_local import formal_checkpoint as cp
from scripts.experiments.v4_g5_signed_local.train import perform_epoch
from scripts.experiments.v4_g5_signed_local import train as entry
from scripts.experiments.v4_g5_signed_local.monitor import EpochMonitor,ObservedPredictor
from scripts.experiments.v4_g5_signed_local.evaluate import window_diagnostics,evaluate_full,validate_pair_metadata


def first(value):
    if isinstance(value,torch.Tensor):return value[:1]
    if isinstance(value,dict):return {k:(v[:1] if k=='depth_camera_ids' else first(v)) for k,v in value.items()}
    if isinstance(value,list):return value[:1] if value and isinstance(value[0],(str,int)) else [first(v) for v in value]
    return value


def manifest(train=2):
    return dict(train=dict(windows=train,window_keys=[[2,73+i] for i in range(train)],episodes={'2':train}),
                val=dict(windows=2,window_keys=[[3,73],[6,73]],episodes={'3':1,'6':1}))


def validation(score=1.,counts=None):
    counts=counts or {'3':1,'6':1}
    episodes={k:dict(samples=n,render_samples=n,supervised_geometry=score*.5,rgb=score*.4,alpha=score,
                     geometry_mm=score*100,geometry=score*.5) for k,n in counts.items()}
    return {label:dict(objective=score,episodes=copy.deepcopy(episodes)) for label in ('model','copy_initial')}


def identity(root,model,variant):
    m=manifest();orders=[[0,1]]*8
    return dict(run_id=str(Path(root).resolve()),variant=variant,source_sha='test-source',
                training=p.configuration('data','depth','test-source',variant),model=model.config,
                processor=DataProcessor('001-rope').state_dict(),environment={'device':'cpu-test'},
                initialization=dict(variant_model_hashes={variant:p.state_hash(model.state_dict())}),
                data=dict(manifest=m,manifest_hash=p.g4.digest(m),epoch_orders=orders,
                          epoch_order_hashes=[p.g4.digest(o) for o in orders],observed_epoch_order_hashes=[p.g4.digest(o) for o in orders]))


def row(epoch,score=1.,flags=None):
    flags=[True,False] if flags is None else flags
    return dict(epoch=epoch,successful_updates=sum(flags),visited_train_windows=2,visited_val_windows=2,
                optimizer_update_flags=flags,window_order=[[2,73],[2,74]],validation_window_order=[[3,73],[6,73]],
                partial_train=False,partial_validation=False,validation=validation(score),training={})


class G5FormalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_pair_initialization_and_h15_all_attributes(self):
        evidence=p.paired_initialization(True)
        self.assertTrue(evidence['shared_exact']);self.assertTrue(evidence['direct_mapping_exact'])
        self.assertTrue(all(all(v.values()) for v in evidence['h15_all_attributes_exact'].values()))
        self.assertEqual(evidence['parameter_counts'],dict(g4=576107,direct=620511,propagate=674067))

    def test_original_config_loss_and_optimizer_parameters(self):
        for variant in ('direct','propagate'):
            c=p.configuration('data','depth','sha',variant);p.validate_configuration(c)
            baseline=p.g4.configuration('data','depth','sha')
            self.assertEqual({k:v for k,v in c.items() if k not in ('variant','gradient_clip_norm','architecture')},
                             {k:v for k,v in baseline.items() if k!='architecture'})
            model=p.fresh_model(variant);opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
            self.assertEqual({id(v) for g in opt.param_groups for v in g['params']},{id(v) for v in model.parameters()})
            c['chamfer_weight']=.1
            with self.assertRaises(ValueError):p.validate_configuration(c)

    def test_actual_loader_probe_eight_epochs_does_not_consume_generator(self):
        for epoch in range(8):
            raw=DataLoader(range(31),batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42+epoch))
            before=raw.generator.get_state().clone();observed=p.probe_order(raw)
            self.assertTrue(torch.equal(before,raw.generator.get_state()))
            self.assertEqual(observed,p.g4.epoch_order(31,epoch))
            self.assertEqual(observed,[int(v[0]) for v in raw])
        val=DataLoader(range(5),batch_size=1,generator=torch.Generator().manual_seed(42))
        self.assertEqual(p.probe_order(val),list(range(5)))

    def test_preflight_uses_frozen_reference_manifest_and_actual_loader_probes(self):
        from types import SimpleNamespace
        frozen=dict(train=dict(windows=1514,window_keys=[[ep,i] for ep,n in ((1,252),(2,252),(4,252),(5,252),(7,252),(8,254)) for i in range(n)],
                              episodes={str(ep):n for ep,n in ((1,252),(2,252),(4,252),(5,252),(7,252),(8,254))}),
                    val=dict(windows=100,window_keys=[[ep,i*5] for ep in (3,6) for i in range(50)],episodes={'3':50,'6':50}))
        orders=[p.g4.epoch_order(1514,e) for e in range(8)]
        data=dict(manifest=frozen,epoch_orders=orders)
        def fake(c,split,epoch=0):
            return DataLoader(range(1514 if split=='train' else 100),batch_size=1,shuffle=split=='train',
                num_workers=0,generator=torch.Generator().manual_seed(42+epoch))
        raw={k:fake(None,k) for k in ('train','val')}
        with patch.object(p.g4,'preflight',return_value=(raw,data)) as baseline,patch.object(p,'loader',side_effect=fake):
            _,got=p.preflight(p.configuration('data','depth','sha','direct'),'reference')
            self.assertEqual(got['observed_epoch_order_hashes'],[p.g4.digest(o) for o in orders])
            self.assertEqual(baseline.call_args.args[0]['architecture'],'v4_g4_global_motion_v1')
        self.assertNotIn('test',raw)

    def test_full_val100_episode_equal_selection_rejects_partial_or_sample_weighted(self):
        m={'val':dict(windows=100,episodes={'3':30,'6':70})}
        v=validation(1.,{'3':30,'6':70})
        for label in v:
            v[label]['episodes']['3'].update(supervised_geometry=.5,rgb=.4,alpha=1.)
            v[label]['episodes']['6'].update(supervised_geometry=1.,rgb=.8,alpha=2.)
            v[label]['objective']=1.5
        self.assertEqual(p.full_validation_score(v,m),1.5)
        bad=copy.deepcopy(v);bad['model']['objective']=1.7
        with self.assertRaises(ValueError):p.full_validation_score(bad,m)
        bad=copy.deepcopy(v);bad['model']['episodes']['6']['samples']=69
        with self.assertRaises(ValueError):p.full_validation_score(bad,m)
        bad=copy.deepcopy(v);bad['copy_initial']['episodes']['3']['render_samples']=29
        with self.assertRaises(ValueError):p.full_validation_score(bad,m)

    def test_variant_output_paths_no_overwrite_or_code_data_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for variant in ('direct','propagate'):
                output=root/p.BUCKETS[variant]/'run-one'
                actual,_=p.output_check(output,variant)
                self.assertEqual(actual,output.resolve());self.assertFalse(output.exists())
                with self.assertRaises(ValueError):p.output_check(output,'propagate' if variant=='direct' else 'direct')
                with self.assertRaises(ValueError):p.output_check(output,variant,protected=(root,))
                output.mkdir(parents=True)
                with self.assertRaises(FileExistsError):p.output_check(output,variant)

    def test_real_g4_training_loop_with_g5_skip_update_accounting(self):
        processor=DataProcessor('001-rope')
        for variant in ('direct','propagate'):
            model=p.fresh_model(variant);c=p.configuration('data','depth','sha',variant)
            raw=first(synthetic_batch(32,15));second=copy.deepcopy(raw)
            second['meta']['frame_ids']+=1;second['meta']['observation_timestamp_us']+=50000
            second['targets']['target_depth_valid'].fill_(False)
            second['targets']['target_depth'].zero_()
            v3=copy.deepcopy(raw);v3['meta']['episode_id'].fill_(3)
            v6=copy.deepcopy(raw);v6['meta']['episode_id'].fill_(6);v6['inputs']['operation_hands'].fill_(True)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            with patch('torch.nn.utils.clip_grad_norm_',wraps=torch.nn.utils.clip_grad_norm_) as clipped:
                result=perform_epoch(model,opt,processor,GaussianRenderer(analytic_rasterizer),c,'cpu',
                    [raw,second],[v3,v6],[[2,73],[2,74]],[[3,73],[6,73]])
            self.assertEqual(clipped.call_count,1);self.assertEqual(clipped.call_args.args[1],1.)
            self.assertEqual(result['visited_train_windows'],2);self.assertEqual(result['successful_updates'],1)
            self.assertEqual(result['optimizer_update_flags'],[True,False]);self.assertEqual(result['visited_val_windows'],2)
            self.assertEqual(set(result['g5_diagnostics']['phases']),{'train','val'})
            self.assertTrue(model.direct_motion.mapping[-1].weight.count_nonzero())
            self.assertAlmostEqual(p.full_validation_score(result['validation'],manifest()),
                result['validation']['model']['supervised_geometry']+result['validation']['model']['rgb']+.1*result['validation']['model']['alpha'])

    def test_trace_observer_predictions_and_gradients_identical_to_bare_model(self):
        processor=DataProcessor('001-rope')
        for variant in ('direct','propagate'):
            base=p.fresh_model(variant)
            with torch.no_grad():base.direct_motion.mapping[-1].weight.normal_(std=.002)
            other=copy.deepcopy(base);wrapped=ObservedPredictor(other,EpochMonitor(processor.state_dict()['scale'],variant=='propagate'))
            batch=prepare_batch(first(synthetic_batch(32,15)),processor,base)
            a=base(batch['inputs']);b=wrapped(batch['inputs'])
            for key in a:self.assertTrue(torch.equal(a[key],b[key]))
            target=batch['inputs']['gs_xyz']+torch.tensor([.01,-.02,.01])
            (a['xyz'][:,-1]-target).square().mean().backward();(b['xyz'][:,-1]-target).square().mean().backward()
            for (name,x),(name2,y) in zip(base.named_parameters(),other.named_parameters()):
                self.assertEqual(name,name2)
                self.assertTrue((x.grad is None and y.grad is None) or torch.equal(x.grad,y.grad),name)

    def test_checkpoint_epoch_rollback_rng_optimizer_alias_repair_both_variants(self):
        for variant in ('direct','propagate'):
            model=p.fresh_model(variant);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            def step():
                opt.zero_grad(set_to_none=True)
                value=random.random()+float(np.random.rand())+float(torch.rand(()))
                (model.direct_motion.mapping[-1].weight.sum()*value).backward();opt.step()
            with tempfile.TemporaryDirectory() as directory:
                ident=identity(directory,model,variant);ledger=cp.empty_ledger()
                initial=cp.save_boundary(directory,model,opt,ident,ledger)
                step();one=cp.commit_epoch(ledger,row(1,1.),ident['data']['manifest']);cp.save_boundary(directory,model,opt,ident,one)
                torch.save(initial,Path(directory)/'last.pt')
                partial=Path(directory)/'epoch_02.pt.tmp';partial.write_bytes(b'uncommitted evidence')
                step();expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
                restored=cp.restore(directory,model,opt,ident);self.assertEqual(restored['successful_updates'],1)
                self.assertEqual(partial.read_bytes(),b'uncommitted evidence')
                step();self.assertTrue(same_tree(expected,model.state_dict()));self.assertTrue(same_tree(expected_opt,opt.state_dict()))
                two=cp.commit_epoch(one,row(2,2.),ident['data']['manifest']);cp.save_boundary(directory,model,opt,ident,two)
                evaluated=p.fresh_model(variant)
                cp.load_for_evaluation(Path(directory)/'last.pt',evaluated,ident['processor'],source_sha='test-source',manifest=ident['data']['manifest'])
                self.assertTrue(same_tree(model.state_dict(),evaluated.state_dict()))
                with self.assertRaises(ValueError):cp.load_for_evaluation(Path(directory)/'last.pt',evaluated,ident['processor'],source_sha='wrong',manifest=ident['data']['manifest'])
                self.assertEqual(torch.load(Path(directory)/'best.pt',weights_only=False)['ledger']['completed_epochs'],1)
                self.assertEqual(torch.load(Path(directory)/'last.pt',weights_only=False)['ledger']['completed_epochs'],2)
                with self.assertRaises(FileExistsError):cp.save_boundary(directory,model,opt,ident,two)
                with self.assertRaises(ValueError):cp.restore(directory,model,opt,dict(ident,source_sha='wrong'))
                with tempfile.TemporaryDirectory() as other:
                    with self.assertRaises(ValueError):cp.restore(other,model,opt,ident)

    def test_all_eight_immutable_epoch_files_and_earliest_best(self):
        model=p.fresh_model('direct');opt=torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            ident=identity(directory,model,'direct');ledger=cp.empty_ledger();cp.save_boundary(directory,model,opt,ident,ledger)
            for e,score in enumerate((4.,3.,2.,1.,.9,.9,1.,1.2),1):
                ledger=cp.commit_epoch(ledger,row(e,score,[False,False]),ident['data']['manifest'])
                cp.save_boundary(directory,model,opt,ident,ledger)
            self.assertEqual(len(list(Path(directory).glob('epoch_*.pt'))),8)
            self.assertEqual(ledger['visited_train_windows'],16);self.assertEqual(ledger['successful_updates'],0)
            self.assertEqual(cp.restore(directory,model,opt,ident)['best_epoch'],5)
            self.assertEqual(torch.load(Path(directory)/'best.pt',weights_only=False)['ledger']['completed_epochs'],5)

    def test_preflight_entry_never_creates_optimizer_renderer_or_training_run(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/p.BUCKETS['direct']/'run-one';report=Path(directory)/'audit'/'preflight.json'
            argv=['train','--variant','direct','--root','data','--depth-root','depth','--g0-reference','reference',
                  '--output',str(out),'--expected-sha','sha','--preflight-only','--preflight-report',str(report)]
            threads=torch.get_num_threads()
            try:
                with patch.object(sys,'argv',argv),patch.object(entry,'source_preflight',return_value=('sha',{})), \
                     patch.object(entry,'preflight',return_value=({},{})),patch.object(entry,'paired_initialization',return_value={'verified':'fixture'}), \
                     patch.object(torch.optim,'AdamW',side_effect=AssertionError('No optimizer')), \
                     patch.object(entry,'make_renderer',side_effect=AssertionError('No renderer')):
                    entry.main()
            finally:torch.set_num_threads(threads)
            self.assertFalse(out.exists());self.assertTrue(report.exists())
            self.assertFalse(json.loads(report.read_text())['training_started'])

    def test_checkpoint_rejects_wrong_versions_variants_config_and_incomplete_ledgers(self):
        for variant in ('direct','propagate'):
            model=p.fresh_model(variant);opt=torch.optim.AdamW(model.parameters())
            with tempfile.TemporaryDirectory() as directory:
                ident=identity(directory,model,variant);s=cp.save_boundary(directory,model,opt,ident,cp.empty_ledger())
                for version in ('touch4d-g5-direct-architecture-v1','touch4d-g5-propagate-architecture-v1',
                                'touch4d-g4-formal-h15-epochs-v1','touch4d-v4-stage1-smoke-v1',cp.VERSIONS['propagate' if variant=='direct' else 'direct']):
                    with self.assertRaises(ValueError):cp.validate_state(dict(s,version=version),ident)
                altered=copy.deepcopy(s);altered['ledger']['completed_epochs']=1
                with self.assertRaises(ValueError):cp.validate_state(altered,ident)
                altered=copy.deepcopy(s);altered['ledger']=cp.commit_epoch(cp.empty_ledger(),row(1),ident['data']['manifest'])
                altered['ledger']['history'][0]['optimizer_update_flags']=[True,True]
                with self.assertRaises(ValueError):cp.validate_state(altered,ident)
                with self.assertRaises(ValueError):cp.restore(directory,p.fresh_model('propagate' if variant=='direct' else 'direct'),opt,ident)

    def test_read_only_evaluation_uses_original_loss_and_records_signed_regions(self):
        processor=DataProcessor('001-rope')
        for variant in ('direct','propagate'):
            model=p.fresh_model(variant);batch=prepare_batch(first(synthetic_batch(32,15)),processor,model)
            before=copy.deepcopy(model.state_dict())
            with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No optimizer in eval')):
                report,motion,conditions,geometry,prop=window_diagnostics(model,batch,processor,GaussianRenderer(analytic_rasterizer),variant)
            self.assertTrue(same_tree(before,model.state_dict()))
            self.assertEqual({r['horizon'] for r in motion},set(range(1,16)))
            self.assertEqual({r['region'] for r in motion},{'all','near','middle','far'})
            self.assertTrue(all('signed_mean_y_mm' in r and 'pred_region_to_full_gt_mm' in r for r in motion))
            self.assertEqual(len(conditions),30);self.assertTrue(geometry)
            self.assertEqual(bool(prop),variant=='propagate')

    def test_full_evaluate_reuses_episode_equal_loop_and_never_steps(self):
        model=p.fresh_model('direct');processor=DataProcessor('001-rope');c=p.configuration('data','depth','sha','direct')
        raw=first(synthetic_batch(32,15));v3=copy.deepcopy(raw);v3['meta']['episode_id'].fill_(3)
        v6=copy.deepcopy(raw);v6['meta']['episode_id'].fill_(6);v6['inputs']['operation_hands'].fill_(True)
        before=copy.deepcopy(model.state_dict())
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No update in eval')):
            metrics=evaluate_full(model,[v3,v6],processor,GaussianRenderer(analytic_rasterizer),c,'cpu',manifest())
        self.assertTrue(same_tree(before,model.state_dict()));self.assertEqual(metrics['model']['episodes']['3']['samples'],1)

    def test_evaluation_pair_binding_budget_selection_initialization_and_order(self):
        ids={};ledgers={}
        for variant in ('direct','propagate'):
            for selection in ('best','last'):
                key=variant+'-'+selection
                ids[key]=dict(run_id=variant,initialization=dict(g4_shared_hash='shared',direct_mapping_hash='direct'),data=dict(epoch_order_hashes=['same']*8))
                ledgers[key]=dict(completed_epochs=5 if selection=='best' else 8,best_epoch=5,best_score=.1)
        validate_pair_metadata(ids,ledgers)
        bad=copy.deepcopy(ledgers);bad['propagate-last']['completed_epochs']=7
        with self.assertRaises(ValueError):validate_pair_metadata(ids,bad)
        bad=copy.deepcopy(ids);bad['propagate-last']['initialization']['direct_mapping_hash']='wrong'
        with self.assertRaises(ValueError):validate_pair_metadata(bad,ledgers)
        bad=copy.deepcopy(ids);bad['direct-best']['run_id']='another'
        with self.assertRaises(ValueError):validate_pair_metadata(bad,ledgers)


if __name__=='__main__':unittest.main()
