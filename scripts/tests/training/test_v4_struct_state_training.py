"""Original Joint, weak final-GS structure, paired data and one-shot calibration."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from scripts.data_process import DataProcessor
from scripts.training.rendering import GaussianRenderer
from scripts.training.v4_runtime import prepare_batch
from scripts.training.losses import objective
from scripts.training.train import loss_options
from scripts.experiments.v4_stage1.fixtures import synthetic_batch
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from scripts.experiments.v4_g5_signed_local.diagnostics import state_hash
from scripts.experiments.v4_g5_signed_local.protocol import probe_order
from scripts.experiments.v4_g4_global_motion.protocol import epoch_order
from scripts.tests.training.test_v4_g5_training import first,row,manifest
from scripts.experiments.v4_g5_struct_state.model import fresh_model,VARIANTS
from scripts.experiments.v4_g5_struct_state import runner,protocol
from scripts.experiments.v4_g5_struct_state.structure import build_structure_graph,structure_loss
from scripts.experiments.v4_g5_struct_state.checkpoint import empty_ledger,commit_epoch
from scripts.experiments.v4_g5_struct_state import smoke


def raster(**args):
    signal=args['means'].mean(0)
    rgb=torch.sigmoid(signal).expand(1,args['height'],args['width'],3)
    alpha=torch.sigmoid(signal.sum()).expand(1,args['height'],args['width'],1)
    return torch.cat((rgb,alpha*.5),-1),alpha,{}


class StructStateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):self.processor=DataProcessor('001-rope')
    def raw(self):return first(synthetic_batch(32,15,1))
    def config(self,variant='g5_struct',weight=.01):
        return protocol.configuration('data','depth','source',variant,weight)
    def validation(self,raw):
        a=copy.deepcopy(raw);a['meta']['episode_id'][0]=3
        b=copy.deepcopy(raw);b['meta']['episode_id'][0]=6;b['inputs']['operation_hands'][:]=True
        return [a,b]

    def test_losses_exact_original_plus_weighted_final_gs_structure_and_gradient(self):
        model=fresh_model('g5_struct');batch=prepare_batch(self.raw(),self.processor,model)
        with torch.no_grad():base=model(batch['inputs'])
        pred={k:v.detach().clone() for k,v in base.items()}
        pred['xyz']=(batch['inputs']['gs_xyz'][:,None].expand_as(pred['xyz'])*2).clone().requires_grad_(True)
        graph=build_structure_graph(batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'])
        config=self.config();renderer=runner.CachedRenderer(GaussianRenderer(raster))
        total,values,parts,actual_graph=runner.losses(pred,batch,self.processor,renderer,config,graph)
        direct,direct_values=objective(pred,batch,self.processor,renderer,config['depth_weight'],config['chamfer_chunk'],**loss_options(config))
        self.assertIs(actual_graph,graph)
        torch.testing.assert_close(parts['loss_original'],direct,rtol=0,atol=0)
        torch.testing.assert_close(parts['loss_struct_unweighted'],structure_loss(pred['xyz'],graph),rtol=0,atol=0)
        self.assertAlmostEqual(float(parts['loss_struct_unweighted'].detach()),.75,places=5)
        torch.testing.assert_close(total,direct+.01*parts['loss_struct_unweighted'],rtol=0,atol=0)
        self.assertTrue(same_tree(values,direct_values))
        g_total=torch.autograd.grad(total,pred['xyz'],retain_graph=True)[0]
        g_original=torch.autograd.grad(parts['loss_original'],pred['xyz'],retain_graph=True)[0]
        g_struct=torch.autograd.grad(parts['loss_struct_unweighted'],pred['xyz'])[0]
        torch.testing.assert_close(g_total,g_original+.01*g_struct,rtol=1e-5,atol=1e-6)
        self.assertTrue(torch.isfinite(g_total).all())

    def test_real_runner_both_variants_records_visits_flags_losses_and_updates(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model=fresh_model(variant);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
                raw=self.raw();record=[];validation=[];step=[];pending=[]
                before=state_hash(model.state_dict())
                result=runner.perform_epoch(model,optimizer,self.processor,runner.CachedRenderer(GaussianRenderer(raster)),
                    self.config(variant),'cpu',[raw],self.validation(raw),[[2,73]],[[3,73],[6,73]],
                    record=record.append,validation_record=validation.append,
                    step_observer=lambda *args:step.append(args[0]),pending=lambda *args:pending.append(args))
                self.assertEqual(result['successful_updates'],1)
                self.assertEqual(result['optimizer_update_flags'],[True])
                self.assertEqual(result['window_order'],[[2,73]])
                self.assertEqual(result['validation_window_order'],[[3,73],[6,73]])
                self.assertEqual((result['visited_train_windows'],result['visited_val_windows']),(1,2))
                self.assertNotEqual(before,state_hash(model.state_dict()))
                self.assertEqual(step,[1]);self.assertEqual(pending,[(1,0,True),(1,1,False)])
                self.assertEqual(len(record),1);self.assertEqual(len(validation),2)
                item=record[0]
                self.assertAlmostEqual(item['loss_total'],item['loss_original']+item['loss_struct_weighted'],places=7)
                self.assertAlmostEqual(item['loss_struct_weighted'],.01*item['loss_struct_unweighted'],places=7)
                self.assertIn('retained_edges',item['graph'])
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_no_render_visit_cannot_execute_adamw_or_increment_success(self):
        model=fresh_model('g5_struct');optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        raw=self.raw();missing=copy.deepcopy(raw)
        missing['targets']['target_depth_valid'][:]=False;missing['targets']['target_depth'][:]=0
        hashes=[];items=[]
        def record(item):hashes.append(state_hash(model.state_dict()));items.append(item)
        result=runner.train_epoch(model,optimizer,[raw,missing],self.processor,runner.CachedRenderer(GaussianRenderer(raster)),
            self.config(),'cpu',[[2,73],[2,73]],record=record)
        self.assertEqual(result['successful_updates'],1)
        self.assertEqual(result['optimizer_update_flags'],[True,False])
        self.assertEqual(result['visited_train_windows'],2)
        self.assertEqual(hashes[0],hashes[1])
        self.assertEqual([r['successful_update'] for r in items],[True,False])

    def test_validation_is_original_joint_only_readonly_and_attributes_initial(self):
        for variant in VARIANTS:
            model=fresh_model(variant);raw=self.raw();vals=self.validation(raw)
            with torch.no_grad():model.motion[-1].weight.normal_(std=.003)
            before=state_hash(model.state_dict());record=[];structures=[];grad_modes=[]
            def hook(module,args,output):
                grad_modes.append(torch.is_grad_enabled())
                self.assertFalse(module.training)
                runner.frozen_attributes(output,args[0])
            capture=model.register_forward_hook(hook)
            try:
                first_summary,order=runner.evaluate_epoch(model,vals,self.processor,runner.CachedRenderer(GaussianRenderer(raster)),
                    self.config(variant,.01),'cpu',[[3,73],[6,73]],record=record.append,structure_record=structures.append)
                zero_summary,_=runner.evaluate_epoch(model,vals,self.processor,runner.CachedRenderer(GaussianRenderer(raster)),
                    self.config(variant,0.),'cpu',[[3,73],[6,73]])
            finally:capture.remove()
            self.assertTrue(same_tree(first_summary,zero_summary))
            self.assertEqual(order,[[3,73],[6,73]])
            self.assertEqual(before,state_hash(model.state_dict()))
            self.assertTrue(model.training)
            self.assertEqual(grad_modes,[False]*4)
            self.assertEqual(len(record),2);self.assertGreater(len(structures),0)
            val=first_summary['model']
            expected=sum(r['supervised_geometry']+r['rgb']+.1*r['alpha'] for r in val['episodes'].values())/2
            self.assertAlmostEqual(val['objective'],expected,places=12)

    def test_best_selection_ignores_structure_total_and_uses_original_joint(self):
        ledger=empty_ledger();data=manifest()
        a=row(1,.1);a['loss_component_batch_means']=dict(loss_total=100.,loss_struct_unweighted=999.)
        b=row(2,.2);b['loss_component_batch_means']=dict(loss_total=.2,loss_struct_unweighted=0.)
        ledger=commit_epoch(ledger,a,data);ledger=commit_epoch(ledger,b,data)
        self.assertEqual(ledger['best_epoch'],1);self.assertEqual(ledger['best_score'],.1)

    def test_common_protocol_nested_structure_copy_and_mutation_rejection(self):
        a=self.config(VARIANTS[0]);b=self.config(VARIANTS[1])
        self.assertEqual(protocol.common_configuration(a),protocol.common_configuration(b))
        self.assertEqual((a['horizon'],a['prediction_interval'],a['epochs']),(15,1,8))
        self.assertEqual((a['chamfer_weight'],a['rgb_weight'],a['alpha_weight'],a['scale_weight']),(1.,1.,.1,0.))
        a['structure']['ratio_min']=.9
        self.assertEqual(b['structure']['ratio_min'],.8)
        self.assertEqual(protocol.STRUCTURE['ratio_min'],.8)
        with self.assertRaises(ValueError):protocol.common_configuration(a)
        for weight in (-1.,.02,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.config(weight=weight)

    def test_eight_loader_orders_match_original_and_extra_state_does_not_consume_data_rng(self):
        for epoch in range(8):
            gen=torch.Generator().manual_seed(42+epoch)
            raw=DataLoader(range(31),batch_size=1,shuffle=True,num_workers=0,generator=gen)
            before=gen.get_state().clone();fresh_model('g6_state_struct')
            self.assertTrue(torch.equal(before,gen.get_state()))
            observed=probe_order(raw)
            self.assertTrue(torch.equal(before,gen.get_state()))
            self.assertEqual(observed,epoch_order(31,epoch))
            self.assertEqual(observed,[int(v[0]) for v in raw])

    def calibration(self,joint=(1.,2.),struct=(20.,10.),weight=.0025):
        return dict(passed=True,reference_sha256=protocol.REFERENCE_HASH,
            windows=[list(w) for w in protocol.TRAIN_WINDOWS],components={'mock':'component'},
            rule='candidate0.01_downward_once_max_two_training_xyz_gradient_ratios_5percent',lambda_struct=weight,
            records=[dict(episode=ep,start=start,joint_gradient_norm=gj,struct_gradient_norm=gs)
                     for (ep,start),gj,gs in zip(protocol.TRAIN_WINDOWS,joint,struct)])

    def read_calibration(self,value):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'calibration.json';p.write_text(json.dumps(value),encoding='utf-8')
            with patch.object(protocol,'component_hashes',return_value={'mock':'component'}):
                return protocol.calibration_read(p)

    def test_once_calibration_recomputes_exact_two_norm_records_and_only_downscales(self):
        d=self.calibration();self.assertEqual(self.read_calibration(d)['lambda_struct'],.0025)
        for joint,struct in (((1.,1.),(0.,0.)),((10.,20.),(.1,.2)),((0.,0.),(0.,0.))):
            self.assertEqual(self.read_calibration(self.calibration(joint,struct,.01))['lambda_struct'],.01)
        self.assertEqual(self.read_calibration(self.calibration((0.,1.),(1.,1.),0.))['lambda_struct'],0.)
        wrong=self.calibration(weight=.01)
        with self.assertRaises(ValueError):self.read_calibration(wrong)
        wrong=self.calibration(weight=.001)
        with self.assertRaises(ValueError):self.read_calibration(wrong)

    def test_calibration_wrong_windows_components_reference_or_nonfinite_norms_rejected(self):
        original=self.calibration()
        cases=[]
        for key,value in (('windows',[[3,135],[7,105]]),('components',{'wrong':'x'}),('reference_sha256','0'*64),('rule','search')):
            cases.append(dict(original,**{key:value}))
        for value in (-1.,float('nan'),float('inf')):
            wrong=copy.deepcopy(original);wrong['records'][0]['struct_gradient_norm']=value;cases.append(wrong)
        wrong=copy.deepcopy(original);wrong['records']=wrong['records'][:1];cases.append(wrong)
        wrong=copy.deepcopy(original);wrong['records'][0]['episode']=3;cases.append(wrong)
        for case in cases:
            with self.assertRaises(ValueError):self.read_calibration(case)

    def test_smoke_snapshot_full_optimizer_strict_restore_and_reject_other_variant(self):
        for variant in VARIANTS:
            model=fresh_model(variant);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            identity=smoke.checkpoint_identity(model,self.processor,'cpu-source','c'*64)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'smoke.pt'
                for _ in range(4):
                    opt.zero_grad(set_to_none=True)
                    sum(p.square().sum() for p in model.parameters()).backward();opt.step()
                expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
                saved=smoke.save_smoke(path,model,opt,identity,4)
                with torch.no_grad():next(model.parameters()).add_(.1)
                self.assertTrue(same_tree(saved['model'],expected))
                smoke.load_smoke(path,model,opt,identity)
                self.assertTrue(same_tree(expected,model.state_dict()))
                self.assertTrue(same_tree(expected_opt,opt.state_dict()))
                with self.assertRaises(ValueError):smoke.load_smoke(path,model,opt,dict(identity,source_sha='wrong'))
                other=fresh_model(VARIANTS[1] if variant==VARIANTS[0] else VARIANTS[0])
                other_opt=torch.optim.AdamW(other.parameters(),lr=1e-4,weight_decay=1e-4)
                with self.assertRaises(ValueError):smoke.load_smoke(path,other,other_opt,identity)
                with self.assertRaises(ValueError):smoke.save_smoke(Path(d)/'partial.pt',model,opt,identity,2)

    def test_probe_peer_registry_and_live_graph_barrier_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'a';out.mkdir();barrier=root/'barrier';barrier.mkdir()
            registry=dict(source_sha='source',parent_pid=os.getppid(),pids=[os.getpid(),os.getpid()+1],
                outputs={'g5_struct':str(out.resolve())},barrier=str(barrier.resolve()))
            path=root/'peers.json';path.write_text(json.dumps(registry))
            self.assertEqual(smoke.wait_for_peers(path,'source'),registry)
            (barrier/'go.json').write_text(json.dumps(dict(source_sha='source',parent_pid=os.getppid(),go=True)))
            released=smoke.probe_barrier(registry,out,'g5_struct','source',dict(graph_live=True))
            self.assertTrue(released['go'])
            ready=json.loads((barrier/'g5_struct-ready.json').read_text())
            self.assertEqual(ready['pid'],os.getpid());self.assertTrue(ready['graph_live'])
            path.write_text(json.dumps(dict(registry,source_sha='wrong')))
            with self.assertRaises(ValueError):smoke.wait_for_peers(path,'source')
            path.write_text(json.dumps(dict(registry,parent_pid=os.getppid()+1)))
            with self.assertRaises(ValueError):smoke.wait_for_peers(path,'source')

    def test_smoke_state_learning_guard_does_not_accept_weight_decay_alone(self):
        groups={key:dict(clipped_gradient_norm=1.,actual_update_norm=1.)
                for key in ('motion','cross','conditions','global_motion','local_motion','state_readout','gru','state_encoder')}
        first=dict(successful_update=1,groups=copy.deepcopy(groups))
        first['groups']['gru']['clipped_gradient_norm']=0
        first['groups']['state_encoder']['clipped_gradient_norm']=0
        with self.assertRaises(AssertionError):smoke._assert_groups_learned([first],True)
        later=dict(successful_update=2,groups=copy.deepcopy(groups))
        smoke._assert_groups_learned([first,later],True)
        smoke._assert_groups_learned([first],False)


if __name__=='__main__':unittest.main()
