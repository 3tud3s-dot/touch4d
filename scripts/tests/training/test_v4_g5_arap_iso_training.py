"""Original Joint, ARAP/ISO final-GS structure, paired data and one-shot calibration."""
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
from scripts.experiments.v4_g5_arap_iso.protocol import fresh_model,VARIANTS
from scripts.experiments.v4_g5_arap_iso import runner,protocol
from scripts.experiments.v4_g5_arap_iso.structure import build_structure_graph
from scripts.experiments.v4_g5_arap_iso.arap import combined_regularizers
from scripts.experiments.v4_g5_arap_iso.checkpoint import empty_ledger,commit_epoch
from scripts.experiments.v4_g5_arap_iso import smoke


def raster(**args):
    signal=args['means'].mean(0)
    rgb=torch.sigmoid(signal).expand(1,args['height'],args['width'],3)
    alpha=torch.sigmoid(signal.sum()).expand(1,args['height'],args['width'],1)
    return torch.cat((rgb,alpha*.5),-1),alpha,{}


class ArapIsoTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def setUp(self):self.processor=DataProcessor('001-rope')
    def raw(self):return first(synthetic_batch(32,15,1))
    def config(self,variant='g5_arap',weight=.01):
        return protocol.configuration('data','depth','source',variant,weight)
    def validation(self,raw):
        a=copy.deepcopy(raw);a['meta']['episode_id'][0]=3
        b=copy.deepcopy(raw);b['meta']['episode_id'][0]=6;b['inputs']['operation_hands'][:]=True
        return [a,b]

    def test_losses_exact_original_plus_weighted_final_gs_structure_and_gradient(self):
        model=fresh_model('g5_arap');batch=prepare_batch(self.raw(),self.processor,model)
        with torch.no_grad():base=model(batch['inputs'])
        pred={k:v.detach().clone() for k,v in base.items()}
        pred['xyz']=(batch['inputs']['gs_xyz'][:,None].expand_as(pred['xyz'])*2).clone().requires_grad_(True)
        graph=build_structure_graph(batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'])
        config=self.config();renderer=runner.CachedRenderer(GaussianRenderer(raster))
        total,values,parts,actual_graph,statistics=runner.losses(pred,batch,self.processor,renderer,config,graph)
        direct,direct_values=objective(pred,batch,self.processor,renderer,config['depth_weight'],config['chamfer_chunk'],**loss_options(config))
        self.assertIs(actual_graph,graph)
        torch.testing.assert_close(parts['loss_original'],direct,rtol=0,atol=0)
        torch.testing.assert_close(parts['loss_arap_unweighted'],combined_regularizers(pred['xyz'],batch['inputs']['gs_xyz'],graph)[0],rtol=0,atol=0)
        torch.testing.assert_close(total,direct+.01*parts['loss_arap_unweighted'],rtol=0,atol=0)
        self.assertTrue(same_tree(values,direct_values))
        g_total=torch.autograd.grad(total,pred['xyz'],retain_graph=True)[0]
        g_original=torch.autograd.grad(parts['loss_original'],pred['xyz'],retain_graph=True)[0]
        g_struct=torch.autograd.grad(parts['loss_arap_unweighted'],pred['xyz'])[0]
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
                self.assertAlmostEqual(item['loss_total'],item['loss_original']+item['loss_arap_weighted'],places=7)
                self.assertAlmostEqual(item['loss_arap_weighted'],.01*item['loss_arap_unweighted'],places=7)
                self.assertIn('retained_edges',item['graph'])
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_no_render_visit_cannot_execute_adamw_or_increment_success(self):
        model=fresh_model('g5_arap');optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
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
        a=row(1,.1);a['loss_component_batch_means']=dict(loss_total=100.,loss_arap_unweighted=999.)
        b=row(2,.2);b['loss_component_batch_means']=dict(loss_total=.2,loss_arap_unweighted=0.)
        ledger=commit_epoch(ledger,a,data);ledger=commit_epoch(ledger,b,data)
        self.assertEqual(ledger['best_epoch'],1);self.assertEqual(ledger['best_score'],.1)

    def test_common_protocol_nested_structure_copy_and_mutation_rejection(self):
        a=self.config(VARIANTS[0]);b=self.config(VARIANTS[1])
        self.assertEqual(protocol.common_configuration(a),protocol.common_configuration(b))
        self.assertEqual((a['horizon'],a['prediction_interval'],a['epochs']),(15,1,8))
        self.assertEqual((a['chamfer_weight'],a['rgb_weight'],a['alpha_weight'],a['scale_weight']),(1.,1.,.1,0.))
        a['structure']['iso_beta']=.2
        self.assertEqual(b['structure']['iso_beta'],.1)
        self.assertEqual(protocol.STRUCTURE['iso_beta'],.1)
        with self.assertRaises(ValueError):protocol.common_configuration(a)
        for weight in (-1.,.02,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.config(weight=weight)

    def test_eight_loader_orders_match_original_and_model_initialization_does_not_consume_data_rng(self):
        for epoch in range(8):
            gen=torch.Generator().manual_seed(42+epoch)
            raw=DataLoader(range(31),batch_size=1,shuffle=True,num_workers=0,generator=gen)
            before=gen.get_state().clone();fresh_model('g5_arap_iso')
            self.assertTrue(torch.equal(before,gen.get_state()))
            observed=probe_order(raw)
            self.assertTrue(torch.equal(before,gen.get_state()))
            self.assertEqual(observed,epoch_order(31,epoch))
            self.assertEqual(observed,[int(v[0]) for v in raw])

    def calibration(self,arap=(20.,10.),iso=(20.,10.)):
        records=[dict(episode=ep,start=start,joint_gradient_norm=gj,arap_gradient_norm=ga,iso_gradient_norm=gi)
                 for (ep,start),gj,ga,gi in zip(protocol.TRAIN_WINDOWS,(1.,2.),arap,iso)]
        return dict(passed=True,source_sha='source',manifest_hash='manifest',reference_sha256=protocol.REFERENCE_HASH,
            windows=[list(w) for w in protocol.TRAIN_WINDOWS],components={'mock':'component'},rule=protocol.RULE,
            **protocol.calibrated_weights(records),records=records)

    def read_calibration(self,value):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'calibration.json';p.write_text(json.dumps(value),encoding='utf-8')
            with patch.object(protocol,'component_hashes',return_value={'mock':'component'}):return protocol.calibration_read(p)

    def test_once_calibration_independent_caps_downward_only_and_zero_failure(self):
        d=self.calibration();actual=self.read_calibration(d)
        self.assertEqual(actual['lambda_arap'],.0025);self.assertEqual(actual['lambda_iso'],.0005)
        small=self.calibration((.1,.2),(.1,.2));self.assertEqual(self.read_calibration(small)['lambda_arap'],.01)
        self.assertEqual(self.read_calibration(small)['lambda_iso'],.001)
        for key in ('joint_gradient_norm','arap_gradient_norm','iso_gradient_norm'):
            for value in (0.,-1.,float('nan'),float('inf')):
                wrong=copy.deepcopy(d);wrong['records'][0][key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.read_calibration(wrong)
        wrong=copy.deepcopy(d);wrong['lambda_arap']=.01
        with self.assertRaises(ValueError):self.read_calibration(wrong)
        wrong=copy.deepcopy(d);wrong['lambda_iso']=.001
        with self.assertRaises(ValueError):self.read_calibration(wrong)

    def test_calibration_wrong_windows_components_reference_and_rule_rejected(self):
        d=self.calibration()
        for key,value in (('windows',[[3,135],[7,105]]),('components',{'wrong':'x'}),('reference_sha256','0'*64),('rule','search')):
            with self.subTest(key=key),self.assertRaises(ValueError):self.read_calibration(dict(d,**{key:value}))
        wrong=copy.deepcopy(d);wrong['records']=wrong['records'][:1]
        with self.assertRaises(ValueError):self.read_calibration(wrong)

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

    def test_smoke_learning_guard_rejects_weight_decay_only(self):
        groups={key:dict(clipped_gradient_norm=1.,actual_update_norm=1.) for key in ('motion','cross','conditions','global_motion','local_motion')}
        row=dict(successful_update=3,groups=copy.deepcopy(groups),arap_xyz_gradient_norm=1.,iso_xyz_gradient_norm=1.);smoke._assert_groups_learned([row])
        row['groups']['cross']['clipped_gradient_norm']=0
        with self.assertRaises(AssertionError):smoke._assert_groups_learned([row])


    def test_fresh_C_D_and_original_G5_full_h15_are_bit_equal(self):
        from scripts.experiments.v4_g5_signed_local.protocol import fresh_model as original_fresh
        a=protocol.fresh_model(protocol.VARIANTS[0]);b=protocol.fresh_model(protocol.VARIANTS[1]);original=original_fresh('direct')
        self.assertEqual(sum(p.numel() for p in a.parameters()),620511)
        self.assertEqual(state_hash(a.state_dict()),state_hash(b.state_dict()))
        self.assertEqual(state_hash(a.state_dict()),state_hash(original.state_dict()))
        self.assertEqual(a.config,original.config)
        x=prepare_batch(self.raw(),self.processor,a)['inputs']
        with torch.no_grad():pa,pb,po=a(x),b(x),original(x)
        self.assertTrue(same_tree(pa,pb));self.assertTrue(same_tree(pa,po))
        runner.frozen_attributes(pa,x)

    def test_D_total_and_xyz_gradient_include_ISO_while_C_excludes_it(self):
        model=fresh_model('g5_arap_iso');batch=prepare_batch(self.raw(),self.processor,model)
        with torch.no_grad():pred=model(batch['inputs'])
        h=torch.linspace(.01,.3,15).reshape(1,15,1,1)
        pred['xyz']=(batch['inputs']['gs_xyz'][:,None]*(1+h)).detach().requires_grad_(True)
        total,values,parts,graph,stats=runner.losses(pred,batch,self.processor,runner.CachedRenderer(GaussianRenderer(raster)),self.config('g5_arap_iso'))
        expected=parts['loss_original']+.01*parts['loss_arap_unweighted']+.001*parts['loss_iso_unweighted']
        torch.testing.assert_close(total,expected,rtol=0,atol=0)
        self.assertGreater(float(parts['loss_iso_unweighted'].detach()),0.)
        g=torch.autograd.grad(total,pred['xyz'],retain_graph=True)[0]
        go=torch.autograd.grad(parts['loss_original'],pred['xyz'],retain_graph=True)[0]
        ga=torch.autograd.grad(parts['loss_arap_unweighted'],pred['xyz'],retain_graph=True)[0]
        gi=torch.autograd.grad(parts['loss_iso_unweighted'],pred['xyz'])[0]
        torch.testing.assert_close(g,go+.01*ga+.001*gi,rtol=1e-5,atol=1e-6)
        sentinel=pred['xyz'].detach().clone().requires_grad_(True)
        with patch.object(runner,'combined_regularizers',return_value=(sentinel.sum(),sentinel.square().sum(),{})):
            ctotal,_,cp,_,_=runner.losses(pred,batch,self.processor,runner.CachedRenderer(GaussianRenderer(raster)),self.config('g5_arap'))
            gradient=torch.autograd.grad(ctotal,sentinel)[0]
            torch.testing.assert_close(gradient,torch.full_like(sentinel,.01),rtol=0,atol=0)
            self.assertFalse(cp['loss_iso_weighted'].requires_grad)

    def test_gate_binds_source_manifest_orders_initial_and_two_four_update_proofs(self):
        from scripts.experiments.v4_g5_signed_local.diagnostics import sha256
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cal=root/'calibration.json';cal.write_text('fixed calibration')
            data=dict(manifest_hash='manifest',epoch_order_hashes=['order']*8);initial=dict(variant_model_hashes={'a':'hash'})
            smokeproof={}
            for variant in protocol.VARIANTS:
                path=root/(variant+'.json')
                path.write_text(json.dumps(dict(passed=True,successful_updates=4,variant=variant,source_sha='source',calibration_sha256=sha256(cal),manifest_hash=data['manifest_hash'],epoch_order_hashes=data['epoch_order_hashes'],initialization=initial)))
                smokeproof[variant]=dict(completion_path=str(path),completion_sha256=sha256(path))
            gate=dict(passed=True,source_sha='source',calibration_sha256=sha256(cal),manifest_hash='manifest',
                epoch_order_hashes=['order']*8,initialization=initial,smoke=smokeproof)
            target=root/'gate.json';target.write_text(json.dumps(gate))
            with patch.object(protocol,'calibration_read',return_value=dict(source_sha='source',manifest_hash='manifest')),patch.object(protocol,'_reviewed_graph_gate'):
                self.assertTrue(protocol.gate_read(root,cal,'source',data,initial)['passed'])
                for key,value in (('source_sha','other'),('manifest_hash','changed'),('epoch_order_hashes',['wrong']*8),('initialization',{})):
                    target.write_text(json.dumps(dict(gate,**{key:value})))
                    with self.subTest(key=key),self.assertRaises(ValueError):protocol.gate_read(root,cal,'source',data,initial)
                target.write_text(json.dumps(gate))
                p=Path(smokeproof['g5_arap']['completion_path']);proof=json.loads(p.read_text());proof['successful_updates']=3;p.write_text(json.dumps(proof))
                gate['smoke']['g5_arap']['completion_sha256']=sha256(p);target.write_text(json.dumps(gate))
                with self.assertRaises(ValueError):protocol.gate_read(root,cal,'source',data,initial)

    def test_resource_guard_requires_independent_owned_running_Slurm_step(self):
        def command(args,**kwargs):
            if args[:3]==['scontrol','show','job']:return 'JobId=33333 UserId=tcweng(1025) JobState=RUNNING'
            if args[:3]==['scontrol','show','step']:return 'StepId=33333.0 State=RUNNING'
            if '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free' in args:return '0, GPU-00000000-0000-0000-0000-000000000001, NVIDIA GeForce RTX 5090, 32768, 0, 32768'
            if '--query-compute-apps=pid' in args:return ''
            raise AssertionError(args)
        with patch.dict(os.environ,{'SLURM_JOB_ID':'33333','SLURM_STEP_ID':'0','CUDA_VISIBLE_DEVICES':'0'}),patch.object(protocol.os,'getuid',return_value=1025,create=True),patch.object(protocol.subprocess,'check_output',side_effect=command),patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'get_device_properties',return_value=type('GPU',(),{'uuid':'00000000-0000-0000-0000-000000000001'})()):
            result=protocol.resource_guard(require_idle=True)
            self.assertEqual(result['job_id'],'33333');self.assertEqual(result['step_id'],'0');self.assertGreaterEqual(result['reserve_mib'],4096)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'32087'}):
                with self.assertRaises(RuntimeError):protocol.resource_guard()
            with patch.dict(os.environ,{'SLURM_STEP_ID':''}):
                with self.assertRaises(RuntimeError):protocol.resource_guard()
            with patch.object(protocol.os,'getuid',return_value=999):
                with self.assertRaises(RuntimeError):protocol.resource_guard()


    def test_resource_mapping_uses_cuda_UUID_or_step_GPU_not_visible_logical_zero(self):
        import types
        rows='0, GPU-00000000-0000-0000-0000-000000000001, NVIDIA GeForce RTX 5090, 32768, 0, 32768\n5, GPU-00000000-0000-0000-0000-000000000005, NVIDIA GeForce RTX 5090, 32768, 0, 32768'
        def command(args,**kwargs):
            if args[:3]==['scontrol','show','job']:return 'JobId=33333 UserId=tcweng(1025) JobState=RUNNING'
            if args[:3]==['scontrol','show','step']:return 'StepId=33333.0 State=RUNNING'
            if '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free' in args:return rows
            if '--query-compute-apps=pid' in args:
                self.assertIn('--id=GPU-00000000-0000-0000-0000-000000000005',args);return ''
            raise AssertionError(args)
        env={'SLURM_JOB_ID':'33333','SLURM_STEP_ID':'0','CUDA_VISIBLE_DEVICES':'0','SLURM_STEP_GPUS':'5'}
        with patch.dict(os.environ,env),patch.object(protocol.os,'getuid',return_value=1025,create=True),patch.object(protocol.subprocess,'check_output',side_effect=command),patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'get_device_properties',return_value=types.SimpleNamespace(uuid='00000000-0000-0000-0000-000000000005')):
            result=protocol.resource_guard();self.assertEqual(result['physical_index'],'5')
            self.assertEqual(result['mapping'],'authoritative_cuda_device_uuid')
            with patch.object(torch.cuda,'get_device_properties',side_effect=AssertionError('Parent must not initialize CUDA')):
                result=protocol.resource_guard(inspect_cuda=False);self.assertEqual(result['physical_index'],'5')
                with patch.dict(os.environ,{'SLURM_STEP_GPUS':''}):
                    with self.assertRaises(RuntimeError):protocol.resource_guard(inspect_cuda=False)

    def test_solver_fallback_stops_before_any_optimizer_and_retains_exact_matrices(self):
        model=fresh_model('g5_arap');batch=prepare_batch(self.raw(),self.processor,model)
        with torch.no_grad():pred=model(batch['inputs'])
        graph=build_structure_graph(batch['inputs']['gs_xyz'],batch['inputs']['gs_mask'])
        stats=dict(kabsch_fallback_count=torch.ones((1,15),dtype=torch.long),solver_failures=dict(
            indices=torch.tensor([[0,2,3]]),covariance=torch.eye(3)[None],rotation=torch.eye(3)[None],singular_values=torch.tensor([[1.,0.,0.]])))
        with patch.object(runner,'combined_regularizers',return_value=(pred['xyz'].sum()*0,pred['xyz'].sum()*0,stats)):
            with self.assertRaises(runner.SolverFallbackError) as caught:
                runner.losses(pred,batch,self.processor,runner.CachedRenderer(GaussianRenderer(raster)),self.config(),graph)
        self.assertEqual(caught.exception.diagnostics['solver_failures']['indices'],[[0,2,3]])
        self.assertEqual(caught.exception.diagnostics['solver_failures']['covariance'][0],torch.eye(3).tolist())

    def test_smoke_existing_output_is_not_modified_when_refused(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sentinel=root/'historical.txt';sentinel.write_text('preserve')
            args=Namespace(output=str(root),expected_sha='source',variant='g5_arap')
            with patch.object(smoke.argparse.ArgumentParser,'parse_args',return_value=args),patch.object(smoke,'run',side_effect=FileExistsError('Owned history')):
                with self.assertRaises(FileExistsError):smoke.main()
            self.assertEqual([p.name for p in root.iterdir()],['historical.txt']);self.assertEqual(sentinel.read_text(),'preserve')

    def test_smoke_regularizers_must_activate_late_and_not_only_have_finite_zeros(self):
        groups={key:dict(clipped_gradient_norm=1.,actual_update_norm=1.) for key in ('motion','cross','conditions','global_motion','local_motion')}
        rows=[dict(successful_update=h,groups=copy.deepcopy(groups),arap_xyz_gradient_norm=0.,iso_xyz_gradient_norm=0.) for h in range(1,5)]
        with self.assertRaises(AssertionError):smoke._assert_groups_learned(rows)
        rows[-1]['arap_xyz_gradient_norm']=1.;rows[-1]['iso_xyz_gradient_norm']=1.
        smoke._assert_groups_learned(rows)
        rows[-1]['iso_xyz_gradient_norm']=float('nan')
        with self.assertRaises(AssertionError):smoke._assert_groups_learned(rows)


    def test_failed_solver_nonfinite_evidence_is_JSON_safe_without_relaxing_losses(self):
        stats=dict(kabsch_fallback_count=torch.tensor([[1]]),solver_failures=dict(
            indices=torch.tensor([[0,0,2]]),covariance=torch.full((1,3,3),float('nan')),
            rotation=torch.eye(3)[None],singular_values=torch.tensor([[float('inf'),float('-inf'),float('nan')]])))
        error=runner.SolverFallbackError(stats)
        encoded=json.dumps(error.diagnostics,allow_nan=False)
        self.assertIn('NaN',encoded);self.assertIn('+Inf',encoded);self.assertIn('-Inf',encoded)
        self.assertEqual(error.diagnostics['solver_failures']['indices'],[[0,0,2]])

    def test_context_free_single_NVML_namespace_requires_single_GPU_Slurm_proof(self):
        def command(args,**kwargs):
            if args[:3]==['scontrol','show','job']:return 'JobId=33333 UserId=tcweng(1025) JobState=RUNNING AllocTRES=cpu=4,gres/gpu=1,gres/gpu:5090=1'
            if args[:3]==['scontrol','show','step']:return 'StepId=33333.0 State=RUNNING'
            if '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free' in args:return '0, GPU-00000000-0000-0000-0000-000000000005, NVIDIA GeForce RTX 5090, 32768, 0, 32768'
            if '--query-compute-apps=pid' in args:return ''
            raise AssertionError(args)
        with patch.dict(os.environ,{'SLURM_JOB_ID':'33333','SLURM_STEP_ID':'0','CUDA_VISIBLE_DEVICES':'0','SLURM_STEP_GPUS':'5'}),patch.object(protocol.os,'getuid',return_value=1025,create=True),patch.object(protocol.subprocess,'check_output',side_effect=command),patch.object(torch.cuda,'get_device_properties',side_effect=AssertionError('No parent CUDA context')):
            result=protocol.resource_guard(inspect_cuda=False)
            self.assertEqual(result['mapping'],'single_nvml_device_namespace_job_gpu_count_one')
            self.assertEqual(result['gpu_uuid'],'GPU-00000000-0000-0000-0000-000000000005')


if __name__=='__main__':unittest.main()
