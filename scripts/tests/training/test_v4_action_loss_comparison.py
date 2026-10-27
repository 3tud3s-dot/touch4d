import copy
from pathlib import Path
import tempfile
import unittest
import json
import hashlib
from unittest.mock import patch
import torch
from scripts.experiments.v4_action_loss_comparison.run import mse_by_horizon,loss_samples,update
from scripts.experiments.v4_action_identifiability.run import update as original_update
from scripts.tests.training.test_v4_action_identifiability import fixture
from scripts.experiments.v4_action_identifiability.data import normalize
from scripts.training.v4_runtime import build_model,save_checkpoint,load_checkpoint


class LossComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_sum_xyz_and_equal_branch_time_means_padding_nan_excluded(self):
        keep=torch.tensor([[True,False],[True,True]])
        xyz=torch.zeros(2,2,2,3,requires_grad=True)
        target=torch.ones_like(xyz);target[0,:,1]=float('nan')
        batch={'inputs':{'gs_mask':keep},'targets':{'target_xyz':[target[:,h] for h in range(2)],'target_mask':[keep,keep]}}
        v=mse_by_horizon(xyz,batch)
        torch.testing.assert_close(v,torch.full((2,2),3.))
        v.mean().backward();self.assertTrue(torch.isfinite(xyz.grad).all());self.assertTrue((xyz.grad[0,:,1]==0).all())
        batch['targets']['target_mask'][1]=~keep
        with self.assertRaises(ValueError):mse_by_horizon(xyz,batch)

    def test_no_nearest_neighbor_reassignment(self):
        xyz=torch.tensor([[[[0.,0.,0.],[1.,0.,0.]]]])
        keep=torch.ones(1,2,dtype=torch.bool)
        batch={'inputs':{'gs_mask':keep},'targets':{'target_xyz':[xyz[:,0].flip(1)],'target_mask':[keep]}}
        torch.testing.assert_close(mse_by_horizon(xyz,batch),torch.ones(1,1))

    def test_cd_optimizer_step_matches_v1_exactly(self):
        raw,*_=fixture();torch.manual_seed(42);m=build_model('v4_stage1');other=copy.deepcopy(m)
        batch,p=normalize(raw,m);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
        oo=torch.optim.AdamW(other.parameters(),lr=1e-4,weight_decay=1e-4)
        for _ in range(2):
            a=update(m,opt,batch,'cd');b=original_update(other,oo,batch)
            self.assertEqual(a['loss'],b['loss']);self.assertEqual(a['gradient_norm'],b['gradient_norm'])
            for k,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)

    def test_mse_backward_and_checkpoint_restoration(self):
        raw,*_=fixture();torch.manual_seed(42);m=build_model('v4_stage1');batch,p=normalize(raw,m)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
        rows=[update(m,opt,batch,'mse') for _ in range(3)]
        for group in ('controller','attention_score','motion','gs_encoder'):
            self.assertTrue(any(r['raw_gradients'][group]['gradient_norm']>0 for r in rows[1:]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mse.pt'
            save_checkpoint(path,m,opt,3,source_sha='test',processor_state=p.state_dict(),run_state={'diagnostic_loss':'mse','multiplier':1})
            clone=build_model('v4_stage1');co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4)
            load_checkpoint(path,clone,co,processor_state=p.state_dict())
            update(m,opt,batch,'mse');update(clone,co,batch,'mse')
            for k,v in m.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[k],rtol=0,atol=0)

    def test_initial_identity_and_correspondence_metrics(self):
        raw,w,a,meta=fixture();m=build_model('v4_stage1');batch,p=normalize(raw,m)
        pred=m(batch['inputs'])
        mse=mse_by_horizon(pred['xyz'],batch)
        self.assertEqual(mse.shape,(2,10));torch.testing.assert_close(mse[:,0],torch.zeros(2))
        self.assertTrue((mse[:,1:]>0).all())
        expected=torch.stack([(pred['xyz'][:,h]-t).square().sum(-1).mean(-1) for h,t in enumerate(batch['targets']['target_xyz'])],1)
        torch.testing.assert_close(mse,expected)

    def test_report_charts_and_common_bounds(self):
        import numpy as np
        from scripts.experiments.v4_action_loss_comparison.report import chart,projection
        x=np.array([[0.,0.,0.],[1.,1.,1.]])
        self.assertIn('polyline',chart('MSE',[('MSE',[(0,1.),(400,.1)],'red')]))
        self.assertIn('<svg',projection(x,x,x,'same',(0,2),(x.min(0),x.max(0))))

    def test_frozen_identity_rejects_tampering(self):
        from scripts.experiments.v4_action_loss_comparison.run import read_frozen,BASE
        from scripts.experiments.fixed_window.fixed_fit import digest
        raw,w,a,manifest=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'conditioned').mkdir()
            torch.save(dict(raw=raw,weights=w,affected=a),root/'frozen_data.pt')
            (root/'data_manifest.json').write_text(json.dumps(manifest))
            (root/'config.json').write_text(json.dumps({'source_sha':BASE}))
            (root/'conditioned/metrics.json').write_text(json.dumps({'initial_parameter_sha256':'fixture','checkpoints':{}}))
            actual=hashlib.sha256((root/'frozen_data.pt').read_bytes()).hexdigest()
            with patch('scripts.experiments.v4_action_loss_comparison.run.FROZEN_SHA',actual):
                frozen,ids=read_frozen(root)
                self.assertEqual(ids['target_sha256'],digest(raw['targets']))
                manifest['data_sha256']='changed';(root/'data_manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):read_frozen(root)

    def test_complete_report_fixture(self):
        from scripts.experiments.v4_action_loss_comparison.run import measure,export_npz
        from scripts.experiments.v4_action_loss_comparison.report import write_report
        raw,w,a,_=fixture();m=build_model('v4_stage1');batch,p=normalize(raw,m)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results={}
            for mode in ('cd','mse'):
                (root/mode).mkdir();row,pred=measure(m,batch,p,a,mode);row['update']=0
                results[mode]=dict(best_update=0,evaluations=[row],updates=[dict(raw_loss=0.,gradient_norm=0.,actual_updates={'motion':{'update_norm':0.}},clipped=False)],probes={'best':dict(update=0,conditions={'static':row})})
                for label in ('best','last'):export_npz(root/mode/f'{label}_predictions.npz',pred,batch,p)
            (root/'results.json').write_text(json.dumps(results));(root/'config.json').write_text(json.dumps({'source_sha':'fixture','job_id':'fixture'}))
            write_report(root)
            self.assertEqual(len(list((root/'figures').glob('*.html'))),8)
            self.assertIn('全部10帧',(root/'report.html').read_text(encoding='utf-8'))
