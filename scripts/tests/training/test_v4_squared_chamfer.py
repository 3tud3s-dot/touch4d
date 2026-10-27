import copy
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.experiments.v4_squared_chamfer.loss import squared_chamfer_samples
from scripts.experiments.v4_squared_chamfer.run import diagnostic_adapter
from scripts.experiments.v4_action_loss_comparison import run as paired
from scripts.training.losses import chamfer_samples,nearest_distances
from scripts.training.v4_runtime import build_model,save_checkpoint,load_checkpoint
from scripts.tests.training.test_v4_action_identifiability import fixture
from scripts.experiments.v4_action_identifiability.data import normalize


def batch_of(p,t,mask=None,tmask=None):
    if mask is None:mask=torch.ones(p.shape[:2],dtype=torch.bool)
    if tmask is None:tmask=torch.ones(t.shape[:2],dtype=torch.bool)
    return {'inputs':{'gs_xyz':p,'gs_mask':mask},'targets':{'target_xyz':[t],'target_mask':[tmask]}}


class SquaredChamferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_reference_and_not_square_of_reduction(self):
        p=torch.tensor([[[0.,0.,0.],[4.,0.,0.]]]);t=torch.tensor([[[1.,0.,0.],[2.,0.,0.],[7.,0.,0.]]])
        b=batch_of(p,t);d=torch.cdist(p,t).square()
        expected=.5*(d.min(2).values.mean(1)+d.min(1).values.mean(1))
        for chunk in (1,2,512):torch.testing.assert_close(squared_chamfer_samples(p[:,None],b,chunk),expected)
        self.assertFalse(torch.allclose(expected,chamfer_samples(p[:,None],b).square()))

    def test_same_nearest_indices_and_permutation_not_correspondence(self):
        torch.manual_seed(7);p=torch.randn(5,3,dtype=torch.double);t=torch.randn(8,3,dtype=torch.double)
        d=torch.cdist(p,t,compute_mode='donot_use_mm_for_euclid_dist')
        self.assertTrue(torch.equal(d.argmin(1),d.square().argmin(1)))
        torch.testing.assert_close(nearest_distances(p,t,2).square(),d.square().min(1).values)
        b=batch_of(p[None],p.flip(0)[None]);torch.testing.assert_close(squared_chamfer_samples(p[None,None],b),torch.zeros(1,dtype=torch.double))

    def test_variable_target_counts_and_padding_nonfinite(self):
        p=torch.tensor([[[0.,0.,0.],[float('nan'),0.,0.]],[[1.,0.,0.],[3.,0.,0.]]],requires_grad=True)
        t=torch.tensor([[[2.,0.,0.],[float('nan'),0.,0.],[float('inf'),0.,0.]],[[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]]])
        b=batch_of(p,t,torch.tensor([[True,False],[True,True]]),torch.tensor([[True,False,False],[True,True,True]]))
        out=squared_chamfer_samples(p[:,None],b,1);torch.testing.assert_close(out,torch.tensor([4.,1.]))
        out.mean().backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertTrue((p.grad[0,1]==0).all())

    def test_gradcheck_and_identical_zero(self):
        p=torch.tensor([[[.2,.1,0.],[2.2,0.,.1]]],dtype=torch.double,requires_grad=True)
        t=torch.tensor([[[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]]],dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda v:squared_chamfer_samples(v[:,None],batch_of(v,t),1),(p,)))
        z=squared_chamfer_samples(p[:,None],batch_of(p,p.detach()),1);z.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertEqual(float(z.detach().sum()),0.)

    def test_reject_invalid_valid_points_and_empty(self):
        p=torch.zeros(1,2,3);t=p.clone();t[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):squared_chamfer_samples(p[:,None],batch_of(p,t))
        with self.assertRaises(ValueError):squared_chamfer_samples(p[:,None],batch_of(p,p,tmask=torch.zeros(1,2,dtype=torch.bool)))

    def test_adapter_restores_and_base_output_loss_unchanged(self):
        raw,*_=fixture();torch.manual_seed(42);m=build_model('v4_stage1');b,_=normalize(raw,m)
        old=paired.loss_samples;before=m(b['inputs'])['xyz'];loss=chamfer_samples(before,b)
        with diagnostic_adapter():
            torch.testing.assert_close(m(b['inputs'])['xyz'],before,rtol=0,atol=0)
            torch.testing.assert_close(paired.loss_samples({'xyz':before},b,'cd'),loss,rtol=0,atol=0)
        self.assertIs(old,paired.loss_samples)
        try:
            with diagnostic_adapter():raise RuntimeError('test')
        except RuntimeError:pass
        self.assertIs(old,paired.loss_samples)

    def test_checkpoint_optimizer_gradient_and_pair_identity(self):
        raw,*_=fixture();frozen=copy.deepcopy(raw);torch.manual_seed(42);m=build_model('v4_stage1');b,p=normalize(raw,m)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4)
        with diagnostic_adapter():
            rows=[paired.update(m,opt,b,'squared_cd') for _ in range(3)]
            for group in ('motion','controller','attention_score','gs_encoder'):
                self.assertTrue(any(r['raw_gradients'][group]['gradient_norm']>0 for r in rows[1:]))
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'state.pt';save_checkpoint(path,m,opt,3,source_sha='fixture',processor_state=p.state_dict(),run_state={'mode':'squared_cd','coefficient':1})
                clone=build_model('v4_stage1');co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4)
                state=load_checkpoint(path,clone,co,processor_state=p.state_dict());self.assertEqual(state['successful_steps'],3)
                paired.update(m,opt,b,'squared_cd');paired.update(clone,co,b,'squared_cd')
                for key,v in m.state_dict().items():torch.testing.assert_close(v,clone.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(paired.digest(raw),paired.digest(frozen))

    def test_horizon_branch_means(self):
        p=torch.zeros(2,2,1,3);t1=torch.zeros(2,1,3);t1[:,0,0]=torch.tensor([1.,3.]);t2=t1*2
        b=batch_of(p[:,0],t1);b['targets']={'target_xyz':[t1,t2],'target_mask':[b['inputs']['gs_mask']]*2}
        torch.testing.assert_close(squared_chamfer_samples(p,b),torch.tensor([2.5,22.5]))

    def test_three_way_report_fixture(self):
        import json
        from scripts.experiments.v4_squared_chamfer.report import write_report
        raw,_,affected,_=fixture();m=build_model('v4_stage1');b,p=normalize(raw,m)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new';old=Path(tmp)/'old';root.mkdir();old.mkdir();results={}
            with diagnostic_adapter():
                for mode in ('cd','squared_cd','mse'):
                    dest=(root if mode=='squared_cd' else old)/mode;dest.mkdir()
                    row,pred=paired.measure(m,b,p,affected,mode);row['update']=0
                    results[mode]=dict(best_update=0,evaluations=[row],updates=[dict(raw_loss=0.,gradient_norm=0.,actual_updates={'motion':{'update_norm':0.}},clipped=False)],probes={label:dict(conditions={'static':row}) for label in ('best','last')},checkpoint_sha256={})
                    for label in ('best','last'):paired.export_npz(dest/f'{label}_predictions.npz',pred,b,p)
                    (dest/'metrics.json').write_text('{}');(dest/'training.csv').write_text('update,loss\n')
            (root/'results.json').write_text(json.dumps(results));(root/'config.json').write_text(json.dumps({'source_sha':'fixture','job_id':'fixture'}))
            write_report(root,old)
            self.assertEqual(len(list((root/'figures').glob('*.html'))),12)
            self.assertIn('十帧几何',(root/'report.html').read_text(encoding='utf-8'))
