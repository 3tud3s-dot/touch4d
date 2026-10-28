import copy,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import torch
from scripts.experiments.v4_contact_optimization.protocol import *
from scripts.experiments.v4_contact_optimization.loss import squared_chamfer_samples,loss
from scripts.training.v4_runtime import build_model,prepare_batch,v4_objective
from scripts.experiments.v4_stage1.fixtures import synthetic_batch,analytic_rasterizer
from scripts.data_process import DataProcessor
from scripts.training.rendering import GaussianRenderer

class ContactOptimizationTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def annotation(self):return {'episodes':{str(i):{'total_frames':n} for i,n in enumerate([257,191,292,288,364,307,237,225,225,307])}}
 def test_split_count_and_half_open(self):
  for split in COUNTS:
   rows=keys(self.annotation(),split);self.assertEqual(len(rows),COUNTS[split])
   for ep,t in rows:self.assertTrue(BOUNDARIES[ep][0]<=t and t+10<BOUNDARIES[ep][1]);self.assertEqual(t%(1 if split=='train' else 5),0)
  self.assertEqual(keys(self.annotation(),'val')[0],(3,65))
 def test_sequence_pairing(self):
  rows=keys(self.annotation(),'train')
  for sampling in ('window','balanced'):
   a=sequence(rows,sampling,800);self.assertEqual(a,sequence(rows,sampling,800));self.assertEqual(len(a),800);self.assertTrue(set(a)<=set(rows))
  self.assertEqual(len(matrix()),8)
 def test_selection_fallback_and_balance(self):
  def score(x,y):return {'episodes':{'3':{'mean_cd_mm':x},'6':{'mean_cd_mm':y}},'equal':{'mean_cd_mm':(x+y)/2,'render_score':.1,'common_joint':.2}}
  s={'S5':score(2,2),'S6':score(3,1),'S7':score(1,3),'S8':score(3,3)}
  self.assertTrue(select_candidate(s)['fallback']);s['S8']=score(1.8,1.9);self.assertEqual(select_candidate(s)['winner'],'S8')
 def test_squared_reference_padding_variable_points_grad(self):
  x=torch.tensor([[[[0.,0.,0.],[2.,0.,0.],[999.,0.,0.]]]],requires_grad=True)
  y=torch.tensor([[[1.,0.,0.],[3.,0.,0.],[4.,0.,0.],[999.,0.,0.]]])
  b={'inputs':{'gs_xyz':x[:,0].detach(),'gs_mask':torch.tensor([[True,True,False]])},'targets':{'target_xyz':[y],'target_mask':[torch.tensor([[True,True,True,False]])]}}
  value=squared_chamfer_samples(x,b,2);self.assertAlmostEqual(float(value),1.5);value.sum().backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad[:,:,2].abs().sum()),0)
 def test_resume_keeps_selection_ledger_and_rejects_conflicts(self):
  cfg={'updates':800,'loss':'cd'};ledger={'config':cfg,'run_id':'r','updates':[{'update':1}],'best_joint':.1,'best_step':1,'successful_updates':1}
  state={'source_sha':'s','successful_steps':1,'run_state':ledger}
  self.assertEqual(validate_resume(state,cfg,'s','r')['best_joint'],.1)
  for changed in ({'updates':800,'loss':'squared'},{'updates':0,'loss':'cd'}):
   with self.assertRaises(ValueError):validate_resume(state,changed,'s','r')
  with self.assertRaises(ValueError):validate_resume(state,cfg,'different','r')
  state['successful_steps']=2
  with self.assertRaises(ValueError):validate_resume(state,cfg,'s','r')
 def test_original_loss_equivalence_and_squared_backward(self):
  torch.manual_seed(42);m=build_model('v4_stage1',horizon=2);p=DataProcessor('001-rope');b=prepare_batch(synthetic_batch(points=16,horizon=2),p,m);pred=m(b['inputs']);r=GaussianRenderer(analytic_rasterizer)
  a,_=v4_objective(pred,b,p,r);z,_=loss(pred,b,p,r,'cd',1.);torch.testing.assert_close(a,z,rtol=0,atol=0)
  s,_=loss(pred,b,p,r,'squared',10.);s.backward();self.assertTrue(torch.isfinite(s));self.assertIsNotNone(m.motion[-1].weight.grad)
 def test_interrupted_validation_resumes_without_repeating_optimizer_step(self):
  from scripts.experiments.v4_contact_optimization import run
  def fresh_cpu():
   torch.manual_seed(42);return build_model('v4_stage1',horizon=2)
  class FakeData:
   processor=DataProcessor('001-rope');windows={'train':[(1,0)],'val':[(3,0)]}
   def batch(self,key,model):
    raw=synthetic_batch(points=16,horizon=2);return raw,prepare_batch(raw,self.processor,model)
  metric={'equal':{'common_joint':.1,'mean_cd_mm':1.}}
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'provenance').mkdir();(root/'contact_boundaries.json').write_text('{}')
   (root/'provenance/calibration.json').write_text(json.dumps({'lambda_sq':1.}))
   (root/'provenance/gate.json').write_text(json.dumps({'passed':True,'annotation_sha256':sha(root/'contact_boundaries.json'),'initial_hash':run.digest(fresh_cpu().state_dict())}))
   cfg={'loss':'cd','sampling':'window','episodes':[1],'updates':2,'eval_every':1,'seed':42}
   with patch.object(run,'FROZEN',root),patch.object(run,'Data',FakeData),patch.object(run,'fresh',fresh_cpu),patch.object(run,'provenance',return_value={'source_sha':'test'}),patch.object(run,'GaussianRenderer',side_effect=lambda:GaussianRenderer(analytic_rasterizer)),patch.object(run.torch.cuda,'reset_peak_memory_stats'),patch.object(run.torch.cuda,'max_memory_allocated',return_value=0),patch.object(run.torch.cuda,'max_memory_reserved',return_value=0),patch.object(run,'update',wraps=run.update) as updates:
    with patch.object(run,'evaluate',side_effect=RuntimeError('simulated validation interruption')):
     with self.assertRaisesRegex(RuntimeError,'simulated'):run.train(root,'S1',cfg)
    with patch.object(run,'evaluate',side_effect=lambda *a,**k:copy.deepcopy(metric)):
     run.train(root,'S1',cfg,resume=True)
    self.assertEqual(updates.call_count,2)
    result=json.loads((root/'short_experiments/S1/metrics.json').read_text())
    self.assertEqual(result['successful_updates'],2);self.assertEqual(result['best_step'],1)
 def test_report_curve_finite_and_safe_labels(self):
  from scripts.experiments.v4_contact_optimization.report import multi_curve,loss_curves
  svg=multi_curve({'a':[(1,0.),(2,0.)]},'<title>','horizon')
  self.assertIn('&lt;title&gt;',svg);self.assertNotIn('nan',svg)
  for value in ({},{'a':[]},{'a':[(1,float('nan'))]}):
   with self.assertRaises(ValueError):multi_curve(value,'x','h')
  rows=[dict(update=i,loss=.2,geometry=.03,rgb=.16,alpha=.1) for i in range(1,401)]
  self.assertIn('400',loss_curves({'updates':rows},'CD'))
 def test_report_answers_use_episode_equal_values_and_keep_limits(self):
  from scripts.experiments.v4_contact_optimization.report import summary_sections
  results={}
  for n,value in {'copy':5.,'baseline_best':4.,'baseline_last':4.1,'A_best':3.,'A_last':3.2,'B_best':2.,'B_last':2.2}.items():
   e=dict(mean_cd_mm=value,mean_rgb=.1,mean_alpha=.2,common_joint=.3,cd_mm=[value]*10,rgb=[.1]*10,alpha=[.2]*10,motion_rms_mm=[1.]*10)
   results[n]={'equal':e,'episodes':{'3':e,'6':e}}
  analysis={'models':{n:{e:{'conditions':{}} for e in ('3','6')} for n in results},'short_factorial_delta_cd_mm':{'S5_to_S7':{'3':.1,'6':-.2}},'optimization':{n:{'best_step':1544,'clipped_fraction':.5,'allocated_peak':1024**3} for n in ('A','B')}}
  refs={n:{'source_sha':'frozen'} for n in ('A_best','baseline_best')}
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'selection.json').write_text(json.dumps({'winner':'S6','fallback':True}))
   md,charts=summary_sections(root,results,analysis,[],refs)
   self.assertIn('+25.00%',md);self.assertIn('Long B',md);self.assertIn('不是独立、无偏的泛化证明',md);self.assertIn('12352',md);self.assertIn('horizon',charts)

if __name__=='__main__':unittest.main()
