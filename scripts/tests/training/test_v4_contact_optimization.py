import copy,json,tempfile,unittest
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
 def test_original_loss_equivalence_and_squared_backward(self):
  torch.manual_seed(42);m=build_model('v4_stage1',horizon=2);p=DataProcessor('001-rope');b=prepare_batch(synthetic_batch(points=16,horizon=2),p,m);pred=m(b['inputs']);r=GaussianRenderer(analytic_rasterizer)
  a,_=v4_objective(pred,b,p,r);z,_=loss(pred,b,p,r,'cd',1.);torch.testing.assert_close(a,z,rtol=0,atol=0)
  s,_=loss(pred,b,p,r,'squared',10.);s.backward();self.assertTrue(torch.isfinite(s));self.assertIsNotNone(m.motion[-1].weight.grad)

if __name__=='__main__':unittest.main()
