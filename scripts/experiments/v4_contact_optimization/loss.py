"""Squared nearest-neighbour diagnostic port from acfe977; production unchanged."""
import torch
from scripts.training.losses import nearest_distances,prediction_horizon,chamfer_samples,objective

def squared_chamfer_samples(prediction,batch,chunk=512):
 horizon=prediction_horizon(prediction,batch);x,y=batch['inputs'],batch['targets'];samples=[]
 for b in range(len(prediction)):
  frames=[]
  for k in range(horizon):
   p=prediction[b,k,x['gs_mask'][b]];t=y['target_xyz'][k][b,y['target_mask'][k][b]].detach()
   if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise FloatingPointError('Nonfinite valid points')
   frames.append(.5*(nearest_distances(p,t,chunk).square().mean()+nearest_distances(t,p,chunk).square().mean()))
  samples.append(torch.stack(frames).mean())
 return torch.stack(samples)

def loss(pred,batch,processor,renderer,mode,weight):
 if mode=='cd':return objective(pred,batch,processor,renderer,chunk=512,chamfer_weight=1.,rgb_weight=1.,alpha_weight=.1,scale_weight=0.,depth_weight=0.)
 if mode!='squared' or not weight>0:raise ValueError('Loss config')
 render,parts=objective(pred,batch,processor,renderer,chunk=512,chamfer_weight=0.,rgb_weight=1.,alpha_weight=.1,scale_weight=0.,depth_weight=0.)
 sq=squared_chamfer_samples(pred['xyz'],batch)
 if not parts['render_available'].all():raise ValueError('No valid supervision in sample')
 parts['squared_geometry']=sq.detach()
 return render+weight*sq.mean(),parts
