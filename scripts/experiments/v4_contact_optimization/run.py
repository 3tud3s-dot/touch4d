"""Bounded contact-only experiments using unchanged V4, processor and renderer."""
import argparse,copy,csv,hashlib,json,math,os,random,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.v4_runtime import build_model,build_dataset,prepare_batch,save_checkpoint,load_checkpoint
from scripts.training.train import GS_CAMERAS,gs_render_settings,copy_initial,to_device,atomic_json
from scripts.training.losses import chamfer_samples
from scripts.training.rendering import GaussianRenderer
from scripts.experiments.fixed_window.fixed_fit import TargetCache,digest,prediction_metrics
from scripts.experiments.v4_stage1.single_window import group_stats,norm
from .protocol import labels,keys,representative,sequence,matrix,select_candidate,sha
from .loss import loss,squared_chamfer_samples
SHARED=Path('/data1/userdata/tcweng/projects/touch4d')
FROZEN=Path('/data1/userdata/tcweng/experiments/touch4d/contact-boundaries-001-frozen-v1')
BASELINE=Path('/data1/userdata/tcweng/experiments/touch4d/v4-001-h10-e8/run-30947')

def dump(path,obj):atomic_json(Path(path),obj)
def seed():
 torch.manual_seed(42);random.seed(42);np.random.seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def fresh():
 seed();return build_model('v4_stage1',horizon=10,prediction_interval=1).cuda()

def provenance():return dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),torch=torch.__version__)

class Data:
 def __init__(self):
  self.annotation=labels(FROZEN/'contact_boundaries.json');self.processor=DataProcessor('001-rope')
  self.datasets={split:build_dataset('001-rope',split,SHARED/'datasets/touch4d-v1',horizon=10,prediction_interval=1,with_depth=True,depth_root=SHARED/'datasets/touch4d-depth-v1',depth_cameras=GS_CAMERAS['001-rope']) for split in ('train','val')}
  self.indices={k:{tuple(v):i for i,v in enumerate(d.windows)} for k,d in self.datasets.items()}
  self.windows={s:keys(self.annotation,s) for s in ('train','val','test')}
  for s in ('train','val'):assert all(k in self.indices[s] for k in self.windows[s])
 def batch(self,key,model):
  ep,t=key;s='train' if ep in (1,2,4,5,7,8) else 'val'
  if ep in (0,9):raise ValueError('Test GS access forbidden')
  raw=collate_fn([self.datasets[s][self.indices[s][tuple(key)]]])
  assert raw['meta']['frame_ids'].tolist()==[list(range(t,t+11))]
  b=to_device(prepare_batch(raw,self.processor,model),'cuda')
  # Empty individual views are valid mask states; each target needs some supervision.
  assert b['targets']['target_depth_valid'].flatten(2).any(-1).all()
  return raw,b

def summary(rows):
 def avg(rr):
  out={k:np.mean([r[k] for r in rr],axis=0).tolist() for k in ('cd_mm','rgb','alpha','motion_rms_mm')}
  out.update(mean_cd_mm=float(np.mean(out['cd_mm'])),mean_rgb=float(np.mean(out['rgb'])),mean_alpha=float(np.mean(out['alpha'])),common_joint=float(np.mean([r['common_joint'] for r in rr])))
  out['render_score']=out['mean_rgb']+.1*out['mean_alpha'];return out
 eps={str(e):avg([r for r in rows if r['episode']==e]) for e in sorted({r['episode'] for r in rows})}
 equal={k:np.mean([v[k] for v in eps.values()],axis=0).tolist() for k in next(iter(eps.values()))}
 return dict(equal=equal,episodes=eps,windows=rows)

@torch.no_grad()
def record(pred,b,p,renderer):
 r=prediction_metrics(pred,b,p,renderer,512);scale=p.state_dict()['scale'];keep=b['inputs']['gs_mask'][0]
 d=(pred['xyz'][0,:,keep]-b['inputs']['gs_xyz'][0,keep])*scale*1000
 r['motion_rms_mm']=d.square().sum(-1).mean(-1).sqrt().tolist();r['signed_mean_motion_mm']=d.mean(1).tolist()
 r['common_joint']=r['mean_cd_mm']/(scale*1000)+r['mean_rgb']+.1*r['mean_alpha']
 return r

@torch.no_grad()
def attention(trace,b,p):
 out=[];scale=p.state_dict()['scale']*1000;am=trace['field'].anchor_mask
 for k,t in enumerate(trace['steps']):
  a=t['alpha'][am];cp=t['features13'][...,:3];dist=(t['anchor_xyz'][:,:,None]-cp[:,None]).norm(dim=-1)*scale
  mass=[]
  for lo,hi in ((0,10),(10,30),(30,float('inf'))):
   g=(dist>=lo)&(dist<hi)&t['token_mask62'][:,None,2:];mass.append((t['alpha'][...,:60]*g[:,:,None]).sum(-1)[am].mean(0).tolist())
  out.append(dict(null_by_head=a[...,-1].mean(0).tolist(),controller_mass_by_head=a[...,:60].sum(-1).mean(0).tolist(),valid_controllers=int(t['token_mask62'][0,2:].sum()),distance_bins_mm=[[0,10],[10,30],[30,None]],mass_by_distance=mass,tool_motion_mean_mm=(t['tool_motion'][am].reshape(-1,4,3).mean(0)*scale).tolist(),tool_motion_rms_mm=float(t['tool_motion'][am].square().mean().sqrt()*scale)))
 return out

def alter(b,mode):
 c=copy.deepcopy(b);x=c['inputs']
 if mode=='freeze_action':
  for n in ('action_state','controller_points'):x[n]=x[n][:,:1].expand_as(x[n]).clone()
  for n in ('action_valid','controller_mask'):x[n]=x[n]&x[n][:,:1]
 elif mode=='zero_tactile':x['tactile'].zero_()
 return c

@torch.no_grad()
def evaluate(model,data,windows,diagnostics=False,is_copy=False):
 model.eval();rows=[]
 for key in windows:
  _,b=data.batch(key,model);cache=TargetCache(GaussianRenderer())
  pred=copy_initial(b['inputs'],'gs_geometry') if is_copy else model(b['inputs'])
  r=record(pred,b,data.processor,cache);r.update(episode=key[0],start=key[1])
  if diagnostics and not is_copy:
   original,trace=model(b['inputs'],return_trace=True);r['attention']=attention(trace,b,data.processor);r['conditions']={}
   for mode in ('freeze_action','zero_tactile'):
    c=alter(b,mode);changed,ct=model(c['inputs'],return_trace=True);z=record(changed,b,data.processor,cache)
    keep=b['inputs']['gs_mask'][0];diff=(changed['xyz'][0,:,keep]-original['xyz'][0,:,keep])*data.processor.state_dict()['scale']*1000
    r['conditions'][mode]=dict(metrics=z,delta_cd_mm=z['mean_cd_mm']-r['mean_cd_mm'],output_change_rms_mm=diff.square().sum(-1).mean(-1).sqrt().tolist(),attention=attention(ct,c,data.processor))
  rows.append(r)
 return summary(rows)

def update(model,opt,b,p,mode,weight):
 model.train();opt.zero_grad(set_to_none=True);before={n:v.detach().clone() for n,v in model.named_parameters()}
 pred=model(b['inputs']);value,parts=loss(pred,b,p,TargetCache(GaussianRenderer()),mode,weight);value.backward()
 if not torch.isfinite(value) or any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite loss/gradient')
 groups=group_stats(model,gradients=True);gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
 if any(not torch.isfinite(v).all() for v in model.parameters()):raise FloatingPointError('Nonfinite updated parameters')
 return dict(loss=float(value.detach()),rgb=float(parts['rgb'].mean()),alpha=float(parts['alpha'].mean()),geometry=float(parts.get('squared_geometry',parts['geometry']).mean()),gradient_norm=gn,clipped=gn>1,raw_groups=groups,updates=group_stats(model,before=before))

def snapshot(source,dest):
 # Historical file remains read-only. Detect atomic replacement during copy.
 h=sha(source);shutil.copyfile(source,dest)
 if sha(dest)!=h or sha(source)!=h:raise RuntimeError('Historical checkpoint changed during snapshot; no training started')
 return dict(source=str(source),snapshot=str(dest),sha256=h)

def gradient_audit(model,b,p):
 pred=model(b['inputs']);cache=TargetCache(GaussianRenderer());_,parts=loss(pred,b,p,cache,'cd',1.)
 geo=chamfer_samples(pred['xyz'],b).mean();terms={'cd':geo,'rgb':parts['rgb'].mean(),'alpha':parts['alpha'].mean()}
 params=list(model.parameters());vectors={};groups={}
 for name,value in terms.items():
  model.zero_grad(set_to_none=True);value.backward(retain_graph=True);groups[name]=dict(value=float(value.detach()),groups=group_stats(model,gradients=True));vectors[name]=torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in params]).detach()
 cosine={a+'_'+c:float(torch.nn.functional.cosine_similarity(vectors[a][None],vectors[c][None])) for a,c in (('cd','rgb'),('cd','alpha'),('rgb','alpha'))}
 model.zero_grad(set_to_none=True);return dict(terms=groups,cosine=cosine)

def preflight(root):
 out=root/'provenance';out.mkdir(exist_ok=False);data=Data();m=fresh();p=data.processor
 annotation_hash=sha(FROZEN/'contact_boundaries.json');initial_hash=digest(m.state_dict())
 manifest=dict(annotation_sha256=annotation_hash,windows=data.windows,counts={k:len(v) for k,v in data.windows.items()},calibration=representative(data.windows['train']),visualization=representative(data.windows['train'])+representative(data.windows['val']))
 dump(root/'contact_window_manifest.json',manifest)
 dump(root/'experiment_registry.json',dict(protocol='frozen before training',short=matrix(),long_A=dict(loss='cd',sampling='window',updates=12352,eval_every=1544,seed=42),long_B='selection rule protocol.select_candidate; fallback S6',checkpoint_selection='common validation CD(normalized)+RGB+0.1Alpha, episode equal for ALL groups',no_seed43_runs=True,initial_model_hash=initial_hash,source=provenance()))
 cal=[]
 for key in manifest['calibration']:
  _,b=data.batch(key,m)
  with torch.no_grad():pred=m(b['inputs'])
  xyz=pred['xyz'].detach().requires_grad_(True);cd=chamfer_samples(xyz,b).mean();sq=squared_chamfer_samples(xyz,b).mean()
  gc=torch.autograd.grad(cd,xyz)[0];gs=torch.autograd.grad(sq,xyz)[0]
  if not torch.isfinite(gc).all() or not torch.isfinite(gs).all() or gs.norm()==0:raise RuntimeError('Invalid calibration')
  with torch.no_grad():rr=record(pred,b,p,TargetCache(GaussianRenderer()))
  cal.append(dict(key=key,cd=float(cd),squared=float(sq),cd_gradient=norm(gc),squared_gradient=norm(gs),ratio=norm(gc)/norm(gs),initial_rgb=rr['mean_rgb'],initial_alpha=rr['mean_alpha']))
 weight=float(np.median([r['ratio'] for r in cal]));assert math.isfinite(weight) and weight>0
 dump(out/'calibration.json',dict(rule='median of six CD/SquaredCD XYZ gradient norm ratios at paired seed42 initialization; no optimizer updates',lambda_sq=weight,windows=cal,scaled_gradient=[r['squared_gradient']*weight for r in cal],initial_model_hash=initial_hash))
 baseline_refs={name:snapshot(BASELINE/f'{name}.pt',out/f'baseline_{name}.pt') for name in ('best','last') if (BASELINE/f'{name}.pt').exists()}
 if 'best' not in baseline_refs:raise RuntimeError('No trusted historical best checkpoint')
 load_checkpoint(out/'baseline_best.pt',m,processor_state=p.state_dict())
 bugs=[]
 for key in (manifest['calibration'][0],manifest['calibration'][3],*manifest['visualization'][-2:]):
  _,b=data.batch(key,m);pred,tr=m(b['inputs'],return_trace=True)
  gradients=torch.autograd.grad(pred['xyz'][:,-1].square().mean(),[t['state'] for t in tr['steps']],allow_unused=True)
  assert all(g is not None and torch.isfinite(g).all() for g in gradients)
  cp=b['inputs']['controller_points'].reshape(1,10,60,3)
  for k,t in enumerate(tr['steps']):
   delta=torch.zeros_like(cp[:,k]) if k==0 else torch.where(t['adjacent_valid'][...,None],cp[:,k]-cp[:,k-1],0.)
   torch.testing.assert_close(t['tool_motion'],torch.einsum('bahj,bjd->bahd',t['alpha'][...,:60],delta).flatten(-2),rtol=0,atol=0)
   if k:torch.testing.assert_close(t['anchor_history'],t['anchor_xyz']-tr['steps'][k-1]['anchor_xyz'])
  with torch.no_grad():
   changed=copy.deepcopy(b['inputs']);off=~changed['operation_hands']
   for name in ('controller_points','action_state','tactile'):
    if name=='tactile':mask=off.repeat_interleave(2,-1)[:,None,:,None,None]
    else:mask=off[:,None,:,None] if name=='action_state' else off[:,None,:,None,None]
    changed[name]=torch.where(mask,torch.ones_like(changed[name])*123.,changed[name])
   torch.testing.assert_close(m(changed)['xyz'],pred['xyz'],rtol=0,atol=0)
  bugs.append(dict(key=key,mask_perturbation_exact=True,motion_formula_exact=True,anchor_history_exact=True,graph_connected=True,initial_state_gradient_norms=[norm(g) for g in gradients],attention=attention(tr,b,p)))
 dump(root/'bug_audit.json',dict(status='No deterministic bug found in tested contracts',real_cases=bugs,interpretation='30/60 tools change the softmax denominator by design; null is not a contact probability. Zero initial head can have zero connected gradients.',model_input_width=151))
 # Two updates per Loss, isolated disposable models: gate numerical path and resume.
 smoke=[]
 for mode in ('cd','squared'):
  m=fresh();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);_,b=data.batch(manifest['calibration'][0],m);rows=[];start=time.time();torch.cuda.reset_peak_memory_stats()
  for _ in range(2):rows.append(update(m,opt,b,p,mode,weight))
  ck=out/f'smoke_{mode}.pt';save_checkpoint(ck,m,opt,2,source_sha=provenance()['source_sha'],processor_state=p.state_dict())
  clone=fresh();co=torch.optim.AdamW(clone.parameters(),lr=1e-4,weight_decay=1e-4);load_checkpoint(ck,clone,co,processor_state=p.state_dict())
  with torch.no_grad():torch.testing.assert_close(m(b['inputs'])['xyz'],clone(b['inputs'])['xyz'],rtol=0,atol=0)
  smoke.append(dict(mode=mode,updates=rows,allocated_peak=torch.cuda.max_memory_allocated(),seconds=time.time()-start,restore_exact=True))
 dump(out/'cuda_smoke.json',dict(passed=True,source=provenance(),cases=smoke))
 m=fresh();base_rows={}
 base_rows['copy']=evaluate(m,data,data.windows['val'],is_copy=True)
 for name in ('best','last'):
  source=BASELINE/f'{name}.pt'
  if source.exists():
   ref=baseline_refs[name];load_checkpoint(out/f'baseline_{name}.pt',m,processor_state=p.state_dict());base_rows[name]=evaluate(m,data,data.windows['val'],diagnostics=True);base_rows[name]['provenance']=ref
 dump(out/'baseline_contact.json',base_rows)
 dump(out/'gate.json',dict(passed=True,annotation_sha256=annotation_hash,initial_hash=initial_hash,source=provenance()))


def train(root,run_id,config,resume=False):
 out=root/('short_experiments' if run_id.startswith('S') else 'long_experiment_'+run_id)/run_id;out.mkdir(parents=True,exist_ok=resume)
 data=Data();p=data.processor;m=fresh();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);prov=provenance();weight=json.loads((root/'provenance/calibration.json').read_text())['lambda_sq']
 gate=json.loads((root/'provenance/gate.json').read_text());assert gate['passed'] and gate['annotation_sha256']==sha(FROZEN/'contact_boundaries.json') and gate['initial_hash']==digest(m.state_dict())
 windows=[k for k in data.windows['train'] if k[0] in config['episodes']];order=sequence(windows,config['sampling'],config['updates'],config['seed'])
 config=dict(config,lambda_sq=weight,lr=1e-4,weight_decay=1e-4,clip_norm=1.,horizon=10,prediction_interval=1,render_settings=gs_render_settings(),cameras=GS_CAMERAS['001-rope'],annotation_sha256=gate['annotation_sha256'],initial_model_hash=digest(m.state_dict()))
 dump(out/'config.json',dict(experiment=config,model=m.config));dump(out/'source_provenance.json',prov);dump(out/'window_manifest.json',dict(training=windows,validation=data.windows['val'],sample_order=order))
 ledger=dict(run_id=str(out),config=config,updates=[],evaluations=[],best_joint=None,lowest_cd=None,successful_updates=0)
 def save(name):save_checkpoint(out/name,m,opt,ledger['successful_updates'],source_sha=prov['source_sha'],processor_state=p.state_dict(),run_state=ledger,replace_existing=(out/name).exists())
 if resume:
  candidates=[out/n for n in ('progress.pt','last.pt') if (out/n).exists()]
  if not candidates:raise ValueError('No recovery checkpoint')
  states=[(torch.load(q,map_location='cpu',weights_only=False),q) for q in candidates]
  state,path=max(states,key=lambda pair:pair[0]['successful_steps'])
  old=state.get('run_state') or {}
  if state['source_sha']!=prov['source_sha'] or old.get('config')!=config or old.get('run_id')!=str(out):raise ValueError('Unsafe resume: source/config/run conflict')
  if [r['update'] for r in old['updates']]!=list(range(1,state['successful_steps']+1)):raise ValueError('Invalid successful step ledger')
  load_checkpoint(path,m,opt,processor_state=p.state_dict());ledger=old
 else:save('initial.pt')
 completed=ledger['successful_updates'];start=time.time();torch.cuda.reset_peak_memory_stats()
 for step,key in enumerate(order,1):
  if step<=completed:continue
  before=time.time();_,b=data.batch(key,m);row=update(m,opt,b,p,config['loss'],weight);row.update(update=step,key=key,seconds=time.time()-before);ledger['updates'].append(row);ledger['successful_updates']=step
  if step%config['eval_every']==0 or step==len(order):
   val=evaluate(m,data,data.windows['val']);val['update']=step;ledger['evaluations'].append(val);score=val['equal']['common_joint'];cd=val['equal']['mean_cd_mm']
   if ledger['best_joint'] is None or score<ledger['best_joint']:ledger['best_joint']=score;ledger['best_step']=step;save('best_joint.pt')
   if ledger['lowest_cd'] is None or cd<ledger['lowest_cd']:ledger['lowest_cd']=cd;ledger['lowest_cd_step']=step;save('lowest_cd.pt')
   save('last.pt');dump(out/'metrics.json',ledger);print(run_id,step,val['equal'],flush=True)
  elif step%100==0:save('progress.pt');print(run_id,step,'train_loss',row['loss'],flush=True)
 assert ledger['successful_updates']==config['updates']
 ledger['seconds']=time.time()-start;ledger['allocated_peak']=torch.cuda.max_memory_allocated();ledger['reserved_peak']=torch.cuda.max_memory_reserved()
 load_checkpoint(out/'best_joint.pt',m,processor_state=p.state_dict())
 ledger['best_validation']=evaluate(m,data,data.windows['val'],diagnostics=True)
 ledger['train_diagnostic']=evaluate(m,data,windows if run_id in ('S1','S2','S3','S4') else representative(windows))
 if not run_id.startswith('S'):
  ledger['all_window_validation']=evaluate(m,data,list(data.datasets['val'].windows))
  _,b=data.batch(representative(data.windows['train'])[0],m);ledger['gradient_audit']=gradient_audit(m,b,p)
 ledger['checkpoint_hashes']={n:sha(out/n) for n in ('initial.pt','best_joint.pt','lowest_cd.pt','last.pt')}
 dump(out/'metrics.json',ledger)
 with (out/'training_metrics.csv').open('w',newline='') as f:
  fields=['update','key','loss','geometry','rgb','alpha','gradient_norm','clipped','seconds'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(ledger['updates'])
 dump(out/'status.json',dict(status='COMPLETED',successful_updates=config['updates'],source=prov))


def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--stage',choices=['preflight','short','long'],required=True);a.add_argument('--id');a.add_argument('--resume',action='store_true');a.add_argument('--expected-sha',required=True);args=a.parse_args()
 if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires normal Slurm GPU allocation')
 if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=args.expected_sha or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Source drift')
 torch.set_num_threads(4);root=Path(args.root);root.mkdir(exist_ok=True)
 try:
  if args.stage=='preflight':preflight(root)
  elif args.stage=='short':train(root,args.id,matrix()[args.id],args.resume)
  else:
   cfg=dict(loss='cd',sampling='window',episodes=[1,2,4,5,7,8],updates=12352,eval_every=1544,seed=42)
   if args.id=='B':
    selection=json.loads((root/'selection.json').read_text());c=matrix()[selection['winner']];cfg.update(loss=c['loss'],sampling=c['sampling'])
   train(root,args.id,cfg,args.resume)
 except BaseException:
  dump(root/f'failure_{args.stage}_{args.id or "gate"}.json',dict(traceback=traceback.format_exc(),source=provenance()));raise

if __name__=='__main__':main()
