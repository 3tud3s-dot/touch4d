"""24 bounded independent anchor-only fits, with fail-closed mapping pilot."""
import argparse,csv,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from scripts.training.losses import objective,nearest_distances
from scripts.training.train import to_device,copy_initial,make_renderer
from scripts.training.v4_model import fixed_field
from scripts.training.visualize import png_bytes
from scripts.experiments.full_training.v4_epochs import loader
from scripts.experiments.fixed_window.fixed_fit import digest
from .io import checked_model,batch_for,sha,write,cpu,metadata
from .core import decode,one_horizon,loss_parts,edges,structure
from .regions import fixed_regions,rms_mm

WINDOWS=[('train',7,105),('val',3,135)]
HS=(5,10,15)
CONFIG=dict(optimizer='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0,clip=None,updates=300,seed=42,
            losses=['cd','joint'],initializations=['zero','normal'],horizons=HS,windows=WINDOWS,
            selection='lowest own objective over steps 0..300; last also retained',pilot='first update of first formal fit; counts toward 300')


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def images_save(root,key,mode,images):
    for view in range(3):
        for kind in ('rgb','alpha'):
            a=images[kind][0,0,view].detach().cpu().numpy()
            if a.ndim==2:a=np.repeat(a[...,None],3,-1)
            (root/'images'/f'{key}_{mode}_v{view}_{kind}.png').write_bytes(png_bytes(np.rint(np.clip(a,0,1)*255).astype(np.uint8)))


@torch.no_grad()
def evaluate(p,b,processor,renderer,target,regions,edge,normal):
    _,parts=loss_parts(p,b,processor,renderer,target)
    scale=processor.state_dict()['scale'];pos=p['xyz'][0,0];gt=b['targets']['target_xyz'][0][0,b['targets']['target_mask'][0][0]]
    out={k:float(v) for k,v in parts.items()};out['cd_mm']=out['cd']*scale*1000
    for name,idx in regions.items():
        out[name+'_motion_mm']=rms_mm(pos[idx]-b['inputs']['gs_xyz'][0,idx],scale*1000)
        out[name+'_error_mm']=float(nearest_distances(pos[idx],gt,512).mean())*scale*1000
    out['normal_change_mm']=rms_mm(pos[regions['all']]-normal['xyz'][0,0,regions['all']],scale*1000)
    s,heat=structure(b['inputs'],pos,edge,scale);out.update(s)
    return out,heat


def execute(root,expected):
    if metadata()['source_sha']!=expected or subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Source not clean/pinned')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Allocated CUDA required')
    if (root/'provenance/config.json').exists():raise FileExistsError('No implicit restart or overwrite')
    for d in ['results','figures','images','logs','provenance','arrays']:(root/d).mkdir(parents=True,exist_ok=True)
    write(root/'provenance/config.json',CONFIG)
    torch.manual_seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,processor,state,config,ref=checked_model();model=model.cuda();model.requires_grad_(False)
    before=digest(model.state_dict());renderer=make_renderer(config);sets={s:loader(config,s).dataset for s in ('train','val')}
    write(root/'provenance/checkpoint.json',ref)
    write(root/'provenance/data_config.json',config)
    prepared=[];gates=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    # All six equivalence gates pass before ANY optimization, including the pilot.
    for split,ep,start in WINDOWS:
        raw,b=batch_for(sets[split],(ep,start),processor,model);b=to_device(b,'cuda')
        with torch.no_grad():
            normal,trace=model(b['inputs'],return_trace=True);f=trace['field'];rebuilt=fixed_field(b['inputs']['gs_xyz'],b['inputs']['gs_mask'])
            assert torch.equal(f.weights,rebuilt.weights) and torch.equal(f.anchor_indices,rebuilt.anchor_indices)
            zero=torch.zeros_like(trace['steps'][0]['cumulative_d']);copy=copy_initial(b['inputs'],'gs_geometry')
            assert torch.equal(decode(b['inputs'],f,zero)['xyz'][:,0],copy['xyz'][:,0])
            for h in HS:
                reconstructed=decode(b['inputs'],f,trace['steps'][h-1]['cumulative_d'])
                assert torch.equal(reconstructed['xyz'][:,0],normal['xyz'][:,h-1]),'D0 Normal equivalence failure'
            assert not f.weights.requires_grad and f.anchor_mask.sum()==128
            regions,_=fixed_regions(b,f)
        gates.append(dict(episode=ep,start=start,frames=raw['meta']['frame_ids'][0].tolist(),batch_hash=digest(raw),field_hash=digest(f.__dict__),zero_bit_exact=True,normal_bit_exact_horizons=HS,rebuild_bit_exact=True))
        prepared.append((split,ep,start,b,normal,trace,regions))
    write(root/'provenance/equivalence.json',dict(passed=True,gates=gates,state_before=before))
    all_rows=[];curves=[];pilot_done=False
    for split,ep,start,b,normal,trace,regions in prepared:
        f=trace['field'];field_hash=digest(f.__dict__);edge=edges(b['inputs']);keep=b['inputs']['gs_mask'][0];scale=processor.state_dict()['scale']
        for h in HS:
            key=f'ep{ep}_start{start}_h{h:02}';one=one_horizon(b,h);npred={k:v[:,h-1:h] for k,v in normal.items()}
            with torch.no_grad():target=renderer.render_targets(one,processor)
            images_save(root,key,'gt',target)
            arrays={'initial':processor.denormalize_xyz(b['inputs']['gs_xyz'],b['inputs']['gs_mask'])[0,keep].cpu().numpy(),'gt':processor.denormalize_xyz(one['targets']['target_xyz'][0],one['targets']['target_mask'][0])[0,one['targets']['target_mask'][0][0]].cpu().numpy(),'near':regions['near'].cpu().numpy(),'far':regions['far'].cpu().numpy()}
            baseline={}
            for name,p in [('copy',decode(b['inputs'],f,torch.zeros_like(trace['steps'][h-1]['cumulative_d']))),('normal',npred)]:
                vals,heat=evaluate(p,one,processor,renderer,target,regions,edge,npred);baseline[name]=vals
                all_rows.append(dict(window=key,split=split,horizon=h,method=name,initialization='none',snapshot='fixed',selected_step=0,**vals))
                with torch.no_grad():images_save(root,key,name,renderer(p,one,processor))
                arrays[name]=processor.denormalize_xyz(p['xyz'],b['inputs']['gs_mask'][:,None])[0,0,keep].cpu().numpy();arrays[name+'_heat']=heat[keep].cpu().numpy()
            for loss_name in ('cd','joint'):
                for initialization in ('zero','normal'):
                    initial=trace['steps'][h-1]['cumulative_d'].detach().clone()
                    if initialization=='zero':initial.zero_()
                    d=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([d],lr=.001,weight_decay=0)
                    assert len(opt.param_groups[0]['params'])==1 and opt.param_groups[0]['params'][0] is d
                    best=float('inf');best_d=None;best_step=0;grad=update=0.;fit_curve=[]
                    for step in range(301):
                        p=decode(b['inputs'],f,d)
                        # Evaluate both reporting terms without changing the CD-only gradient.
                        with torch.no_grad():_,parts=loss_parts(p,one,processor,renderer,target)
                        score=float(parts[loss_name]);row=dict(window=key,horizon=h,loss=loss_name,initialization=initialization,step=step,cd=float(parts['cd']),cd_mm=float(parts['cd'])*scale*1000,rgb=float(parts['rgb']),alpha=float(parts['alpha']),joint=float(parts['joint']),objective=score,previous_gradient_norm=grad,previous_update_norm=update)
                        fit_curve.append(row);curves.append(row)
                        if score<best:best=score;best_d=d.detach().clone();best_step=step
                        if step==300:break
                        opt.zero_grad(set_to_none=True)
                        if loss_name=='cd':loss,_=loss_parts(p,one,processor,renderer,target,render=False)
                        else:loss,_=loss_parts(p,one,processor,renderer,target)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                        loss.backward()
                        if d.grad is None or not torch.isfinite(d.grad).all():raise FloatingPointError('Invalid D gradient')
                        grad=float(d.grad.norm());old=d.detach().clone();opt.step();update=float((d-old).norm())
                        if not torch.isfinite(d).all():raise FloatingPointError('Invalid anchor update')
                        if not pilot_done:
                            # CD, joint reduction and renderer gradient verified on the first fixed target.
                            trial=decode(b['inputs'],f,d);joint,pp=loss_parts(trial,one,processor,renderer,target)
                            reference,_=objective(trial,one,processor,renderer,chamfer_weight=1,rgb_weight=1,alpha_weight=.1,scale_weight=0)
                            torch.testing.assert_close(joint,reference,rtol=1e-6,atol=1e-7)
                            joint_grad=torch.autograd.grad(joint,d)[0]
                            assert torch.isfinite(joint_grad).all() and joint_grad.norm()>0 and grad>0 and update>0
                            assert digest(model.state_dict())==before and all(q.grad is None for q in model.parameters())
                            write(root/'provenance/pilot.json',dict(passed=True,window=key,step=1,cd_gradient=grad,update=update,joint_gradient=float(joint_grad.norm()),joint_reduction_equal=True,model_unchanged=True))
                            pilot_done=True;print('PILOT PASS',flush=True)
                    assert digest(f.__dict__)==field_hash and digest(model.state_dict())==before and all(q.grad is None for q in model.parameters())
                    name=f'{loss_name}_{initialization}'
                    torch.save(dict(best_d=cpu(best_d),last_d=cpu(d),initial_d=cpu(initial),best_step=best_step,optimizer=cpu(opt.state_dict()),updates=300),root/'arrays'/f'{key}_{name}.pt')
                    csv_write(root/'results'/f'{key}_{name}_curve.csv',fit_curve)
                    for snapshot,dd,chosen in [('best',best_d,best_step),('last',d.detach(),300)]:
                        pp=decode(b['inputs'],f,dd);vals,heat=evaluate(pp,one,processor,renderer,target,regions,edge,npred)
                        all_rows.append(dict(window=key,split=split,horizon=h,method='D0-'+loss_name,initialization=initialization,snapshot=snapshot,selected_step=chosen,**vals))
                        arrays[name+'_'+snapshot]=processor.denormalize_xyz(pp['xyz'],b['inputs']['gs_mask'][:,None])[0,0,keep].detach().cpu().numpy();arrays[name+'_'+snapshot+'_heat']=heat[keep].cpu().numpy()
                        with torch.no_grad():images_save(root,key,name+'_'+snapshot,renderer(pp,one,processor))
                    print(json.dumps(dict(event='fit_completed',window=key,loss=loss_name,initialization=initialization,best_step=best_step,initial=fit_curve[0]['objective'],best=best,last=fit_curve[-1]['objective'])),flush=True)
            np.savez_compressed(root/'arrays'/f'{key}.npz',**arrays)
            csv_write(root/'results/summary.csv',all_rows);csv_write(root/'results/optimization_curves.csv',curves)
    csv_write(root/'results/geometry_metrics.csv',all_rows)
    assert sha(ref['snapshot'])==ref['sha256'] and digest(model.state_dict())==before
    write(root/'results/results.json',all_rows)
    write(root/'provenance/completion.json',dict(passed=True,fits=24,updates_per_fit=300,network_updates=0,model_state_before=before,model_state_after=digest(model.state_dict()),checkpoint_after_sha256=sha(ref['snapshot']),seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),execution=metadata()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-sha',required=True);a=p.parse_args()
    try:execute(a.root,a.expected_sha)
    except Exception:
        write(a.root/'provenance/error.json',dict(error=traceback.format_exc(),execution=metadata()));raise
