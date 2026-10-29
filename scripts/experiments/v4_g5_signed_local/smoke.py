"""Bounded real CUDA/gsplat diagnostics: two fresh updates, frozen validation."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from scripts.data_process import DataProcessor
from scripts.dataloader.collate import collate_fn
from scripts.training.v4_runtime import prepare_batch, v4_objective
from scripts.experiments.full_training.v4_epochs import configuration, loader
from scripts.training.train import GS_CAMERAS, make_renderer, gs_render_settings, to_device, atomic_json
from scripts.experiments.v4_g4_global_motion.model import G4GlobalMotionPredictor
from scripts.experiments.v4_g4_global_motion.checkpoint import load_for_evaluation
from scripts.experiments.v4_g4_global_motion.smoke import same_tree
from .checkpoint import build_model, save_checkpoint, load_checkpoint
from .diagnostics import (SELECTED, REGION_NOTE, sha256, state_hash, finite_tree, cpu_tree,
    mapping_context, collect, graph_diagnostics, controlled_neighbor_probe, parameter_evidence, write_csv)

BASE = 'cb242284eb6fd151c5a5f7d90ed6bc582993c087'
G4_HASH = '240045e3ccb0a37bfcc954feabb2b61f4a1f828d63566eb856619ebe6335b92d'
WINDOWS = (('train',7,105),('val',3,135),('val',6,110))
MODELS = ('model.py','motion.py','propagation.py')


def source_preflight(expected):
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source != expected or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Source must be clean and pinned')
    hashes={}
    for name in MODELS:
        rel='scripts/experiments/v4_g5_signed_local/'+name
        if subprocess.check_output(['git','show',BASE+':'+rel]) != subprocess.check_output(['git','show','HEAD:'+rel]):
            raise AssertionError('Frozen G5 graph changed: '+rel)
        hashes[rel]=sha256(rel)
    return source,hashes


def load_windows(args, processor):
    cpu_model=build_model('direct');result={};manifest={}
    for split in ('train','val'):
        dataset=loader(configuration(args.root,args.depth_root,args.expected_sha,horizon=15),split).dataset
        for part,ep,start in WINDOWS:
            if part != split: continue
            raw=collate_fn([dataset[dataset.windows.index((ep,start))]])
            batch=prepare_batch(raw,processor,cpu_model)
            if raw['meta']['frame_ids'].tolist()!=[list(range(start,start+16))]:
                raise AssertionError('Wrong real H15 frame IDs')
            expected=torch.tensor([[ep>=5,True]]);x,y=batch['inputs'],batch['targets']
            if not torch.equal(x['operation_hands'],expected):raise AssertionError('Wrong operating hands')
            if not torch.equal(x['condition_frame_ids'],raw['meta']['frame_ids'][:,:-1]):
                raise AssertionError('Condition indices drifted')
            if tuple(y['target_depth'].shape)!=(1,15,3,720,1280):
                raise AssertionError('Must use original three cameras at 1280x720')
            finite_tree((x['gs_xyz'][x['gs_mask']],x['gs_attr'][x['gs_mask']],y['target_depth']))
            for j in range(15):
                if not y['target_mask'][j].any() or not y['target_depth_valid'][:,j].any():
                    raise AssertionError('Missing GS or rendering supervision')
            for hand,active in enumerate(expected[0]):
                if active and not x['controller_mask'][0,:,hand].any(-1).all():
                    raise AssertionError('Operating hand has no valid Controller')
            name=f'ep{ep}_start{start}';result[name]=batch
            manifest[name]=dict(split=split,episode=ep,start=start,frame_ids=raw['meta']['frame_ids'].tolist()[0],
                condition_frames=list(range(start,start+15)),target_frames=list(range(start+1,start+16)),
                timestamps_us=raw['meta']['observation_timestamp_us'].tolist()[0],
                operating_hands=expected.tolist()[0],hand_order=['left','right'],cameras=GS_CAMERAS['001-rope'],
                render_height=720,render_width=1280,initial_gs_count=int(x['gs_mask'].sum()),
                target_gs_counts=[int(v.sum()) for v in y['target_mask']],
                controller_valid_per_hand=x['controller_mask'][0].sum(-1).tolist(),
                action_valid=x['action_valid'][0].tolist(),input_hash=state_hash({k:v for k,v in x.items() if isinstance(v,torch.Tensor)}),
                available_split_windows=len(dataset.windows),region_definition=REGION_NOTE)
    return result,manifest


class AuditedRenderer:
    """Original renderer/reduction; selected previews are analysis-only copies."""
    def __init__(self,out):
        self.inner=make_renderer(dict(output_mode='gs_geometry',render_settings=gs_render_settings()))
        self.out=out;self.label=None;self.target_previews={};self.finite_calls=0

    def previews(self,images):
        from PIL import Image
        finite_tree(images);self.finite_calls+=1
        if self.label is None:return {}
        saved={}
        for h in SELECTED:
            for view in range(3):
                for key in ('rgb','alpha'):
                    v=images[key][0,h-1,view].detach().clamp(0,1).cpu().numpy()
                    if key=='alpha':v=np.repeat(v[...,None],3,-1)
                    saved[h,view,key]=Image.fromarray((v*255).round().astype('uint8')).resize((640,360))
        return saved

    def render_targets(self,batch,processor):
        value=self.inner.render_targets(batch,processor);self.target_previews=self.previews(value)
        return value

    def __call__(self,prediction,batch,processor):
        from PIL import Image,ImageDraw
        value=self.inner(prediction,batch,processor);preview=self.previews(value)
        if self.label is not None:
            for h in SELECTED:
                for view in range(3):
                    canvas=Image.new('RGB',(2560,400),'white');draw=ImageDraw.Draw(canvas)
                    panels=[(self.target_previews,'rgb','GT GS RGB'),(preview,'rgb','G5 RGB'),
                            (self.target_previews,'alpha','GT Alpha'),(preview,'alpha','G5 Alpha')]
                    for i,(source,key,title) in enumerate(panels):
                        canvas.paste(source[h,view,key],(i*640,40));draw.text((i*640+8,8),title,fill='black')
                    draw.text((1100,8),f'{self.label} H{h} camera{view}; rendered 1280x720, preview 640x360',fill='black')
                    canvas.save(self.out/f'{self.label}_h{h}_camera{view}.jpg',quality=90)
            self.target_previews={}
        return value


def metric_numbers(loss,metrics,processor):
    finite_tree((loss,metrics))
    return dict(joint=float(loss.detach()),cd_mm=float(metrics['geometry'].mean())*processor.state_dict()['scale']*1000,
        rgb=float(metrics['rgb'].detach().mean()),alpha=float(metrics['alpha'].detach().mean()),
        rgb_by_horizon=metrics['rgb_by_horizon'][0].detach().cpu().tolist(),
        alpha_by_horizon=metrics['alpha_by_horizon'][0].detach().cpu().tolist())


def snapshot(model,batch,pred,trace,contexts,out,variant,window,stage,processor,tables):
    x=cpu_tree(batch['inputs']);tr=cpu_tree(trace);ctx=cpu_tree(contexts);factor=processor.state_dict()['scale']*1000
    geometry,propagation=collect(model,x,tr,ctx,variant=variant,window=window,stage=stage,factor=factor)
    tables['geometry'].extend(geometry);tables['propagation'].extend(propagation)
    field=tr['field'];keep=x['gs_mask'][0];valid_a=field.anchor_mask[0]
    initial=processor.denormalize_xyz(x['gs_xyz'],x['gs_mask'])[0,keep]
    world=processor.denormalize_xyz(pred['xyz'].detach().cpu(),x['gs_mask'][:,None].expand(-1,15,-1))[0,:,keep]
    arrays=dict(initial_world=initial.numpy(),prediction_world=world.numpy(),
        direct_seed=np.stack([s['direct_seed_per_hand'][0,valid_a].numpy() for s in tr['steps']]),
        local_correction=np.stack([s['local_correction'][0,valid_a].numpy() for s in tr['steps']]),
        global_delta=np.stack([s['g4_global_delta'][0].numpy() for s in tr['steps']]),
        bandwidth=field.bandwidth.numpy(),factor_mm=np.array(factor),anchor_indices=field.anchor_indices[0,valid_a].numpy(),
        mapping_context=np.stack([v[0,valid_a].numpy() for v in ctx]))
    for h in SELECTED:
        gt=batch['targets']['target_xyz'][h-1].detach().cpu();mask=batch['targets']['target_mask'][h-1].detach().cpu()
        arrays[f'target_h{h}_world']=processor.denormalize_xyz(gt,mask)[0,mask[0]].numpy()
    if variant=='propagate':
        arrays.update(propagated_seed=np.stack([s['propagated_seed_per_hand'][0,valid_a].numpy() for s in tr['steps']]),
            weights=np.stack([s['anchor_attention_weights'][0,valid_a].numpy() for s in tr['steps']]),
            neighbor_indices=tr['anchor_neighbor_indices'][0,valid_a].numpy(),neighbor_mask=tr['anchor_neighbor_mask'][0,valid_a].numpy())
    np.savez_compressed(out/'arrays'/f'{variant}_{window}_{stage}.npz',**arrays)
    return graph_diagnostics(x,tr,factor) if variant=='propagate' else None


def run_variant(variant,args,processor,cpu_batches,out,renderer,tables,source):
    directory=out/variant;directory.mkdir();torch.manual_seed(42);model=build_model(variant)
    shared_hash=state_hash({k:v for k,v in model.state_dict().items() if not k.startswith(('direct_motion.','propagation.'))})
    direct_hash=state_hash(model.direct_motion.state_dict());model.cuda().train()
    batch=to_device(cpu_batches['ep7_start105'],'cuda');opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    records=[];graphs={};start=time.perf_counter()
    for update in (1,2):
        print(f'{variant}: update {update}/2 begin',flush=True)
        opt.zero_grad(set_to_none=True);before={n:p.detach().clone() for n,p in model.named_parameters()}
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();clock=time.perf_counter()
        with mapping_context(model) as contexts:pred,trace=model(batch['inputs'],return_trace=True)
        finite_tree(pred);loss,metrics=v4_objective(pred,batch,processor,renderer);loss.backward()
        grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()};finite_tree(grads)
        raw_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();finite_tree(model.state_dict());finite_tree(opt.state_dict());torch.cuda.synchronize()
        record=dict(successful_update=update,**metric_numbers(loss,metrics,processor),raw_gradient_norm=float(raw_norm),
            clipped=bool(raw_norm>1),optimization_wall_seconds=time.perf_counter()-clock,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            groups=parameter_evidence(model,before,grads))
        if record['groups']['direct_output']['raw_gradient_norm']<=0 or record['groups']['direct_output']['actual_update_norm']<=0:
            raise AssertionError('Direct output projection did not learn')
        if variant=='propagate' and update==2:
            score=record['groups']['propagation_score']
            if score['raw_gradient_norm']<=0 or score['actual_update_norm']<=0:raise AssertionError('Score inactive on second real update')
        records.append(record)
        # Durable evidence before any optional analysis/plotting can fail.
        save_checkpoint(directory/('smoke.pt' if update==2 else 'update_1.pt'),model,opt,
            source_sha=source,processor_state=processor.state_dict(),successful_steps=update)
        atomic_json(directory/'progress.json',dict(successful_updates=update,updates=records))
        graph=snapshot(model,batch,pred,trace,contexts,out,variant,'ep7_start105',f'after_{update-1}',processor,tables)
        if graph:graphs[f'after_{update-1}']=graph
        print(f'{variant}: update {update}/2 complete; joint={record["joint"]:.8g}',flush=True)
        del pred,trace,contexts,loss,metrics,before,grads
    if len(records)!=2:raise AssertionError('Exactly two updates required')
    model.eval();frozen_hash=state_hash(model.state_dict());validation={};probe=None
    for window,cpu_batch in cpu_batches.items():
        current=batch if window=='ep7_start105' else to_device(cpu_batch,'cuda');renderer.label=f'{variant}_{window}'
        with torch.no_grad(),mapping_context(model) as contexts:
            pred,trace=model(current['inputs'],return_trace=True);finite_tree(pred)
            loss,metrics=v4_objective(pred,current,processor,renderer)
        renderer.label=None;validation[window]=metric_numbers(loss,metrics,processor)
        graph=snapshot(model,current,pred,trace,contexts,out,variant,window,'after_2',processor,tables)
        if graph:graphs[window+'_after_2']=graph
        if variant=='propagate' and window=='ep7_start105':
            s=trace['steps'][-1];hand=int(s['hand_valid'][0].nonzero()[0]);field=trace['field']
            probe=controlled_neighbor_probe(model.propagation,query=contexts[-1][:,:,hand,:128],
                hand_tokens=s['tokens62'][:,:2],basis18=s['anchor_hand_basis18'],seed=s['direct_seed_per_hand'],
                initial_xyz=current['inputs']['gs_xyz'][torch.arange(1,device=current['inputs']['gs_xyz'].device)[:,None],field.anchor_indices],
                current_xyz=s['anchor_xyz'],anchor_valid=field.anchor_mask,hand_valid=s['hand_valid'],
                indices=trace['anchor_neighbor_indices'],neighbor_mask=trace['anchor_neighbor_mask'],
                reference_weights=s['anchor_attention_weights'])
            atomic_json(directory/'controlled_neighbor_seed.json',probe)
        del pred,trace,contexts,loss,metrics
        if window!='ep7_start105':del current
    if state_hash(model.state_dict())!=frozen_hash:raise AssertionError('Inference modified model')
    path=directory/'smoke.pt'
    restored=build_model(variant).cuda().eval();opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-4)
    state=load_checkpoint(path,restored,opt2,processor_state=processor.state_dict(),source_sha=source)
    if state['successful_steps']!=2 or not same_tree(opt.state_dict(),opt2.state_dict()):raise AssertionError('Optimizer restore mismatch')
    with torch.no_grad():
        original=model(batch['inputs']);recovered=restored(batch['inputs']);exact={k:torch.equal(original[k],recovered[k]) for k in original}
    if not all(exact.values()):raise AssertionError('Checkpoint prediction changed')
    result=dict(passed=True,source_sha=source,initialization='fresh seed42; never G4 Best/Smoke initialization',
        initial_g4_shared_hash=shared_hash,initial_direct_hash=direct_hash,successful_updates=2,updates=records,
        frozen_inference=validation,validation_weights_unchanged=True,checkpoint_path=str(path),checkpoint_sha256=sha256(path),
        checkpoint_restore_exact=exact,optimizer_restore_exact=True,graph_diagnostics=graphs,controlled_neighbor_probe=probe,
        elapsed_seconds=time.perf_counter()-start)
    atomic_json(directory/'results.json',result)
    del model,restored,opt,opt2,batch,original,recovered,state;gc.collect();torch.cuda.empty_cache()
    return result


@torch.no_grad()
def equivalence(args,processor,cpu_batches,out):
    if sha256(args.g4_checkpoint)!=G4_HASH:raise AssertionError('Wrong frozen G4 checkpoint')
    g4=G4GlobalMotionPredictor(horizon=15,prediction_interval=1);state=load_for_evaluation(args.g4_checkpoint,g4,processor.state_dict())
    expected=state_hash(g4.state_dict());g4.cuda().eval();baselines={}
    for name,batch in cpu_batches.items():
        gpu=to_device(batch,'cuda');pred=g4(gpu['inputs']);finite_tree(pred);baselines[name]=cpu_tree(pred);del pred,gpu
    reports={}
    for variant in ('direct','propagate'):
        torch.manual_seed(42);model=build_model(variant);extra={k for k in model.state_dict() if k.startswith(('direct_motion.','propagation.'))}
        missing=model.load_state_dict(g4.state_dict(),strict=False)
        if set(missing.missing_keys)!=extra or missing.unexpected_keys:raise AssertionError('Unexpected shared transfer keys')
        if any(p.count_nonzero() for p in model.direct_motion.mapping[-1].parameters()):raise AssertionError('Local head not zero')
        if state_hash({k:v for k,v in model.state_dict().items() if k not in extra})!=expected:raise AssertionError('Shared transfer mismatch')
        model.cuda().eval();rows={}
        for name,batch in cpu_batches.items():
            gpu=to_device(batch,'cuda');pred=model(gpu['inputs']);finite_tree(pred)
            keys={k:dict(bit_exact=torch.equal(v.cpu(),baselines[name][k]),maximum_absolute_difference=float((v.cpu()-baselines[name][k]).abs().max())) for k,v in pred.items()}
            if not all(v['bit_exact'] for v in keys.values()):raise AssertionError('Real H15 G4/G5 equivalence failed: '+variant+'/'+name)
            rows[name]=keys;del gpu,pred
        reports[variant]=rows;del model
    if sha256(args.g4_checkpoint)!=G4_HASH:raise AssertionError('Original G4 checkpoint changed')
    result=dict(passed=True,checkpoint_path=args.g4_checkpoint,checkpoint_sha256=G4_HASH,
        checkpoint_identity=state['identity']['training'] if 'training' in state['identity'] else state['identity'].get('source_sha'),
        g4_shared_state_hash=expected,models=reports,scope='Isolated read-only compatibility; never saved as training initialization')
    atomic_json(out/'equivalence.json',result);del g4,state,baselines;gc.collect();torch.cuda.empty_cache()
    return result


def build_plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for window in ('ep7_start105','ep3_start135','ep6_start110'):
        arrays={v:np.load(out/'arrays'/f'{v}_{window}_after_2.npz') for v in ('direct','propagate')}
        init=arrays['direct']['initial_world'];targets=np.concatenate([arrays['direct'][f'target_h{h}_world'] for h in SELECTED])
        lo=np.minimum(init.min(0),targets.min(0));hi=np.maximum(init.max(0),targets.max(0))
        for a in arrays.values():lo=np.minimum(lo,a['prediction_world'].min((0,1)));hi=np.maximum(hi,a['prediction_world'].max((0,1)))
        center=(lo+hi)/2;extent=max(float((hi-lo).max()),.01)*.55
        fig,axes=plt.subplots(2,4,figsize=(17,8),subplot_kw={'projection':'3d'})
        for row,(variant,a) in enumerate(arrays.items()):
            for col,h in enumerate(SELECTED):
                ax=axes[row,col];gt=a[f'target_h{h}_world'];pred=a['prediction_world'][h-1]
                for points,label,color in [(init,'Copy','gray'),(gt,'GT','tab:green'),(pred,variant,'tab:orange')]:
                    points=points[::max(1,len(points)//1800)];ax.scatter(*points.T,s=1,label=label,c=color,alpha=.65)
                ax.set(xlim=(center[0]-extent,center[0]+extent),ylim=(center[1]-extent,center[1]+extent),zlim=(center[2]-extent,center[2]+extent),xlabel='World X / m',ylabel='World Y / m',zlabel='World Z / m',title=f'{variant} H{h}')
                ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=-65)
        axes[0,0].legend(markerscale=4);fig.suptitle(f'{window}: after exactly 2 diagnostic updates; fixed world scale, not performance acceptance')
        fig.tight_layout();fig.savefig(out/'figures'/f'{window}_geometry.png',dpi=140);plt.close(fig)
        for a in arrays.values():a.close()
    fig,axes=plt.subplots(2,2,figsize=(12,8));horizons=np.arange(1,16)
    for stage in (0,1,2):
        with np.load(out/'arrays'/f'propagate_ep7_start105_after_{stage}.npz') as a:
            d,p=a['direct_seed'],a['propagated_seed'];factor=float(a['factor_mm']);w=a['weights']
            axes[0,0].plot(horizons,np.sqrt((d*d).sum(-1).mean((1,2)))*factor,label=f'after {stage}')
            axes[0,1].plot(horizons,np.sqrt(((p-d)**2).sum(-1).mean((1,2)))*factor,label=f'after {stage}')
            axes[1,0].plot(horizons,w[...,0].mean((1,2)),label=f'after {stage}')
            axes[1,1].plot(horizons,-(w*np.log(np.maximum(w,np.finfo(w.dtype).tiny))).sum(-1).mean((1,2)),label=f'after {stage}')
    for ax,title in zip(axes.flat,('Direct seed RMS / mm','Propagated minus Direct RMS / mm','Self weight','Attention entropy / nats')):
        ax.set(xlabel='Horizon',title=title);ax.legend();ax.grid(alpha=.25)
    fig.suptitle('Train ep7/start105: before and after two diagnostic updates; not a performance comparison')
    fig.tight_layout();fig.savefig(out/'figures'/'propagation_learning.png',dpi=140);plt.close(fig)


def write_report(out,results,env,source,compat,tables):
    import html
    rows=['|模型|更新|Joint|CD mm|RGB|Alpha|Score 原始梯度|Score 实际更新|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant,r in results.items():
        for u in r['updates']:
            score=u['groups'].get('propagation_score',{})
            rows.append(f"|{variant}|{u['successful_update']}|{u['joint']:.7g}|{u['cd_mm']:.6g}|{u['rgb']:.6g}|{u['alpha']:.6g}|{score.get('raw_gradient_norm','—')}|{score.get('actual_update_norm','—')}|")
    far=[r for r in tables['geometry'] if r['region']=='far']
    max_ratio=max(r['pair_distance_over_bandwidth_max'] for r in far)
    max_weight=max(r['softmax_max_weight_max'] for r in far)
    max_geometry=max(r['geometry_abs_max'] for r in far)
    text=f'''# G5-v1 真实 CUDA/gsplat Smoke 与专项数值诊断

源码：{source}；Slurm Job：{env['job_id']}；GPU：{env['gpu']}。
环境：PyTorch {env['torch']} / CUDA {env['cuda']} / gsplat {env['gsplat']}。

## 工程验收

两个模型各 Seed42 从零初始化，仅 Train ep7/start105 各完成两次诊断更新。AdamW lr=1e-4、weight_decay=1e-4、clip_norm=1；原始 CD+RGB+0.1Alpha、三相机1280×720渲染。ep3/start135、ep6/start110仅冻结推理。模型、优化器、预测、Loss及Renderer均有限；独立Checkpoint恢复模型和Optimizer后，全部预测属性逐值精确相等。

'''+'\n'.join(rows)+f'''

两组公共G4参数及Direct Mapping初始哈希一致。训练指标表为每次更新前的同一Loss；诊断快照after_0/1/2分别为初始、第一次和第二次更新后的权重。图像为原始gsplat输出的缩略预览，不是真实RGB照片。

## 三项问题的证据

1. **有符号传播及边界**：固定Query、几何、目标Anchor基底，仅扰动邻居Seed，传播输出随之改变，Score权重逐值不变。当前Score没有直接读取邻居Seed；它仍可通过邻居Query间接获得其他信息，不能将其称为完全无邻居信息。正负Seed、传播差异、方向余弦及各手独立统计见 propagation_diagnostics.csv。
2. **Far尺度**：所有快照Far区的有效Controller对距离/bandwidth最大为 {max_ratio:.6g}，几何Context绝对最大为 {max_geometry:.6g}，几何Softmax最大单点权重为 {max_weight:.9g}。有效行仍归一化；没有触发非有限值或梯度失活。大距离/集中权重是空间分配及数值裕度风险，不等于已确认失稳，也不能据两次更新断言Far运动正确。区域严格复用G4初态GS距离四分位，Anchor继承对应GS区域，没有新增硬阈值。
3. **初期平滑与学习**：全零Seed时传播输出也为零，Score原始梯度允许为零；第一次实际更新可能仅含AdamW衰减。第二次Score原始梯度及实际更新非零，说明并非永久失活。逐时刻Self权重、Entropy、与同状态Uniform向量算子的差异均保留。Uniform仅是冻结的算子对照，不是第三个训练模型；不能推断学习型Attention已经优于固定平滑。

## 初始化兼容性

冻结G4 Best哈希 {G4_HASH}。只读复制共享参数到零局部分支后，ep3、ep6、ep7的真实H15全部属性逐值精确一致。此兼容性模型独立于从头Smoke，未另存为训练初始化；原始G4文件前后哈希不变。逐属性结果见 equivalence.json。

## 结论与限制

工程检查通过，第二步Score已实际参与优化。没有观察到必须修改模型才能继续的非有限值、掩码或等价性阻塞。Far归一化、距离尺度、固定空间图和平滑倾向仍须后续受控训练验证。图边不是材料拓扑，局部分支不保证纯形变或有符号奇对称；局部与G4全局修正可抵消且均值不强制为零。本轮没有正式八轮训练、没有性能提升或跨Episode泛化结论。

## 产物

- results.json：梯度、裁剪、AdamW实际更新及扣除纯衰减后的更新、Checkpoint哈希与环境。
- geometry_diagnostics.csv：H1/H5/H10/H15有效节点分区尺度、XYZ、Softmax及RBF后统计。
- propagation_diagnostics.csv：H1–H15各手分区权重、Seed、Uniform算子差异。
- arrays/：所有阶段预测与必要诊断张量；figures/：固定世界尺度几何图；images/：三相机RGB/Alpha预览。
- provenance/run_manifest.json：Split、Frame IDs、Mask、输入哈希、渲染配置与命令。
'''
    (out/'smoke_report.md').write_text(text,encoding='utf-8')
    gallery=''.join(f'<figure><figcaption>{html.escape(p.name)}</figcaption><img loading="lazy" src="{p.relative_to(out).as_posix()}"></figure>' for folder in ('figures','images') for p in sorted((out/folder).glob('*')))
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>G5 Smoke</title><style>body{max-width:1300px;margin:30px auto;font:16px sans-serif;background:#fafafa}pre{white-space:pre-wrap;line-height:1.6}img{width:100%;height:auto}figure{margin:25px 0}</style><pre>'+html.escape(text)+'</pre>'+gallery,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('direct','propagate','all'),required=True)
    for k in ('root','depth-root','output','expected-sha'):p.add_argument('--'+k,required=True)
    p.add_argument('--g4-checkpoint');args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires authorized Slurm CUDA allocation')
    if args.variant=='all' and not args.g4_checkpoint:raise ValueError('all requires frozen G4 checkpoint')
    source,hashes=source_preflight(args.expected_sha)
    if args.g4_checkpoint and sha256(args.g4_checkpoint)!=G4_HASH:raise AssertionError('Wrong frozen G4 checkpoint')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for name in ('arrays','images','figures','logs','provenance'):(out/name).mkdir()
    tables=dict(geometry=[],propagation=[]);results={}
    try:
        server_cpu_log=Path(str(out)+'.cpu-tests.log')
        if server_cpu_log.is_file():
            import shutil
            shutil.copyfile(server_cpu_log,out/'logs'/'server_cpu_tests.txt')
        import gsplat
        env=dict(torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if (env['torch'],env['cuda'],env['gsplat'])!=('2.7.1+cu128','12.8','1.4.0'):raise ValueError('Unreviewed CUDA/gsplat environment')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        processor=DataProcessor('001-rope');batches,manifest=load_windows(args,processor)
        atomic_json(out/'provenance'/'run_manifest.json',dict(source_sha=source,graph_baseline=BASE,graph_files_sha256=hashes,command_line=__import__('sys').argv,environment=env,windows=manifest,processor=processor.state_dict(),protocol=dict(horizon=15,prediction_interval=1,seed=42,updates_per_model=2,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=1.,loss='CD + RGB + 0.1 Alpha',render_settings=gs_render_settings())))
        renderer=AuditedRenderer(out/'images')
        for variant in ('direct','propagate') if args.variant=='all' else (args.variant,):
            results[variant]=run_variant(variant,args,processor,batches,out,renderer,tables,source)
            for window,batch in batches.items():
                if state_hash({k:v for k,v in batch['inputs'].items() if isinstance(v,torch.Tensor)})!=manifest[window]['input_hash']:
                    raise AssertionError('Input data changed during smoke')
            write_csv(out/'geometry_diagnostics.csv',tables['geometry']);write_csv(out/'propagation_diagnostics.csv',tables['propagation'])
        if args.variant=='all':
            if results['direct']['initial_g4_shared_hash']!=results['propagate']['initial_g4_shared_hash'] or results['direct']['initial_direct_hash']!=results['propagate']['initial_direct_hash']:raise AssertionError('Fresh paired initialization mismatch')
            compat=equivalence(args,processor,batches,out);build_plots(out);write_report(out,results,env,source,compat,tables)
        else:compat=None
        _,final_hashes=source_preflight(args.expected_sha)
        if final_hashes!=hashes:raise AssertionError('Frozen source graph changed')
        atomic_json(out/'results.json',dict(passed=True,source_sha=source,environment=env,variants=results,equivalence=compat,renderer_finite_calls=renderer.finite_calls,scope='Engineering and numerical diagnostics only; not training performance'))
        print('G5 CUDA/gsplat smoke completed successfully',flush=True)
    except BaseException:
        write_csv(out/'geometry_diagnostics.csv',tables['geometry']);write_csv(out/'propagation_diagnostics.csv',tables['propagation'])
        atomic_json(out/'failure.json',dict(source_sha=source,traceback=traceback.format_exc(),completed_variants=results));raise


if __name__=='__main__':main()
