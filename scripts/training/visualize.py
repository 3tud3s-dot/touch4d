"""Export one fixed six-frame window as an offline, dependency-free HTML viewer."""

import argparse
import html
import json
from pathlib import Path
import struct
import zlib
import numpy as np
import torch

from ..dataloader.collate import collate_fn
from .evaluate import load_model
from .rendering import DepthRenderer, GaussianRenderer
from .train import build_loader, to_device, copy_initial


def png_bytes(rgb):
    """Small RGB PNG encoder, so visualization needs no extra image packages."""
    height, width, _ = rgb.shape
    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag+data) & 0xffffffff)
    scanlines = b''.join(b'\0' + row.tobytes() for row in rgb.astype(np.uint8))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def depth_rgb(depth, mask, low, high, error=False):
    value = np.clip((depth-low)/max(high-low, 1e-12), 0, 1)
    if error:
        rgb = np.stack((np.minimum(2*value, 1), 1-np.abs(2*value-1), np.minimum(2*(1-value), 1)), -1)
    else:
        rgb = np.stack((value, np.sqrt(value), 1-value), -1)
    return np.where(mask[..., None], np.rint(rgb*255), 20).astype(np.uint8)


def viewer_html(data, title):
    # Data contains only numeric arrays and controlled camera IDs, never executable text.
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GS 预测对比</title>
<style>body{font:14px system-ui;background:#f5f7fa;color:#172437;margin:20px}h1{font-size:20px}.row{display:flex;gap:12px}.panel{flex:1;background:white;padding:10px;border-radius:8px;min-width:0}canvas{width:100%;height:270px;touch-action:none}img{width:100%}button,select,input{margin:8px}small{color:#64748b}#info{margin:10px 0}</style>
<h1>''' + html.escape(title) + '''</h1><p>上方为 GS 中心点对比（不是完整高斯光栅图）；三窗共用世界坐标、相机与尺度。拖动旋转，滚轮缩放。下方深度使用真实相机；黑色表示无有效值，误差图只计算目标有效区域。</p>
<button id="play">播放</button><input id="time" type="range" min="0" max="4" value="0"><span id="label"></span><select id="camera"></select><div id="info"></div>
<div class="row"><div class="panel">复制初始 GS<canvas id="initial"></canvas></div><div class="panel">预测 GS<canvas id="prediction"></canvas></div><div class="panel">真实目标 GS<canvas id="target"></canvas></div></div>
<div class="row"><div class="panel">目标深度<img id="gt"></div><div class="panel">预测深度<img id="pred"></div><div class="panel">复制初态深度<img id="copy"></div><div class="panel">预测误差（±5 mm 截断）<img id="err"></div></div>
<small>深度零值不是背景或遮挡的确定标签。各时刻目标点数可以不同，没有逐点 ID 对应。可视化显示坐标做六位小数舍入；prediction.npz 保留原始 float32。</small>
<script>const D=''' + encoded + ''';
const slider=document.getElementById('time'),camera=document.getElementById('camera');
D.cameras.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c;camera.appendChild(o)});
let yaw=.4,pitch=.5,zoom=1,drag=null,timer=null;
const all=[D.initial,...D.prediction,...D.target].flat(),lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
all.forEach(p=>p.forEach((v,i)=>{lo[i]=Math.min(lo[i],v);hi[i]=Math.max(hi[i],v)}));
const center=lo.map((v,i)=>(v+hi[i])/2),radius=Math.max(...lo.map((v,i)=>hi[i]-v),.001);
function draw(id,pts,colored){let c=document.getElementById(id),r=c.getBoundingClientRect();c.width=Math.round(r.width*devicePixelRatio);c.height=Math.round(r.height*devicePixelRatio);let ctx=c.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);let s=Math.min(c.width,c.height)*.8/radius*zoom,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
let projected=pts.map((p,i)=>{let [x,y,z]=p.map((v,j)=>v-center[j]);let u=cy*x-sy*y,v=sy*x+cy*y;return [u,cp*z-sp*v,sp*z+cp*v,i]}).sort((a,b)=>a[2]-b[2]);
projected.forEach(([x,y,z,i])=>{let color=colored?D.colors[i]:[.25,.42,.62];ctx.fillStyle=`rgb(${color.map(v=>Math.round(v*255)).join(',')})`;ctx.beginPath();ctx.arc(c.width/2+x*s,c.height/2-y*s,1.3*devicePixelRatio,0,Math.PI*2);ctx.fill()});}
function update(){let k=+slider.value,v=+camera.value;document.getElementById('label').textContent=`预测 t+${k+1} / 原始帧 ${D.frame_ids[k+1]}`;document.getElementById('info').textContent=`初始/预测点数 ${D.initial.length}；目标点数 ${D.target[k].length}；深度颜色范围 ${D.depth_range.map(x=>x.toFixed(3)).join('–')} m`;
draw('initial',D.initial,true);draw('prediction',D.prediction[k],true);draw('target',D.target[k],false);for(let name of ['gt','pred','copy','err'])document.getElementById(name).src=`figures/frame_${k+1}_view_${v}_${name}.png`;}
document.querySelectorAll('canvas').forEach(c=>{c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointercancel=()=>drag=null;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.01;pitch+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];update()}};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.1,Math.min(10,zoom*Math.exp(-e.deltaY*.001)));update()}});
slider.oninput=update;camera.onchange=update;window.onresize=update;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{slider.value=(+slider.value+1)%5;update()},500)};update();</script></html>'''


def gs_viewer_html(data, title):
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GS 渲染对比</title>
<style>body{font:14px system-ui;background:#f5f7fa;color:#172437;margin:20px}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.panel{background:white;padding:10px;border-radius:8px}img{width:100%}select,input,button{margin:8px}</style><h1>''' + html.escape(title) + '''</h1>
<p>真实高斯光栅渲染，零阶 SH、统一黑色背景。颜色、不透明度继承初态；位置、尺度、旋转随预测更新。误差仅在目标 HDF5 有效区域计算，无效区域显示灰色，不是背景标签。</p>
<button id="play">播放</button><input id="time" type="range" min="0" max="4" value="0"><span id="label"></span><select id="camera"></select><select id="mode"><option value="rgb">RGB</option><option value="alpha">Alpha</option><option value="depth">期望深度</option></select><p id="info"></p>
<div class="row"><div class="panel">目标 GS<img id="gt"></div><div class="panel">预测 GS<img id="pred"></div><div class="panel">复制初态 GS<img id="copy"></div><div class="panel">预测误差<img id="err"></div></div>
<p>RGB误差：通道平均绝对误差，0–0.1；alpha误差：0–0.1；深度误差：与HDF5比较，±5mm。深度面板前三图来自GS渲染。颜色仅用于显示，完整世界坐标GS保存在 prediction.npz。</p>
<script>const D=''' + encoded + ''';const $=id=>document.getElementById(id);let timer=null;
D.cameras.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c;$('camera').append(o)});
function update(){let k=+$('time').value,v=+$('camera').value,m=$('mode').value;$('label').textContent=`t+${k+1} / 原始帧 ${D.frame_ids[k+1]}`;$('info').textContent=`初始/预测高斯数 ${D.initial_count}；目标高斯数 ${D.target_counts[k]}；深度显示范围 ${D.depth_range.map(x=>x.toFixed(3)).join('–')}m`;for(let n of ['gt','pred','copy','err'])$(n).src=`figures/frame_${k+1}_view_${v}_${m}_${n}.png`;}
$('time').oninput=$('camera').onchange=$('mode').onchange=update;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{$('time').value=(+$('time').value+1)%5;update()},500)};update();</script></html>'''


def export_gaussians(args, config, raw, batch, prediction, processor):
    settings = config.get('render_settings')
    size = (settings['height'], settings['width']) if settings is not None else None
    renderer = GaussianRenderer(image_size=size)
    renders = dict(pred=renderer(prediction, batch, processor),
                   copy=renderer(copy_initial(batch['inputs'], 'gs_geometry'), batch, processor),
                   gt=renderer.render_targets(batch, processor))
    for value in renders.values():
        if any(not torch.isfinite(image).all() for image in value.values()):
            raise FloatingPointError('Nonfinite rendered image; no visualization written')
    directory = Path(args.output).resolve() if args.output else Path(args.checkpoint).resolve().parent / 'visualizations' / f'{args.split}-episode-{args.episode}-frame-{args.start}'
    print(f'Creating {directory}: prediction.npz, index.html, figures/frame_{{1..5}}_view_{{v}}_{{rgb,alpha,depth}}_{{gt,pred,copy,err}}.png', flush=True)
    keep = batch['inputs']['gs_mask'][0]
    arrays = {'frame_ids': raw['meta']['frame_ids'][0].numpy(),
              'initial_xyz': raw['inputs']['gs_xyz'][0, keep.cpu()].numpy(),
              'initial_attributes': raw['inputs']['gs_attr'][0, keep.cpu()].numpy()}
    for field, value in prediction.items():
        if field == 'xyz':
            value = processor.denormalize_xyz(value, batch['inputs']['gs_mask'][:, None].expand(-1, 5, -1))
        elif field == 'log_scales':
            value = value + np.log(processor.state_dict()['scale'])
        arrays['predicted_'+field] = value[0, :, keep].cpu().numpy()
    for k, (xyz, attr, mask) in enumerate(zip(raw['targets']['target_xyz'], raw['targets']['target_attr'], raw['targets']['target_mask'])):
        arrays[f'target_xyz_{k+1}'] = xyz[0, mask[0]].numpy()
        arrays[f'target_attr_{k+1}'] = attr[0, mask[0]].numpy()
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite GS output; no visualization written')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'figures').mkdir()
    np.savez_compressed(directory/'prediction.npz', **arrays)
    valid = raw['targets']['target_depth_valid'][0].numpy()
    depth = raw['targets']['target_depth'][0].numpy()
    lo, hi = np.percentile(depth[valid], [1, 99]) if valid.any() else (0., 1.)
    images = {name: {key: image[0].cpu().numpy() for key, image in r.items()} for name, r in renders.items()}
    for k in range(5):
        for v in range(depth.shape[1]):
            output = {}
            for name, value in images.items():
                output['rgb_'+name] = np.rint(np.clip(value['rgb'][k, v], 0, 1)*255).astype(np.uint8)
                alpha = np.rint(np.clip(value['alpha'][k, v], 0, 1)*255).astype(np.uint8)
                output['alpha_'+name] = np.repeat(alpha[..., None], 3, axis=-1)
                output['depth_'+name] = depth_rgb(value['depth'][k, v], value['alpha'][k, v] > 0, lo, hi)
            for mode in ('rgb', 'alpha'):
                delta = np.abs(images['pred'][mode][k, v]-images['gt'][mode][k, v])
                if mode == 'rgb': delta = delta.mean(-1)
                output[mode+'_err'] = depth_rgb(delta, valid[k, v], 0., .1)
            output['depth_err'] = depth_rgb((images['pred']['depth'][k, v]-depth[k, v])*1000, valid[k, v], -5., 5., True)
            for name, image in output.items():
                (directory/'figures'/f'frame_{k+1}_view_{v}_{name}.png').write_bytes(png_bytes(image))
    data = dict(cameras=raw['meta']['depth_camera_ids'][0], frame_ids=arrays['frame_ids'].tolist(),
                initial_count=int(keep.sum()), target_counts=raw['targets']['target_lengths'][0].tolist(),
                depth_range=[float(lo), float(hi)])
    (directory/'index.html').write_text(gs_viewer_html(data, f'{config["scene"]} · episode {args.episode} · start {args.start}'), encoding='utf-8')
    print(directory/'index.html')


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--start', type=int, required=True, help='Existing window start in the selected split')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', help='New directory; default under the checkpoint run directory')
    args = p.parse_args(argv)
    device = torch.device(args.device)
    model, processor, config = load_model(args.checkpoint, device)
    ds = build_loader(config, args.split).dataset
    # Overfit Subset is a training sampling option, not a visualization restriction.
    if hasattr(ds, 'dataset'):
        ds = ds.dataset
    try:
        index = ds.windows.index((args.episode, args.start))
    except ValueError as exc:
        raise ValueError('Requested episode/start is not a valid window of this split') from exc
    raw = collate_fn([ds[index]])
    batch = processor(to_device(raw, device))
    prediction = model(batch['inputs'])
    if isinstance(prediction, dict):
        export_gaussians(args, config, raw, batch, prediction, processor)
        return 0
    renderer = DepthRenderer()
    pred_depth = renderer(prediction, batch, processor)[0].cpu().numpy()
    copy_depth = renderer(batch['inputs']['gs_xyz'][:, None].expand(-1, 5, -1, -1), batch, processor)[0].cpu().numpy()
    keep = batch['inputs']['gs_mask'][:, None].expand(-1, 5, -1)
    predicted_world = processor.denormalize_xyz(prediction, keep)[0, :, keep[0, 0]].cpu().numpy()
    initial = raw['inputs']['gs_xyz'][0, raw['inputs']['gs_mask'][0]].numpy()
    target = [x[0, mask[0]].numpy() for x, mask in zip(raw['targets']['target_xyz'], raw['targets']['target_mask'])]
    target_depth = raw['targets']['target_depth'][0].numpy()
    valid = raw['targets']['target_depth_valid'][0].numpy()
    if not np.isfinite(predicted_world).all() or not np.isfinite(pred_depth).all():
        raise FloatingPointError('Nonfinite prediction; no visualization written')
    directory = Path(args.output).resolve() if args.output else Path(args.checkpoint).resolve().parent / 'visualizations' / f'{args.split}-episode-{args.episode}-frame-{args.start}'
    print(f'Creating {directory}: prediction.npz, index.html, figures/frame_{{1..5}}_view_{{v}}_{{gt,pred,copy,err}}.png', flush=True)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'figures').mkdir()
    np.savez_compressed(directory / 'prediction.npz', initial_xyz=initial, predicted_xyz=predicted_world,
        initial_attributes=raw['inputs']['gs_attr'][0, raw['inputs']['gs_mask'][0]].numpy(),
        frame_ids=raw['meta']['frame_ids'][0].numpy(), **{f'target_xyz_{k+1}': x for k, x in enumerate(target)})
    lo, hi = np.percentile(target_depth[valid], [1, 99]) if valid.any() else (0., 1.)
    for k in range(5):
        for v in range(target_depth.shape[1]):
            images = {'gt': depth_rgb(target_depth[k, v], valid[k, v], lo, hi),
                      'pred': depth_rgb(pred_depth[k, v], pred_depth[k, v] > 0, lo, hi),
                      'copy': depth_rgb(copy_depth[k, v], copy_depth[k, v] > 0, lo, hi),
                      'err': depth_rgb((pred_depth[k, v]-target_depth[k, v])*1000, valid[k, v], -5, 5, error=True)}
            for name, image in images.items():
                (directory / 'figures' / f'frame_{k+1}_view_{v}_{name}.png').write_bytes(png_bytes(image))
    colors = (raw['inputs']['gs_attr'][0, raw['inputs']['gs_mask'][0], 8:11].numpy()*.28209479177387814+.5).clip(0, 1)
    data = dict(initial=initial.round(6).tolist(), prediction=predicted_world.round(6).tolist(),
                target=[x.round(6).tolist() for x in target], colors=colors.round(4).tolist(),
                cameras=raw['meta']['depth_camera_ids'][0], frame_ids=raw['meta']['frame_ids'][0].tolist(),
                depth_range=[float(lo), float(hi)])
    (directory / 'index.html').write_text(viewer_html(data, f'{config["scene"]} · episode {args.episode} · start {args.start}'), encoding='utf-8')
    print(directory / 'index.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
