"""CPU-only ten-episode evidence export reusing the ep1 audit/data contracts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from scripts.experiments.contact_audit.audit import sha, dump, nearest, tactile_stats, runs
from scripts.experiments.contact_audit.report import infer


def boundary_stats(start, end, timestamps):
    n=len(timestamps)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= n:
        raise ValueError('Require integer 0 <= start < end <= total_frames')
    # Last observed sample time is known; the terminal exclusive timestamp is not.
    span=(int(timestamps[end-1])-int(timestamps[start]))/1e6
    duration=(int(timestamps[end])-int(timestamps[start]))/1e6 if end<n else None
    return dict(before=start,contact=end-start,after=n-end,h10_windows=max(0,end-start-10),
                observed_span_s=span,boundary_duration_s=duration,terminal_boundary_unknown=end==n)


def template(manifest):
    return dict(scene='001-rope',boundary_convention='half_open',annotation_method='manual_two_cuts',
        episodes={str(e['episode']):dict(contact_start=None,contact_end_exclusive=None,
            review_status='UNREVIEWED',notes='',total_frames=e.get('frames')) for e in manifest['episodes']})


def collect_episode(ep,args,output):
    from scripts.dataloader.dataset import _episode_info,_episode_conditions,read_gs
    from scripts.rebuild_dataset.rebuild import SourceReader,timestamp_array
    core=Path(args.datasets)/'touch4d-v1/001-rope'/f'episode_{ep}'
    raw=Path(args.datasets)/'deform360';prefix=f'processed/001-rope/episode_{ep}'
    info=_episode_info(core,'001-rope',ep);conditions=_episode_conditions(info)
    hands=[1] if ep<5 else [0,1]
    for hand in hands:
        if info.slots[hand] is None:raise ValueError('Operation hand missing from physical metadata')
    files=[core/'metadata.json',core/'controller.npz',core/'tactile.npz',core/'timestamps.npz',*sorted((core/'gs').glob('*.npz'))]
    before={str(p):sha(p) for p in files}
    directory=output/'episodes'/f'ep{ep}';directory.mkdir(parents=True)
    rows=[];geometry=[]
    for f in range(info.frames):
        gs=read_gs(info,f);x=gs['xyz'].astype(np.float64)
        row=dict(frame=f,timestamp_us=int(info.timestamps[f]),time_s=float((info.timestamps[f]-info.timestamps[0])/1e6),
                 gs_count=len(x),centroid=x.mean(0).tolist(),hands=[])
        g=dict(xyz=x[np.linspace(0,len(x)-1,min(384,len(x)),dtype=int)].round(6).tolist(),controllers=[])
        dt=(info.timestamps[f]-info.timestamps[f-1])/1e6 if f else None
        for hand in hands:
            slot=info.slots[hand];q=conditions['controller_points'][f,hand];qm=conditions['controller_mask'][f,hand]
            distances=nearest(q[qm],x)*1000 if qm.any() else np.array([])
            previous=conditions['action_valid'][f-1,hand] if f else False
            pose_valid=bool(conditions['action_valid'][f,hand])
            pair=qm & conditions['controller_mask'][f-1,hand] if f else np.zeros(30,bool)
            h=dict(hand=hand,slot=slot,pose_valid=pose_valid,controller_valid_count=int(qm.sum()),
                opening_mm=float(conditions['openings'][f,slot]*1000) if pose_valid else None,
                min_mm=float(distances.min()) if len(distances) else None,
                median_mm=float(np.median(distances)) if len(distances) else None,
                speed_mm_s=float(np.linalg.norm(conditions['T_worlds'][f,slot,:3,3]-conditions['T_worlds'][f-1,slot,:3,3])/dt*1000) if f and previous and pose_valid else None,
                controller_step_rms_mm=float(np.sqrt(np.mean(np.sum((q[pair]-conditions['controller_points'][f-1,hand,pair])**2,axis=-1)))*1000) if pair.any() else None,
                fingers=[])
            for stream in [2*hand,2*hand+1]:
                values=conditions['tactile'][f,stream];mask=conditions['valid_mask'][f,stream]
                s=tactile_stats(values,mask);s['stream']=stream;s['age_ms']=float(info.ages[f,stream]/1000) if info.ages[f,stream]>=0 else None
                h['fingers'].append(s)
            row['hands'].append(h)
            g['controllers'].append(dict(hand=hand,xyz=q.round(6).tolist(),mask=qm.tolist()))
        rows.append(row);geometry.append(g)
    streams=[i for h in hands for i in [h*2,h*2+1]]
    tactile=conditions['tactile'][:,streams];valid=conditions['valid_mask'][:,streams]
    # Preserve exact full-resolution arrays; JSON is only a browser view.
    np.savez_compressed(directory/'observations.npz',tactile=tactile,valid_mask=valid,
        age_us=info.ages[:,streams],timestamp_us=info.timestamps,controller_points=conditions['controller_points'][:,hands],
        controller_mask=conditions['controller_mask'][:,hands],T_worlds=conditions['T_worlds'],openings=conditions['openings'])
    references=[];warnings=[]
    for k,hand in enumerate(hands):
        adapted=[]
        for row in rows:
            h=row['hands'][k]
            r=dict(controller_min_mm=h['min_mm'],right_pose_valid=h['pose_valid'],controller_valid_count=h['controller_valid_count'])
            for j,s in enumerate(h['fingers']):
                r[f'finger{j}_mean']=s['mean'];r[f'finger{j}_maximum']=s['maximum']
            adapted.append(r)
        try:
            a=infer(adapted,dict(tactile=tactile[:,2*k:2*k+2],valid_mask=valid[:,2*k:2*k+2]))
            references.append(dict(hand=hand,thresholds=a['thresholds'],background_frames=a['background_frames'],
                intervals=[[s,e+1] for s,e,v in runs(a['tactile_sustained']) if v],
                meaning='automatic tactile reference only; not prefilled manual labels'))
        except ValueError as exc:
            references.append(dict(hand=hand,intervals=[],warning=str(exc)))
    cameras=[];rgb_refs={}
    with SourceReader(raw) as reader:
        for camera in args.cameras.split(','):
            try:
                stamps=timestamp_array(reader.read(camera+'_timestamps',prefix+'/cameras.tar',camera+'/aligned_timestamps.txt'),strict=True)
                if not np.array_equal(stamps,info.timestamps):raise ValueError('RGB timestamps disagree')
                rgbdir=directory/'rgb'/camera;rgbdir.mkdir(parents=True)
                reuse=Path(args.ep1_audit)/'rgb'/camera if ep==1 and args.ep1_audit else None
                if reuse and all((reuse/f'{f:06d}.jpg').is_file() for f in range(info.frames)):
                    old=json.loads((Path(args.ep1_audit)/'run_manifest.json').read_text())
                    oldrefs=json.loads((Path(args.ep1_audit)/'rgb_source_refs.json').read_text())
                    if old['frames']!=info.frames or oldrefs[camera+'_time']['sha256']!=reader.refs[camera+'_timestamps']['sha256']:
                        raise ValueError('Reused ep1 RGB timeline/source mismatch')
                    for f in range(info.frames):shutil.copyfile(reuse/f'{f:06d}.jpg',rgbdir/f'{f:06d}.jpg')
                    rgb_refs[camera]=dict(kind='real undistorted RGB; reused ep1 audit',source=str(reuse),original=oldrefs[camera+'_rgb'])
                else:
                    payload=reader.read(camera+'_video',prefix+'/cameras.tar',camera+'/undistorted.mp4')
                    # Seekable temporary MP4 also handles files with moov at EOF.
                    with tempfile.NamedTemporaryFile(suffix='.mp4') as stream:
                        stream.write(payload);stream.flush()
                        subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',stream.name,
                            '-vf','scale=640:360','-threads','2','-vsync','0','-start_number','0','-q:v','4',str(rgbdir/'%06d.jpg')],check=True)
                    rgb_refs[camera]=dict(kind='real undistorted RGB',source=reader.refs[camera+'_video'])
                if len(list(rgbdir.glob('*.jpg')))!=info.frames:raise ValueError('Incomplete RGB frame count')
                cameras.append(camera)
            except (OSError,ValueError,subprocess.CalledProcessError) as exc:
                warnings.append(f'RGB {camera}: {type(exc).__name__}: {exc}')
        rgb_refs['timestamp_sources']=reader.refs
    after={str(p):sha(p) for p in files}
    if before!=after:raise ValueError('Source changed during export')
    dump(directory/'source_integrity.json',dict(before=before,after=after,unchanged=True))
    dump(directory/'rgb_sources.json',rgb_refs)
    data=dict(episode=ep,frames=info.frames,hands=hands,streams=streams,timestamps=info.timestamps.tolist(),
        rows=rows,geometry=geometry,tactile=tactile.reshape(info.frames,len(streams),384).tolist(),
        valid=valid.reshape(info.frames,len(streams),384).astype(int).tolist(),
        cameras=cameras,references=references,warnings=warnings,
        pressure_max=float(tactile.max()),source=str(core),rgb_kind='real camera RGB, undistorted; not GS render')
    dump(directory/'data.json',data)
    # A classic script works on file:// without fetch/CORS or a local server.
    (directory/'data.js').write_text(f'window.CONTACT_EPISODES[{ep}]='+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    return dict(episode=ep,frames=info.frames,span_s=rows[-1]['time_s'],hands=hands,cameras=cameras,
                status='READY' if len(cameras)>=2 else 'INCOMPLETE',warnings=warnings,source=str(core),
                timestamp_first=int(info.timestamps[0]),timestamp_last=int(info.timestamps[-1]),files_verified=len(files))


def main(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('New empty output required')
    manifest=dict(scene='001-rope',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=[sys.executable,*sys.argv],device='CPU',episodes=[],
        units='positions m; distance/motion mm; sensor raw median-subtracted response, not calibrated force',
        boundary_convention='half_open',annotation_method='manual_two_cuts',
        source_note='Per-frame GS independent reconstruction; nearest centers not a collision/contact detector.')
    for ep in range(10):
        try:entry=collect_episode(ep,args,output)
        except Exception as exc:entry=dict(episode=ep,status='ERROR',frames=None,warnings=[f'{type(exc).__name__}: {exc}'])
        manifest['episodes'].append(entry)
        dump(output/'audit_manifest.json',manifest)
        print(json.dumps(entry),flush=True)
    dump(output/'contact_boundaries_template.json',template(manifest))
    source=Path(__file__).parent
    assets=output/'assets';assets.mkdir()
    for name in ['app.js','core.js','style.css']:
        shutil.copyfile(source/name,assets/name)
    shutil.copyfile(source/'index.html',output/'index.html')
    shutil.copyfile(source/'README.md',output/'README.md')
    (assets/'manifest.js').write_text('window.CONTACT_MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';window.CONTACT_EPISODES={};',encoding='utf8')
    if any(e['status']!='READY' for e in manifest['episodes']):raise SystemExit('Some episodes incomplete; see audit_manifest.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ep1-audit',default='')
    p.add_argument('--cameras',default='brics-odroid-001_cam0,brics-odroid-010_cam0,brics-odroid-021_cam0')
    main(p.parse_args())
