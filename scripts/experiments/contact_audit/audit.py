"""Collect read-only episode evidence; no model, renderer, optimizer or CUDA."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def nearest(q, x):
    """Exact Euclidean NN for small CPU queries, world metres, float64."""
    q, x = np.asarray(q, dtype=np.float64), np.asarray(x, dtype=np.float64)
    if not len(q) or not len(x) or not np.isfinite(q).all() or not np.isfinite(x).all():
        raise ValueError('NN requires finite nonempty points')
    return np.concatenate([np.sqrt(((part[:, None] - x[None]) ** 2).sum(-1).min(1))
                           for part in np.array_split(q, max(1, (len(q) + 63) // 64))])


def tactile_stats(values, mask):
    v = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return dict(valid_count=int(len(v)), valid_ratio=float(len(v) / values.size),
                mean=float(v.mean()) if len(v) else None,
                maximum=float(v.max()) if len(v) else None,
                total=float(v.sum()) if len(v) else None,
                positive_count=int((v > 0).sum()),
                positive_ratio=float((v > 0).mean()) if len(v) else None)


def runs(values):
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            yield start, i - 1, values[start]
            start = i


def windows(labels, horizon=10):
    """Categories are explicit overlapping flags; confirmed count stays zero."""
    onset = [i for i in range(1, len(labels)) if labels[i] == 'CONTACT' and labels[i-1] != 'CONTACT']
    release = [i for i in range(1, len(labels)) if labels[i] != 'CONTACT' and labels[i-1] == 'CONTACT']
    entries = []
    for t in range(len(labels) - horizon):
        part = labels[t:t + horizon + 1]
        entries.append(dict(start_frame=t, condition_frames=list(range(t, t+horizon)),
            target_frames=list(range(t+1, t+horizon+1)),
            all_candidate_contact=all(x == 'CONTACT' for x in part),
            all_no_contact=all(x == 'NO_CONTACT' for x in part),
            includes_uncertain='UNCERTAIN' in part,
            onset_overlap=any(t < i <= t+horizon for i in onset),
            release_overlap=any(t < i <= t+horizon for i in release)))
    flags = ['all_candidate_contact', 'all_no_contact', 'includes_uncertain', 'onset_overlap', 'release_overlap']
    return dict(horizon=horizon, prediction_interval=1, interval_convention='closed [t,t+10]',
                categories_may_overlap=True, confirmed_contact_windows=0,
                review_status='pending researcher review; not Dataset labels',
                counts={k: sum(e[k] for e in entries) for k in flags}, windows=entries)


def project(xyz, intrinsic, camera_to_world):
    world_to_camera = np.linalg.inv(camera_to_world)
    xyz = np.asarray(xyz)
    camera = xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    hom = camera @ intrinsic.T
    valid = camera[:, 2] > 0
    uv = np.full((len(xyz), 2), np.nan)
    uv[valid] = hom[valid, :2] / hom[valid, 2:3]
    return uv, valid


def collect(args):
    from scripts.dataloader.dataset import _episode_info, _episode_conditions, _depth_info, read_gs
    from scripts.rebuild_dataset.rebuild import SourceReader, timestamp_array
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Collector requires an empty output directory')
    core = Path(args.datasets) / 'touch4d-v1/001-rope/episode_1'
    raw = Path(args.datasets) / 'deform360'
    prefix = 'processed/001-rope/episode_1'
    info = _episode_info(core, '001-rope', 1)
    if info.slots != (None, 0):
        raise ValueError('Unexpected episode1 physical right-hand mapping')
    c = _episode_conditions(info)
    depth = _depth_info(Path(args.datasets)/'touch4d-depth-v1/001-rope/episode_1', info, '001-rope')
    sources = [core/'metadata.json', core/'controller.npz', core/'tactile.npz', core/'timestamps.npz',
               *sorted((core/'gs').glob('*.npz')),
               depth.path/'metadata.json', depth.path/'camera_parameters.npz',
               raw/'raw/001-rope/metadata.json', raw/prefix/'metadata.json',
               raw/prefix/'split.json', raw/prefix/'robot/robot.npy']
    before = {str(p): sha(p) for p in sources}
    rows, geometry, previous = [], [], None
    fixed = np.load(core/'controller.npz', allow_pickle=False)
    times = np.load(core/'timestamps.npz', allow_pickle=False)
    for f in range(info.frames):
        gs = read_gs(info, f)
        x = gs['xyz'].astype(np.float64)
        q = c['controller_points'][f, 1].astype(np.float64)
        qm = c['controller_mask'][f, 1]
        d = np.full(30, np.nan)
        if qm.any():
            d[qm] = nearest(q[qm], x)
        ids = np.linspace(0, len(x)-1, min(512, len(x)), dtype=int)
        sample = x[ids]
        sampled_cd = None if previous is None else float((nearest(sample, previous).mean()+nearest(previous[np.linspace(0,len(previous)-1,min(512,len(previous)),dtype=int)], x).mean())/2*1000)
        row = dict(frame=f, timestamp_us=int(info.timestamps[f]),
            time_s=float((info.timestamps[f]-info.timestamps[0])/1e6),
            opening_m=float(c['openings'][f, 0]), right_pose_valid=bool(c['action_valid'][f, 1]),
            controller_valid_count=int(qm.sum()), controller_min_mm=float(np.nanmin(d)*1000) if qm.any() else None,
            controller_median_mm=float(np.nanmedian(d)*1000) if qm.any() else None,
            gs_count=len(x), gs_centroid_m=x.mean(0).tolist(), gs_bbox_m=np.ptp(x, axis=0).tolist(),
            gs_scale_p95_mm=float(np.percentile(np.exp(gs['log_scales']), 95)*1000),
            adjacent_sampled_set_cd_mm=sampled_cd)
        for cutoff in [5, 10, 20]:
            row[f'controller_within_{cutoff}mm_count'] = int((d[qm]*1000 <= cutoff).sum())
            row[f'controller_within_{cutoff}mm_ratio'] = float((d[qm]*1000 <= cutoff).mean()) if qm.any() else None
        for j in range(2):
            for key, value in tactile_stats(c['tactile'][f, j+2], c['valid_mask'][f, j+2]).items():
                row[f'finger{j}_{key}'] = value
            row[f'finger{j}_age_ms'] = float(info.ages[f, j+2]/1000) if info.ages[f,j+2] >= 0 else None
        for axis, val in zip('xyz', c['T_worlds'][f, 0, :3, 3]):
            row[f'gripper_{axis}_m'] = float(val)
        dt = None if f == 0 else (info.timestamps[f]-info.timestamps[f-1])/1e6
        row['gripper_speed_mm_s'] = None if f == 0 else float(np.linalg.norm(c['T_worlds'][f,0,:3,3]-c['T_worlds'][f-1,0,:3,3])/dt*1000)
        row['opening_rate_mm_s'] = None if f == 0 else float((c['openings'][f,0]-c['openings'][f-1,0])/dt*1000)
        pair = qm & c['controller_mask'][f-1,1] if f else np.zeros(30, bool)
        row['controller_step_rms_mm'] = float(np.sqrt(np.mean(np.sum((q[pair]-c['controller_points'][f-1,1,pair])**2,-1)))*1000) if pair.any() else None
        rows.append(row)
        geometry.append(dict(xyz=sample.round(6).tolist(), controller=q.round(6).tolist(), controller_mask=qm.tolist(),
            controller_distances_mm=[float(v*1000) if np.isfinite(v) else None for v in d]))
        previous = x
    # Retain the small full-resolution tactile/robot arrays, not all GS or HDF5.
    np.savez_compressed(out/'observations.npz', tactile=c['tactile'][:,2:], valid_mask=c['valid_mask'][:,2:],
        tactile_age_us=info.ages[:,2:], timestamps_us=info.timestamps,
        controller_points=c['controller_points'][:,1], controller_mask=c['controller_mask'][:,1],
        T_worlds=c['T_worlds'][:,0], openings=c['openings'][:,0],
        tactile_raw_index=times['tactile_raw_index'][:,2:])
    dump(out/'frame_evidence.json', rows)
    dump(out/'geometry.json', geometry)
    metadata = {str(p): json.loads(p.read_text()) for p in sources if p.suffix == '.json'}
    robot = np.load(raw/prefix/'robot/robot.npy', allow_pickle=True).item()
    inventory = dict(metadata=metadata, raw_robot_fields={k: dict(shape=list(np.asarray(v).shape), dtype=str(np.asarray(v).dtype)) for k,v in robot.items()},
        reconstructed_schemas={str(core/name): {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in np.load(core/name,allow_pickle=False).items()} for name in ['tactile.npz','controller.npz','timestamps.npz']},
        finding='No explicit contact labels found in inspected sources.',
        scope='Raw scene metadata; processed episode metadata/split/robot; core metadata/NPZ and depth metadata/calibration. Not a claim about every unpublished dataset source.',
        caution='processed start_frame/end_frame and split are selection/split metadata, not explicitly contact-labelled; valid masks are observation validity.')
    dump(out/'label_source_inventory.json', inventory)
    cameras = args.cameras.split(',')
    calibration = {}
    rgb_manifest = {}
    with SourceReader(raw) as reader:
        for camera in cameras:
            if camera not in depth.cameras:
                raise ValueError(camera)
            stamps = timestamp_array(reader.read(camera+'_time', prefix+'/cameras.tar', camera+'/aligned_timestamps.txt'), strict=True)
            if not np.array_equal(stamps, info.timestamps):
                raise ValueError('RGB timeline mismatch')
            payload = reader.read(camera+'_rgb', prefix+'/cameras.tar', camera+'/undistorted.mp4')
            directory = out/'rgb'/camera
            directory.mkdir(parents=True)
            video = directory/'source_undistorted.mp4'
            video.write_bytes(payload)
            subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(video),
                '-vf','scale=640:360','-threads','2','-vsync','0','-start_number','0','-q:v','3',str(directory/'%06d.jpg')], check=True)
            count = len(list(directory.glob('*.jpg')))
            if count != info.frames:
                raise ValueError(f'RGB decoded {count}, expected {info.frames}')
            j = depth.cameras.index(camera)
            calibration[camera] = dict(intrinsics=depth.intrinsics[j].tolist(), camera_to_world=depth.camera_to_world[j].tolist(), width=depth.width,height=depth.height)
            rgb_manifest[camera] = dict(frames=count, timestamps_exact_match=True,kind='real observation RGB, undistorted; not GS render',display_resolution=[640,360])
        dump(out/'rgb_source_refs.json', reader.refs)
    dump(out/'calibration.json', calibration)
    after = {str(p):sha(p) for p in sources}
    if before != after:
        raise RuntimeError('Source data changed during audit')
    dump(out/'source_integrity.json', dict(before=before,after=after,unchanged=True))
    dump(out/'run_manifest.json',dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=[sys.executable,*sys.argv],device='CPU',frames=info.frames,span_seconds=rows[-1]['time_s'],
        timestamps_unit='microseconds',world_unit='metres; derived distances in mm',
        tactile_unit='raw positive median-subtracted sensor response; not calibrated force, no normalization',
        hand='right only, robot slot 0, tactile stream indices 2/3, 30 controllers',
        gs_change='Independent point sets. Approximate adjacent set CD using 512 deterministic samples; not material displacement.',
        rgb=rgb_manifest,source_files_verified=len(before)))
    print(json.dumps({'frames':info.frames,'output':str(out),'source_integrity':True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cameras',default='brics-odroid-001_cam0,brics-odroid-010_cam0,brics-odroid-021_cam0')
    collect(parser.parse_args())
