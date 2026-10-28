"""Read-only CPU validity census for the actual frozen supervision cameras."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, depth_root):
    manifest = json.loads((root / 'contact_window_manifest.json').read_text())
    config = json.loads((root / 'short_experiments/S5/config.json').read_text())
    cameras = config['experiment']['cameras']
    windows = {s: manifest['windows'][s] for s in ('train', 'val')}
    counts, rows, files = {}, {}, {}
    for ep in sorted({e for group in windows.values() for e, _ in group}):
        used = sorted({t+h for group in windows.values() for e,t in group if e == ep for h in range(1,11)})
        directory = depth_root / '001-rope' / f'episode_{ep}'
        metadata = directory / 'metadata.json'
        calibration = directory / 'camera_parameters.npz'
        before = {str(p): sha(p) for p in (metadata, calibration)}
        episode = {}
        for camera in cameras:
            path = directory / camera / 'rendered_depth.h5'
            before_stat = (path.stat().st_size, path.stat().st_mtime_ns)
            by_frame = {}
            with h5py.File(path, 'r') as f:
                data = f['data']
                assert data.dtype == np.uint16 and data.ndim == 3
                pixels = int(np.prod(data.shape[1:]))
                for start in range(0, len(used), 24):
                    ids = used[start:start+24]
                    values = data[ids]
                    valid = np.count_nonzero(values, axis=(1,2))
                    by_frame.update(zip(ids, map(int, valid)))
            assert before_stat == (path.stat().st_size, path.stat().st_mtime_ns)
            values = list(by_frame.values())
            episode[camera] = dict(target_frames=used, valid_pixels=values,
                zero_frames=[t for t,n in by_frame.items() if n == 0],
                min_valid_pixels=min(values), mean_valid_fraction=float(np.mean(values)/pixels),
                source=str(path), source_size=before_stat[0], source_mtime_ns=before_stat[1])
            counts[ep,camera] = by_frame
        assert before == {str(p): sha(p) for p in (metadata, calibration)}
        files.update(before)
        rows[str(ep)] = episode
    summary = {}
    for split, group in windows.items():
        minima = [min(sum(counts[e,c][t+h] > 0 for c in cameras) for h in range(1,11)) for e,t in group]
        summary[split] = dict(windows=len(group), all_targets_all_three_views=sum(v == 3 for v in minima),
            all_targets_at_least_two_views=sum(v >= 2 for v in minima), min_nonempty_views=min(minima))
    result = dict(cameras=cameras, episodes=rows, summary=summary, small_source_file_hashes=files,
        annotation_sha256=manifest['annotation_sha256'], read_only=True, gpu_used=False,
        test_GT_read=False, interpretation='Counts only inherit depth!=0 validity. No physical-contact or GS-quality inference. Camera010 audit results must not be assigned to camera011.')
    destination = root / 'provenance/selected_camera_audit.json'
    with destination.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--depth-root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(audit(a.root,a.depth_root),indent=2))
