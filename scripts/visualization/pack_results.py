"""Package a fresh fixed-window evaluation for the offline demo (no model fitting)."""
import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image

TRAIN = (189, 225, 241)
GENERALIZATION = (181, 260)
MODELS = ('soma_e10', 'g5_e8')
CAMERAS = ('brics-odroid-023_cam0', 'brics-odroid-021_cam1', 'brics-odroid-009_cam1')
G5_SHA256 = '00fe8922bde7356e7290125813c2f0ca456604df03eafa2143158afd30073e57'
SOMA_SHA256 = 'f8b2387eb5e240f63f95bd213920947d697e3d1371ef926c8c28b6e71dce339b'
INITIAL_ATTRIBUTES = ('log_scales', 'quaternions', 'opacity_logits', 'sh_dc')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _modern_completion(record, windows):
    require(isinstance(record, dict), 'Invalid evaluation completion record')
    require(record.get('status') == 'completed_requires_visual_review',
            'Evaluation is not a completed inference awaiting visual review')
    require(type(record.get('optimizer_updates')) is int and record['optimizer_updates'] == 0,
            'Presentation requires inference with zero optimizer updates')
    require(record.get('windows') == windows and record.get('models') == 2,
            'Evaluation window/model completion counts differ')
    for key in ('normalizers_and_weights_unchanged', 'native_continuous_parity',
                'input_and_checkpoint_sha256_unchanged'):
        require(record.get(key) is True, f'Missing successful evaluation gate: {key}')


def _cache_path(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(),
            'Cache manifest paths must be relative to the evaluation directory')
    path = (root/name).resolve()
    require(path != root and root in path.parents, 'Cache manifest path escapes the evaluation directory')
    return path


def _checkpoint_digests(value):
    if isinstance(value, dict):
        return set().union(*(_checkpoint_digests(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_checkpoint_digests(v) for v in value)) if value else set()
    return {value} if isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) else set()


def presentation_inputs():
    required = {'protocol.json', 'metrics.csv'}
    for episode, start in ((0, 225), (0, 241), (1, 181), (1, 260)):
        required.update(f'comparison_assets/ep{episode}_start{start}_cam{camera}_h{h:02d}.webp'
                        for camera in range(3) for h in range(1, 16))
    required.update(('predictions/ep0_start225.npz', 'predictions/ep1_start260.npz'))
    return required


def validate_source(evaluation, verified_cache_manifest=None, required_inputs=()):
    """Accept a real completed inference or an explicit original-cache bridge.

    The cache route preserves the legacy record as written. In particular, it
    does not manufacture modern completion/status/normalizer fields for the
    historical resimulation run, whose original schema did not record them.
    """
    root = Path(evaluation).resolve()
    if verified_cache_manifest is None:
        path = root/'evaluation_complete.json'
        completion = _read_json(path)
        _modern_completion(completion, windows=5)
        return dict(mode='fresh-evaluation', fresh_inference=True, model_updates=0,
                    completion=dict(path=str(path), sha256=digest(path), record=completion),
                    source_files=[str(path)])

    manifest_path = Path(verified_cache_manifest).resolve()
    manifest = _read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest.get('schema') == 'touch4d-verified-cache-v1',
            'Unsupported verified-cache schema')
    files = manifest.get('files_sha256')
    require(isinstance(files, dict) and files, 'Verified cache must declare its complete input hash closure')
    proofs = manifest.get('source_completion_proofs')
    require(isinstance(proofs, list) and len(proofs) == 2 and all(isinstance(p, dict) for p in proofs),
            'Verified cache needs the two original completion/protocol proof pairs')
    require({p.get('role') for p in proofs} == {'resimulation', 'generalization'},
            'Verified cache proof roles must be resimulation and generalization')
    required = set(required_inputs) | {'protocol.json', 'metrics.csv'}
    for proof in proofs:
        require(isinstance(proof, dict), 'Invalid cache completion proof')
        require(isinstance(proof.get('completion'), str) and isinstance(proof.get('protocol'), str),
                'Original completion/protocol proof paths are required')
        required.update((proof.get('completion'), proof.get('protocol')))
    require(None not in required and required <= files.keys(),
            'Cache manifest omits a consumed input or original completion/protocol proof')
    for name, expected in files.items():
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
                'Cache manifest must use full lowercase SHA256 values')
        require(digest(_cache_path(root, name)) == expected, f'Verified cache input hash mismatch: {name}')

    accepted = []
    source_files = [str(manifest_path)]
    for proof in proofs:
        completion_path = _cache_path(root, proof['completion'])
        protocol_path = _cache_path(root, proof['protocol'])
        completion, protocol = _read_json(completion_path), _read_json(protocol_path)
        require(tuple(protocol.get('cameras', ())) == CAMERAS
                and protocol.get('horizon') == 15 and protocol.get('delta') == 1,
                'Historical proof is not the original H15/delta1/camera protocol')
        require({G5_SHA256, SOMA_SHA256} <= _checkpoint_digests(protocol.get('checkpoints', {})),
                'Historical proof does not identify the frozen current/E10 checkpoints')
        if proof['role'] == 'resimulation':
            require(proof.get('schema') == 'legacy-h15-v1', 'Wrong legacy resimulation proof schema')
            require(completion.get('status') not in ('failed', 'running', 'preflight'),
                    'Legacy proof reports an unsuccessful run')
            require(completion.get('windows') == {'train': 140, 'val': 24}
                    and completion.get('model_count') == 11,
                    'Unexpected original resimulation completion coverage')
            require(type(completion.get('optimizer_updates')) is int and completion['optimizer_updates'] == 0
                    and completion.get('native_continuous_parity') is True
                    and completion.get('input_and_checkpoint_sha256_unchanged') is True,
                    'Legacy resimulation native/input completion gates did not pass')
            require(protocol.get('train_starts') == list(range(113, 253))
                    and protocol.get('holdout_starts') == list(range(268, 292)),
                    'Historical resimulation window protocol differs')
        else:
            require(proof.get('schema') == 'cross-h15-v3', 'Wrong cross-episode proof schema')
            _modern_completion(completion, windows=6)
            cases = {(int(c['episode']), int(c['start'])) for c in protocol.get('cases', [])}
            require({(1, 181), (1, 260)} <= cases, 'Historical proof lacks the selected generalization windows')
        accepted.append(dict(role=proof['role'], schema=proof['schema'],
                             completion=dict(path=str(completion_path), sha256=files[proof['completion']], record=completion),
                             protocol=dict(path=str(protocol_path), sha256=files[proof['protocol']])))
        source_files.extend((str(completion_path), str(protocol_path)))
    return dict(mode='verified-cache', fresh_inference=False, model_updates=0,
                manifest=dict(path=str(manifest_path), sha256=digest(manifest_path)),
                cache_inputs_sha256=files, source_completion_proofs=accepted,
                source_files=source_files,
                scope='Repackaged original completed-run caches; no new model inference or invented legacy completion fields.')


def validate_prediction_initial(prediction, observed, start):
    frame = np.asarray(prediction['frame_id'])
    require(frame.ndim == 0 and np.issubdtype(frame.dtype, np.integer) and int(frame) == start,
            'Prediction initial frame_id mismatch')
    for key in ('xyz',) + INITIAL_ATTRIBUTES:
        a, b = prediction[key], observed[key]
        require(a.dtype == b.dtype and np.array_equal(a, b), f'Prediction initial raw attribute mismatch: {key}')
    require(prediction['g5_xyz'].shape == (15, *observed['xyz'].shape),
            'Expected the fixed 15-step prediction track with initial point identity')


def summarize(rows, episode, starts):
    summaries = []
    for model in MODELS:
        windows = []
        for start in starts:
            group = [r for r in rows if int(r['episode']) == episode
                     and int(r['start']) == start and r['model'] == model]
            if len(group) != 15 or sorted(int(r['h']) for r in group) != list(range(1, 16)):
                raise ValueError(f'Incomplete or duplicated H1..H15: ep{episode}/{start}/{model}')
            if any(int(r['frame']) != start + int(r['h']) for r in group):
                raise ValueError('Metric frame/horizon disagreement')
            last = next(r for r in group if int(r['h']) == 15)
            values = [np.mean([float(r['cd_mm']) for r in group]), float(last['cd_mm']),
                      np.mean([float(r['rgb_mae']) for r in group]),
                      np.mean([float(r['alpha_mae']) for r in group])]
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite evaluation metrics')
            windows.append(values)
        mean = np.mean(windows, axis=0)
        summaries.append(dict(model=model, mean_cd_mm=float(mean[0]), h15_cd_mm=float(mean[1]),
                              rgb_mae=float(mean[2]), alpha_mae=float(mean[3])))
    return summaries


def gaussian(gs, xyz=None):
    xyz = gs['xyz'] if xyz is None else xyz
    result = np.concatenate([xyz, np.exp(gs['log_scales']), gs['quaternions'],
                             1 / (1 + np.exp(-gs['opacity_logits'])),
                             np.clip(.5 + .28209479177387814 * gs['sh_dc'], 0, 1)], axis=1)
    if result.shape[1] != 14 or not np.isfinite(result).all():
        raise ValueError('Invalid Gaussian presentation attributes')
    return result


def splat(gs, center, scale):
    count = len(gs)
    values = np.zeros((((count + 127) // 128) * 128, 16), np.float32)
    values[:count, :3] = (gs[:, :3] - center) * scale
    values[:count, 3] = gs[:, 10]
    values[:count, 4:7] = np.log2(gs[:, 3:6] * scale)
    values[:count, 7] = gs[:, 11]
    values[:count, 8:12] = gs[:, 6:10]
    values[:count, 12:14] = gs[:, 12:14]
    half = values.astype('<f2')
    if not np.isfinite(half).all():
        raise ValueError('Nonfinite Gaussian texture')
    return dict(count=count, gzip=base64.b64encode(
        gzip.compress(half.tobytes(), mtime=0)).decode('ascii'))


def package(evaluation, dataset_root, output, overwrite=False, verified_cache_manifest=None):
    evaluation, dataset_root, output = map(Path, (evaluation, dataset_root, output))
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    acceptance = validate_source(evaluation, verified_cache_manifest, presentation_inputs())
    protocol_path = evaluation / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if tuple(protocol['cameras']) != CAMERAS or protocol.get('horizon', 15) != 15 or protocol.get('delta', 1) != 1:
        raise ValueError('Expected the fixed H15, delta1, three-camera evaluation protocol')
    with (evaluation / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    metrics = dict(resimulation=summarize(rows, 0, TRAIN),
                   generalization=summarize(rows, 1, GENERALIZATION))
    files = {}
    sources = {}

    def read(path):
        sources[str(path.resolve())] = digest(path)
        return path.read_bytes()

    read(protocol_path)
    read(evaluation / 'metrics.csv')
    for path in acceptance['source_files']:
        read(Path(path))
    cases = []
    for episode, start in ((0, 225), (0, 241), (1, 181), (1, 260)):
        group = 'Resimulation' if episode == 0 else 'Generalization'
        case = dict(key=f'ep{episode}_start{start}', episode=episode, start=start,
                    group=group, label=f'{group} · Ep{episode} / start {start}', views=[])
        for camera, camera_name in enumerate(CAMERAS):
            frames = []
            for h in range(1, 16):
                name = f'ep{episode}_start{start}_cam{camera}_h{h:02d}.webp'
                path = evaluation / 'comparison_assets' / name
                content = read(path)
                with Image.open(io.BytesIO(content)) as image:
                    width, height = image.size
                    rgb_height = (height - 54) // 2
                    if width % 3 or rgb_height <= 0:
                        raise ValueError(f'Invalid GT/SoMA/ours RGB+Alpha triptych: {name}')
                    if camera == 0 and h == 15 and (episode, start) != (0, 241):
                        buffer = io.BytesIO()
                        image.crop((0, 40, width, 40 + rgb_height)).convert('RGB').save(buffer, 'PNG')
                        files[f'assets/rgb/ep{episode}_start{start}_h15.png'] = buffer.getvalue()
                key = 'assets/rgb/' + name
                files[key] = content
                frames.append(dict(path=key, source=str(path.resolve()), width=width, height=height,
                                   rgb_height=rgb_height, top=40, horizon=h, frame=start+h))
            case['views'].append(dict(camera=camera_name, frames=frames))
        cases.append(case)
    rgb = dict(cameras=list(CAMERAS), cases=cases,
               display='Original three-column WebP; CSS clips the exact RGB row.')
    scenes = []
    for key, episode, start in (('train', 0, 225), ('generalization', 1, 260)):
        prediction_path = evaluation / f'predictions/ep{episode}_start{start}.npz'
        read(prediction_path)
        scene_path = dataset_root / '008-pink-cloth' / f'episode_{episode}' / 'gs'
        targets = []
        initial_raw = None
        for frame in range(start, start + 16):
            path = scene_path / f'frame_{frame:06d}.npz'
            read(path)
            with np.load(path, allow_pickle=False) as gs:
                if frame == start:
                    initial_raw = {name: gs[name].copy() for name in ('xyz',) + INITIAL_ATTRIBUTES}
                targets.append(gaussian(gs))
        with np.load(prediction_path, allow_pickle=False) as data:
            validate_prediction_initial(data, initial_raw, start)
            predicted = [gaussian(data, xyz) for xyz in data['g5_xyz']]
        # Preserve the existing fixed view recipe. This display transform is
        # shared by GT/prediction and never enters model states or metrics.
        reference = targets[1] if key == 'train' else targets[0]
        center = np.median(reference[:, :3], axis=0)
        radius = np.quantile(np.linalg.norm(reference[:, :3] - center, axis=1), .99)
        if radius <= 0:
            raise ValueError('Degenerate scene display bounds')
        scale = .60 / float(radius)
        assets, pairs = [], []
        for gt, pred in zip(targets[1:], predicted):
            pairs.append([len(assets), len(assets)+1])
            assets.extend([splat(gt, center, scale), splat(pred, center, scale)])
        data = dict(assets=assets, frames=pairs, frame_ids=list(range(start+1, start+16)), pivot=[0, 0, 0])
        files[f'assets/3d/{key}.js'] = ('window.DEMO_SCENES=window.DEMO_SCENES||{};window.DEMO_SCENES['
                                      + json.dumps(key) + ']=' + json.dumps(data, separators=(',', ':')) + ';').encode()
        scenes.append(dict(key=key, episode=episode, start=start, display_center_m=center.tolist(), display_scale=scale))
    provenance = dict(schema='touch4d-current-demo-v1', checkpoints=protocol.get('checkpoints', {}),
                      training_windows=[dict(episode=0, start=s) for s in TRAIN],
                      generalization_windows=[dict(episode=1, start=s) for s in GENERALIZATION],
                      aggregation='Per-window H1..15 mean, then equal-weight windows; H15 CD final step only.',
                      metrics=metrics, rgb_sequences=rgb, scenes=scenes, inputs_sha256=sources,
                      protocol=protocol, model_updates=0, fresh_inference=acceptance['fresh_inference'],
                      source_acceptance=acceptance)
    files['provenance.json'] = json.dumps(provenance, ensure_ascii=False, indent=2).encode('utf-8')
    files['metrics.js'] = ('window.DEMO_METRICS=' + json.dumps(metrics) + ';').encode()
    files['rgb_sequences.js'] = ('window.DEMO_RGB=' + json.dumps(rgb, separators=(',', ':')) + ';').encode()
    manifest = dict(schema='touch4d-demo-bundle-v1', files_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    return dict(output=str(output.resolve()), files=len(files), sha256=digest(output), metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verified-cache-manifest', type=Path,
                        help='Explicitly bridge the two original completed inference sources; never fabricate a fresh completion')
    args = parser.parse_args()
    print(json.dumps(package(args.evaluation, args.dataset_root, args.output, args.overwrite,
                             args.verified_cache_manifest), indent=2))


if __name__ == '__main__':
    main()
