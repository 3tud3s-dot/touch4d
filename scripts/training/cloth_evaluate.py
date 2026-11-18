"""Frozen H15 cloth/SoMA comparison, independent of historical experiment paths.

Run from the project root with an explicit assets root and configuration. This
command never trains, refits normalization, refreshes observed GT, or overwrites
an existing output directory. Real rendering requires the audited CUDA runtime.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from scripts.dataloader.collate import collate_fn
from scripts.dataloader.dataset import Touch4DDataset, read_gs
from scripts.training import cloth_data
from scripts.training.cloth_protocol import fresh_model
from scripts.training.rendering import GaussianRenderer, assemble_gaussians
from scripts.training.train import to_device
from scripts.training.soma_adapter import SomaAdapter, require


G5_SHA256 = '00fe8922bde7356e7290125813c2f0ca456604df03eafa2143158afd30073e57'
SOMA_SHA256 = 'f8b2387eb5e240f63f95bd213920947d697e3d1371ef926c8c28b6e71dce339b'
SOMA_CONFIG_SHA256 = 'c4042fa409bdb4bc2746629a390f9763aa7f277a8ec54d2e43e314b30566e16a'
CASES = ((0, 189), (0, 225), (0, 241), (1, 181), (1, 260))
HORIZON = 15
PATH_FIELDS = ('data_root', 'depth_root', 'g5_checkpoint', 'soma_checkpoint',
               'g5_source_manifest', 'soma_source', 'soma_config', 'soma_package', 'soma_geometry')
COLUMNS = ('episode', 'start', 'frame', 'h', 'model', 'cd_mm', 'rgb_mae',
           'alpha_mae', 'psnr_db', 'target_copy_cd_mm', 'covariance_nonpositive')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def resolve_configuration(config, assets_root, project_root=None):
    config = dict(config)
    assets_root = Path(assets_root).resolve()
    project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    for key in PATH_FIELDS + ('gsplat_dir', 'gsplat_binary_dir'):
        value = config.get(key)
        if value is None:
            require(key not in PATH_FIELDS, f'Missing required path {key}')
            config[key] = None
            continue
        path = Path(value)
        base = project_root if key in ('soma_source', 'soma_config') else assets_root
        config[key] = (base / path).resolve() if not path.is_absolute() else path.resolve()
        require(config[key].exists(), f'Missing {key}: {config[key]}')
    cases = tuple((int(c['episode']), int(c['start'])) for c in config['windows'])
    require(cases == CASES, 'The delivered demo freezes the five specified windows and order')
    require(config.get('horizon') == 15 and config.get('delta') == 1,
            'The delivered protocol requires H15 and one-frame spacing')
    return config


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def dataset(config, episode, start):
    result = Touch4DDataset(
        cloth_data.SCENE, 'train', config['root'], horizon=HORIZON, prediction_interval=1,
        with_depth=True, depth_root=config['depth_root'], depth_cameras=cloth_data.CAMERAS,
        episode_frame_range=(episode, start, start+16))
    require(result.windows == ((episode, start),) and result.episodes[episode].slots == (None, 0),
            'Window identity or reviewed right-hand mapping mismatch')
    return result


def prepare(raw, processor, model, episode, start):
    """Original generic adapter: fixed train statistics and t..t+14 conditions."""
    ids = raw['meta']['frame_ids']
    times = raw['meta']['observation_timestamp_us']
    require(ids.tolist() == [list(range(start, start+16))]
            and raw['meta']['episode_id'].tolist() == [episode], 'Window/frame identity mismatch')
    require((times[:, 1:] > times[:, :-1]).all().item(), 'Nonmonotonic observation timestamps')
    hands = torch.tensor([False, True], device=ids.device).expand(1, 2)
    require(torch.equal(raw['inputs']['action_valid'], hands[:, None].expand(-1, 15, -1)),
            'Invalid reviewed right-hand window')
    require(model.config['horizon'] == 15 and model.config['prediction_interval'] == 1,
            'Model time protocol mismatch')
    batch = processor(raw)
    batch['inputs']['operation_hands'] = hands.clone()
    batch['inputs']['condition_frame_ids'] = ids[:, :-1].clone()
    batch['meta']['prediction_interval'] = 1
    batch['meta']['operation_hand_policy'] = cloth_data.HAND_POLICY
    return batch


def validate_cadence(timestamps, expected_dt=1/30):
    intervals = np.diff(np.asarray(timestamps))/1e6
    require(intervals.shape == (15,) and np.isfinite(intervals).all()
            and np.all(intervals > 0), 'Expected 16 consecutive finite increasing timestamps')
    require(np.max(np.abs(intervals-expected_dt)) < .005,
            'Cadence differs from the original observed 30 Hz data')
    return intervals


def tensors(gs, device='cuda'):
    return {key: torch.from_numpy(gs[key]).to(device) for key in
            ('xyz', 'log_scales', 'quaternions', 'opacity_logits', 'sh_dc')}


def radiance(gs):
    return dict(means=gs['xyz'].contiguous(), scales=gs['log_scales'].exp().contiguous(),
                quats=gs['quaternions'].contiguous(),
                opacities=gs['opacity_logits'][:, 0].sigmoid().contiguous(),
                colors=(gs['sh_dc']*.28209479177387814+.5).clamp(0, 1).contiguous())


def nearest_distances(pred, target, knn_points=None):
    if knn_points is None:
        # CPU diagnostics only. GPU scoring uses the original pytorch3d routine.
        distances = torch.cdist(pred[None], target[None])[0]
        return distances.min(1).values, distances.min(0).values
    a = knn_points(pred[None], target[None], K=1, return_nn=False).dists[0, :, 0].clamp_min(0).sqrt()
    b = knn_points(target[None], pred[None], K=1, return_nn=False).dists[0, :, 0].clamp_min(0).sqrt()
    return a, b


def chamfer_mm(pred, target, knn_points=None):
    require(pred.ndim == target.ndim == 2 and pred.shape[1] == target.shape[1] == 3
            and len(pred) > 0 and len(target) > 0, 'Expected nonempty independent XYZ point sets')
    a, b = nearest_distances(pred, target, knn_points)
    return 500*(a.mean()+b.mean())


def image_errors(rgb, alpha, target_rgb, target_alpha, mask):
    diff = rgb-target_rgb
    pixels = mask.sum((-1, -2))
    used = pixels > 0
    require(used.any().item(), 'Missing original target-depth-valid pixels')
    def aggregate(error):
        return ((error*mask).sum((-1, -2))[used]/pixels[used]).mean()
    return torch.stack((aggregate(diff.abs().mean(-1)), aggregate((alpha-target_alpha).abs()),
                        (-10*((diff.square().mean(-1)*mask).sum((-1, -2))/pixels.clamp_min(1))
                         .clamp_min(1e-12).log10())[used].mean()))


def render(gs, intrinsics, view_matrices, rasterization):
    rgbd, alpha, _ = rasterization(
        **gs, Ks=intrinsics, viewmats=view_matrices, width=1280, height=720,
        backgrounds=gs['means'].new_zeros((len(intrinsics), 3)), **GaussianRenderer.settings)
    require(torch.isfinite(rgbd).all().item() and torch.isfinite(alpha).all().item(),
            'Nonfinite common render')
    return rgbd[..., :3].contiguous(), alpha[..., 0].contiguous()


def byte_image(value):
    return value.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()


def roi_for(frames):
    visible = np.stack([alpha > 6 for _, alpha in frames]).any(0)
    ys, xs = visible.nonzero()
    require(len(ys) > 0, 'Reference foreground is empty')
    return max(0, xs.min()-40), max(0, ys.min()-40), min(1280, xs.max()+41), min(720, ys.max()+41)


def save_visual(root, episode, start, camera_names, visual):
    """Identical RGB/Alpha layout and GT-only ROI to the frozen source exporter."""
    from PIL import Image, ImageDraw
    for camera, camera_name in enumerate(camera_names):
        target_frames = [(rgb[camera], alpha[camera]) for rgb, alpha in visual['reference']]
        x0, y0, x1, y1 = roi_for(target_frames)
        width, height = x1-x0, y1-y0
        for k in range(HORIZON):
            canvas = Image.new('RGB', (3*width, 2*height+54), '#141820')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 4), f'Ep{episode} Start {start} H{k+1:02d} Frame {start+k+1} {camera_name}', fill='white')
            for i, (name, title) in enumerate((('reference', 'GT (V1 GS)'), ('soma_e10', 'SoMA Stage 2 E10'), ('g5_e8', 'G5-Direct E8'))):
                rgb, alpha = visual[name][k]
                draw.text((i*width+8, 24), title, fill='white')
                canvas.paste(Image.fromarray(rgb[camera, y0:y1, x0:x1]), (i*width, 40))
                canvas.paste(Image.fromarray(alpha[camera, y0:y1, x0:x1]).convert('RGB'), (i*width, height+54))
            draw.text((8, height+40), 'Alpha', fill='white')
            canvas.save(root/'comparison_assets'/f'ep{episode}_start{start}_cam{camera}_h{k+1:02d}.webp', quality=87)


def validate_completed_state(state, completion, actual_sha, source_sha, model_config,
                             processor_state, expected_config):
    """An explicit new-run route; it never relaxes the frozen-demo identity gate."""
    require(state.get('version') == 'touch4d-g5-direct-008-episode0-v1', 'Unsupported training checkpoint version')
    identity = state.get('identity', {})
    require(completion.get('status') == 'completed' and completion.get('mode') == identity.get('mode') == 'formal',
            'Only completed formal runs can be evaluated in completed-run mode')
    require(completion.get('checkpoint_sha256') == actual_sha, 'Completed checkpoint SHA mismatch')
    require(completion.get('source_bundle_sha256') == identity.get('source_bundle_sha256') == source_sha,
            'Completed checkpoint source bundle mismatch')
    require(identity.get('model') == model_config and identity.get('processor') == processor_state,
            'Completed checkpoint model or normalization mismatch')
    actual_config = identity.get('config', {})
    # Filesystem roots may relocate; every mathematical/protocol field is fixed.
    for key, value in expected_config.items():
        if key not in ('root', 'depth_root', 'source_sha'):
            require(actual_config.get(key) == value, f'Completed training configuration mismatch: {key}')
    require(identity.get('base_sha') == cloth_data.BASE_SHA, 'Completed training base identity mismatch')
    orders, history = identity.get('epoch_orders', []), state.get('history', [])
    expected_starts = list(range(113, 253))
    require(len(orders) == len(history) == 8, 'Completed run must contain eight full epochs')
    for order, row in zip(orders, history):
        require(sorted(order) == expected_starts and row.get('starts') == order
                and row.get('successful_updates') == 140, 'Completed update/window ledger mismatch')
    trained = [start for row in history for start in row['starts']]
    require(completion.get('updates') == 1120 and completion.get('trained_starts') == trained,
            'Completed run update totals mismatch')
    require(identity.get('held_out_starts') == completion.get('evaluated_starts') == list(range(268, 292)),
            'Completed run held-out coverage mismatch')


def load_current_model(config, model_mode='frozen-demo', completion=None):
    actual_sha = sha256(config['g5_checkpoint'])
    if model_mode == 'frozen-demo':
        require(completion is None, '--completion is only valid with explicit completed-run mode')
        require(actual_sha == G5_SHA256, 'Current model checkpoint hash mismatch')
    else:
        require(model_mode == 'completed-run' and completion is not None,
                'Completed-run mode requires an explicit completion.json')
    require(sha256(config['soma_checkpoint']) == SOMA_SHA256, 'SoMA E10 checkpoint hash mismatch')
    require(sha256(config['soma_config']) == SOMA_CONFIG_SHA256, 'Frozen SoMA configuration hash mismatch')
    state = torch.load(config['g5_checkpoint'], map_location='cpu', weights_only=False)
    model, processor = fresh_model('direct'), cloth_data.Cloth008Processor(cloth_data.SCENE)
    if model_mode == 'completed-run':
        completed = load_json(completion)
        expected_config = cloth_data.configuration(config['data_root'], config['depth_root'], cloth_data.BASE_SHA)
        validate_completed_state(state, completed, actual_sha, sha256(config['g5_source_manifest']),
                                 model.config, processor.state_dict(), expected_config)
    require(state['identity']['processor'] == processor.state_dict()
            and state['identity']['model'] == model.config, 'Checkpoint normalization/model identity mismatch')
    require(state['identity']['source_bundle_sha256'] == sha256(config['g5_source_manifest']),
            'Original source bundle identity mismatch')
    require(len(state['history']) == 8
            and all(row['successful_updates'] == 140 for row in state['history']),
            'Checkpoint is not the original single-Episode 8x140 update run')
    episode = config['data_root']/cloth_data.SCENE/'episode_0'
    depth = config['depth_root']/cloth_data.SCENE/'episode_0'
    actual_data = {key: sha256(episode/key) for key in
                   ('metadata.json', 'controller.npz', 'timestamps.npz', 'tactile.npz')}
    actual_data['depth_metadata.json'] = sha256(depth/'metadata.json')
    require(state['identity'].get('data_sha256') == actual_data,
            'Training data fingerprints differ from checkpoint identity')
    from scripts.training.cloth_checkpoint import finite
    finite(state['model'])
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    data_config = dict(state['identity']['config'], root=str(config['data_root']), depth_root=str(config['depth_root']))
    return model, processor, data_config


def preflight(config, model, processor, data_config):
    checks, hashes = [], {str(config[k]): sha256(config[k]) for k in ('g5_checkpoint', 'soma_checkpoint', 'g5_source_manifest', 'soma_config')}
    # The generic Ep1 adapter must reduce exactly to the original Ep0 adapter.
    canonical_dataset = dataset(data_config, 0, 113)
    raw = collate_fn([canonical_dataset[0]])
    require(equal(cloth_data.prepare_batch(raw, processor, model), prepare(raw, processor, model, 0, 113)),
            'Generic adapter differs from the original Ep0 input protocol')
    for episode, start in CASES:
        ds = dataset(data_config, episode, start)
        info = ds.episodes[episode]
        raw = collate_fn([ds[0]])
        prepare(raw, processor, model, episode, start)
        controls = np.load(info.path/'controller.npz')
        cp = controls['controller_points'][start:start+16, 1]
        require(controls['controller_valid_mask'][start:start+16, 1].all()
                and np.isfinite(cp).all(), 'Invalid native SoMA control trajectory')
        np.testing.assert_array_equal(raw['inputs']['controller_points'][0, :, 1].numpy(), cp[:-1])
        require(raw['targets']['target_depth_valid'].flatten(-2).any(-1).all().item(), 'Empty camera mask')
        dt = validate_cadence(info.timestamps[start:start+16])
        metadata = load_json(info.path/'metadata.json')
        keys = ['metadata.json', 'controller.npz', 'tactile.npz', 'timestamps.npz']
        keys += [f'gs/frame_{f:06d}.npz' for f in range(start, start+16)]
        for key in keys:
            path = info.path/key
            digest = sha256(path)
            if key != 'metadata.json':
                require(metadata['output_files'][key] == digest, f'V1 data identity mismatch: {path}')
            hashes[str(path)] = digest
        for key in ('metadata.json', 'camera_parameters.npz'):
            path = ds.depth_episodes[episode].path/key
            hashes[str(path)] = sha256(path)
        checks.append(dict(episode=episode, start=start, end=start+15, points=len(read_gs(info, start)['xyz']),
                           dt_seconds=dt.tolist(), valid_tactile_values=int(raw['inputs']['valid_mask'].sum()),
                           depth_shape=list(raw['targets']['target_depth_valid'].shape)))
    return canonical_dataset, checks, hashes


def code_sources(project_root):
    """Record the actual copied source bytes, including an uncommitted review tree."""
    project_root = Path(project_root)
    folders = ('scripts/training', 'scripts/dataloader', 'scripts/data_process',
               'third_party/soma/mmgs', 'third_party/soma/configs',
               'third_party/soma/gaussian-splatting')
    return {str(path.resolve()): sha256(path) for folder in folders
            for path in sorted((project_root/folder).rglob('*.py')) if path.is_file()}


def cuda_dependencies(config):
    require(os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1,
            'Full evaluation requires exactly one scheduled GPU')
    require(torch.__version__ == '2.7.1+cu128', 'Original renderer ABI requires torch 2.7.1+cu128')
    if config.get('gsplat_dir'):
        sys.path.append(str(config['gsplat_dir'].parent))
    from gsplat import rasterization
    if config.get('gsplat_binary_dir'):
        from torch.utils.cpp_extension import _import_module_from_library
        sys.modules['gsplat.csrc'] = _import_module_from_library('gsplat_cuda', str(config['gsplat_binary_dir']), True)
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci
    from pytorch3d.ops import knn_points
    return rasterization, quat_scale_to_covar_preci, knn_points


@torch.no_grad()
def run(args):
    raw_config = load_json(args.config)
    for key in PATH_FIELDS + ('gsplat_dir', 'gsplat_binary_dir'):
        override = getattr(args, key, None)
        if override is not None:
            raw_config[key] = str(override)
    config = resolve_configuration(raw_config, args.assets_root)
    torch.set_num_threads(4)
    torch.manual_seed(5)
    np.random.seed(5)
    model, processor, data_config = load_current_model(config, args.model_mode, args.completion)
    canonical_dataset, checks, hashes = preflight(config, model, processor, data_config)
    hashes.update(code_sources(Path(__file__).resolve().parents[2]))
    hashes[str(Path(args.config).resolve())] = sha256(args.config)
    if args.completion:
        hashes[str(Path(args.completion).resolve())] = sha256(args.completion)
    output = Path(args.output).resolve()
    require(not output.exists(), 'Output already exists; historical results cannot be overwritten')
    if args.preflight:
        output.mkdir(parents=True)
        write_json(output/'preflight.json', dict(status='PASS', cases=checks, source_sha256=hashes,
                   normalization=processor.state_dict(), refit_statistics=False, optimizer_updates=0,
                   original_ep0_batch_exact=True, gpu_evaluation_run=False, model_mode=args.model_mode,
                   completion_origin=str(args.completion) if args.completion else None))
        return
    rasterization, covariance, knn = cuda_dependencies(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = canonical_dataset.episodes[0]
    canonical = tensors(read_gs(info, 113))
    cp = np.load(info.path/'controller.npz')['controller_points'][113:307, 1]
    baseline = SomaAdapter(source=config['soma_source'], config=config['soma_config'], package=config['soma_package'],
                           geometry=config['soma_geometry'], checkpoint=config['soma_checkpoint'],
                           canonical=canonical, canonical_controller=cp, quat_scale_to_covar_preci=covariance)
    for path in baseline.dependencies:
        hashes[str(path)] = sha256(path)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.cuda()
    output.mkdir(parents=True)
    for folder in ('comparison_assets', 'predictions'):
        (output/folder).mkdir()
    write_json(output/'protocol.json', dict(
        version='cloth-demo-core-h15-v1', job=os.environ['SLURM_JOB_ID'], step=os.environ.get('SLURM_STEP_ID'),
        device=torch.cuda.get_device_name(), uuid=str(torch.cuda.get_device_properties(0).uuid),
        cases=checks, cameras=list(cloth_data.CAMERAS), horizon=15, delta=1,
        model_mode=args.model_mode, completion_origin=str(args.completion) if args.completion else None,
        source_sha256=hashes, checkpoints={str(config['g5_checkpoint']): sha256(config['g5_checkpoint']), str(config['soma_checkpoint']): SOMA_SHA256},
        normalization=processor.state_dict(), refit_statistics=False,
        g5_condition='Original frames t..t+14 action/tactile, fixed training normalization',
        soma_condition='World controller frames t..t+15; native destination kinematic control; no tactile',
        soma_template='Native H10 refresh from own prediction; previous=current at boundary',
        soma_runtime_overrides=baseline.runtime_overrides,
        time_caveat='Actual observations ~1/30s; original SoMA real_dt=1/15s and gravity scale4 retained',
        gravity=baseline.gravity.tolist(), volume_scalar=baseline.volume,
        truth='Independent V1 GS, common world metres, SH0; not real-camera photometric ground truth',
        metric='Mean bidirectional unsquared NN CD mm; original target-depth mask RGB/Alpha',
        aggregation='Mean H1..15 per window, then equal windows; H15 metric is final step only',
        limitations=['Selected windows, not full-distribution evaluation',
                     'Models differ in available conditioning and training budget']))
    begin = time.monotonic()
    with (output/'metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for episode, start in CASES:
            ds = dataset(data_config, episode, start)
            info = ds.episodes[episode]
            raw = collate_fn([ds[0]])
            batch = prepare(to_device(raw, 'cuda'), processor, model, episode, start)
            prediction = model(batch['inputs'])
            initial_data = read_gs(info, start)
            initial = tensors(initial_data)
            appearance = radiance(initial)
            cp = np.load(info.path/'controller.npz')['controller_points'][start:start+16, 1]
            actions = torch.as_tensor(cp, device='cuda', dtype=torch.float32)
            xyz, cov, matrices = baseline.predict(initial, actions)
            nonpositive = (torch.linalg.eigvalsh(matrices.double())[..., 0] <= 0).sum(-1)
            target = batch['targets']
            intrinsics = target['camera_intrinsics'][0]
            views = torch.linalg.inv(target['camera_to_world'][0])
            masks = target['target_depth_valid'][0]
            visual = {key: [] for key in ('reference', 'soma_e10', 'g5_e8')}
            g5_xyz = []
            for k in range(HORIZON):
                truth = radiance(tensors(read_gs(info, start+k+1)))
                tr, ta = render(truth, intrinsics, views, rasterization)
                visual['reference'].append((byte_image(tr), byte_image(ta)))
                copy = float(chamfer_mm(initial['xyz'], truth['means'], knn))
                for name in ('g5_e8', 'soma_e10'):
                    if name == 'g5_e8':
                        gs = assemble_gaussians(prediction, batch, processor, 0, k)
                        g5_xyz.append(gs['means'].cpu().numpy())
                    else:
                        gs = dict(appearance, means=xyz[k].contiguous(), covars=matrices[k].contiguous(), quats=None, scales=None)
                    rgb, alpha = render(gs, intrinsics, views, rasterization)
                    scores = image_errors(rgb, alpha, tr, ta, masks[k]).tolist()
                    cd = float(chamfer_mm(gs['means'], truth['means'], knn))
                    require(np.isfinite([cd, *scores]).all(), 'Nonfinite metric; rollout cannot be omitted')
                    writer.writerow(dict(zip(COLUMNS, (episode, start, start+k+1, k+1, name, cd, *scores, copy,
                                                       int(nonpositive[k]) if name == 'soma_e10' else 0))))
                    visual[name].append((byte_image(rgb), byte_image(alpha)))
            stream.flush()
            save_visual(output, episode, start, cloth_data.CAMERAS, visual)
            np.savez_compressed(output/'predictions'/f'ep{episode}_start{start}.npz',
                                g5_xyz=np.stack(g5_xyz), soma_xyz=xyz.cpu().numpy(),
                                soma_covariance=cov.cpu().numpy(), **initial_data)
            print(f'COMPLETE Ep{episode} start{start}: both models H15, native parity, 45 images', flush=True)
            del raw, batch, prediction, target, masks, visual, initial, appearance, xyz, cov, matrices, g5_xyz
    baseline.verify_unchanged()
    require(equal(before, {k: v.detach().cpu() for k, v in model.state_dict().items()}),
            'Current model state changed during inference')
    require(all(sha256(path) == digest for path, digest in hashes.items()), 'Input/checkpoint changed during evaluation')
    write_json(output/'evaluation_complete.json', dict(status='completed_requires_visual_review', windows=len(CASES), models=2,
               optimizer_updates=0, normalizers_and_weights_unchanged=True, native_continuous_parity=True,
               input_and_checkpoint_sha256_unchanged=True, seconds=time.monotonic()-begin,
               gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/evaluation.json'))
    parser.add_argument('--assets-root', '--runtime-root', dest='assets_root', type=Path, required=True)
    for key in PATH_FIELDS + ('gsplat_dir', 'gsplat_binary_dir'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            help='Override the corresponding path in the configuration')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-mode', choices=('frozen-demo', 'completed-run'), default='frozen-demo')
    parser.add_argument('--completion', type=Path, help='Required, explicitly selected cloth_train completion.json for completed-run mode')
    parser.add_argument('--preflight', action='store_true', help='CPU input/checkpoint checks only; no render or GPU inference')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
