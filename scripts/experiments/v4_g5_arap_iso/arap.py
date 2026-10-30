"""Fixed-initial spatial-graph ARAP and log-length penalties on final GS XYZ.

The spatial graph is a material-topology *proxy*, not a correspondence to GT.
For each directed center/neighbour pair the short-time ARAP fit maps the
PREVIOUS predicted edge vector onto the CURRENT one.  Its proper rotation is
estimated without autograd; the residual still differentiates through BOTH
edge vectors.  No full-rank test drops collinear or planar centers.

ARAP normalizes each center by its FIXED INITIAL sum of squared edge lengths,
then averages over nonisolated centers and predicted times, then windows.
ISO uses the unique undirected edges and compares every predicted length with
the initial length.  Empty windows contribute differentiable zero and retain
their equal batch weight.  All graph weights are one.
"""
import torch
from torch.nn import functional as F

from .structure import StructureGraph, _validate_initial


ARAP_DENOM_EPSILON = 1e-12  # Squared normalized-world-coordinate length.
ISO_LENGTH_EPSILON = 1e-6  # Normalized-world-coordinate length, not a ratio.
ISO_SMOOTH_L1_BETA = 0.1


def _inputs(pred_xyz, initial_xyz, graph):
    if not isinstance(graph, StructureGraph):
        raise TypeError('Expected the frozen initial-GS StructureGraph')
    _validate_initial(initial_xyz, graph.gs_mask)
    if (pred_xyz.ndim != 4 or pred_xyz.shape[0] != len(initial_xyz)
            or pred_xyz.shape[1] < 1 or pred_xyz.shape[2:] != initial_xyz.shape[1:]):
        raise ValueError('Prediction must be final GS centers [B,H,N,3], matching initial GS')
    if (not pred_xyz.is_floating_point() or pred_xyz.device != initial_xyz.device
            or pred_xyz.device != graph.edges.device or pred_xyz.dtype != initial_xyz.dtype):
        raise ValueError('Initial GS, predicted GS and graph must share device and floating dtype')
    valid = graph.gs_mask[:, None, :, None]
    current = torch.where(valid, pred_xyz, torch.zeros_like(pred_xyz))
    initial = torch.where(graph.gs_mask[..., None], initial_xyz.detach(), torch.zeros_like(initial_xyz))
    if not torch.isfinite(current).all():
        raise FloatingPointError('Nonfinite valid predicted GS coordinate')
    if (graph.edges.ndim != 3 or graph.edges.shape[0] != len(initial_xyz)
            or graph.edges.shape[-1] != 2 or graph.edge_mask.shape != graph.edges.shape[:2]
            or graph.rest_lengths.shape != graph.edge_mask.shape or graph.edge_mask.dtype != torch.bool):
        raise ValueError('Malformed fixed initial structure graph')
    if graph.edges.numel() and ((graph.edges < 0).any() or (graph.edges >= initial_xyz.shape[1]).any()):
        raise ValueError('Graph endpoint outside initial GS identity range')
    edge = graph.edges
    batches = torch.arange(len(initial_xyz), device=initial_xyz.device)[:, None]
    legal = graph.gs_mask[batches, edge[..., 0]] & graph.gs_mask[batches, edge[..., 1]]
    if (graph.edge_mask & (~legal | (edge[..., 0] == edge[..., 1]))).any():
        raise ValueError('Valid graph edge connects Padding or Self')
    if not torch.isfinite(graph.rest_lengths[graph.edge_mask]).all() or (graph.rest_lengths[graph.edge_mask] <= 0).any():
        raise ValueError('Valid initial graph lengths must be positive and finite')
    return current, initial


@torch.no_grad()
def _proper_rotations(covariance):
    """Kabsch R from covariance=sum(current_edge previous_edge^T).

    SVD is detached, including the determinant correction.  Rank-deficient
    fits remain valid fits and are NEVER masked from the loss.  If a backend
    batched SVD raises, retry its small matrices separately; an individually
    unsolved/nonfinite fit uses identity and is explicitly counted.  This
    preserves that center's residual rather than silently suppressing it.
    """
    shape = covariance.shape[:-2]
    work = covariance.detach()
    # PyTorch SVD does not support half/bfloat16, while the formal path is FP32.
    if work.dtype not in (torch.float32, torch.float64):
        work = work.float()
    eye = torch.eye(3, dtype=work.dtype, device=work.device)
    batch_retry = torch.zeros(shape, dtype=torch.bool, device=work.device)
    fallback = torch.zeros(shape, dtype=torch.bool, device=work.device)
    try:
        u, singular, vh = torch.linalg.svd(work, full_matrices=False)
    except RuntimeError:
        flat = work.reshape(-1, 3, 3)
        matrices, spectra = [], []
        failed = []
        for matrix in flat:
            try:
                a, s, b = torch.linalg.svd(matrix, full_matrices=False)
                matrices.append((a, b)); spectra.append(s); failed.append(False)
            except RuntimeError:
                matrices.append((eye, eye)); spectra.append(matrix.new_zeros(3)); failed.append(True)
        u = torch.stack([p[0] for p in matrices]).reshape(*shape, 3, 3)
        vh = torch.stack([p[1] for p in matrices]).reshape(*shape, 3, 3)
        singular = torch.stack(spectra).reshape(*shape, 3)
        batch_retry = torch.ones_like(batch_retry)
        fallback = torch.tensor(failed, device=work.device, dtype=torch.bool).reshape(shape)
    determinant = torch.linalg.det(u @ vh)
    diagonal = torch.ones_like(singular)
    diagonal[..., -1] = torch.where(determinant < 0, -torch.ones_like(determinant), torch.ones_like(determinant))
    rotation = (u * diagonal[..., None, :]) @ vh
    finite = torch.isfinite(rotation).all(dim=(-2, -1)) & torch.isfinite(singular).all(-1)
    fallback = fallback | ~finite
    rotation = torch.where(fallback[..., None, None], eye, rotation)
    # Rank is diagnostic only.  It never changes the valid-center loss mask.
    tolerance = torch.finfo(work.dtype).eps * 3 * singular[..., :1]
    rank = (singular > tolerance).sum(-1)
    return rotation.to(covariance.dtype), dict(
        fallback=fallback, batch_retry=batch_retry, rank=rank,
        reflection_corrected=(determinant < 0), singular_values=singular)


def _compute(pred_xyz, initial_xyz, graph, *, include_rotations=False, capture_solver_failures=False):
    current, initial = _inputs(pred_xyz, initial_xyz, graph)
    b, h, n, _ = current.shape
    e = graph.edges.shape[1]
    mask = graph.edge_mask
    connected_zero = current.reshape(b, -1)[:, :1].sum(-1)*0.
    if not e or not n:
        zero_h = connected_zero[:, None].expand(b, h)
        zero_counts = mask.new_zeros((b, h), dtype=torch.long)
        stats = dict(arap_per_window=connected_zero.detach(), iso_per_window=connected_zero.detach(),
            arap_per_horizon=zero_h.detach(), iso_per_horizon=zero_h.detach(),
            edge_count=mask.sum(-1), center_count=mask.sum(-1),
            kabsch_fallback_count=zero_counts, kabsch_batch_retry_count=zero_counts,
            kabsch_rank0_count=zero_counts, kabsch_rank1_count=zero_counts,
            kabsch_rank2_count=zero_counts, kabsch_rank3_count=zero_counts,
            kabsch_reflection_correction_count=zero_counts)
        if include_rotations:
            stats['rotations'] = torch.eye(3, device=current.device, dtype=current.dtype).expand(b, h, n, 3, 3).clone()
        if capture_solver_failures:
            stats['solver_failures'] = dict(indices=graph.edges.new_empty((0, 3)),
                covariance=current.new_empty((0, 3, 3)), rotation=current.new_empty((0, 3, 3)),
                singular_values=current.new_empty((0, 3)))
        return connected_zero, connected_zero, stats
    # The fixed undirected graph supplies one unit-weight edge for each center
    # direction, while ISO below counts each undirected edge exactly once.
    source = torch.cat((graph.edges[..., 0], graph.edges[..., 1]), -1)
    target = torch.cat((graph.edges[..., 1], graph.edges[..., 0]), -1)
    directed_mask = torch.cat((mask, mask), -1)
    rest_squared = torch.cat((graph.rest_lengths.square(), graph.rest_lengths.square()), -1)
    rest_squared = torch.where(directed_mask, rest_squared, torch.zeros_like(rest_squared))
    denominator = current.new_zeros((b, n)).scatter_add(1, source, rest_squared)
    degree = torch.zeros((b, n), dtype=torch.long, device=current.device).scatter_add(1, source, directed_mask.long())
    center_mask = degree > 0
    center_count = center_mask.sum(-1)
    sequence = torch.cat((initial[:, None], current), 1)
    gather_source = source[:, None, :, None].expand(b, h+1, 2*e, 3)
    gather_target = target[:, None, :, None].expand_as(gather_source)
    vectors = sequence.gather(2, gather_target) - sequence.gather(2, gather_source)
    vectors = torch.where(directed_mask[:, None, :, None], vectors, torch.zeros_like(vectors))
    previous, following = vectors[:, :-1], vectors[:, 1:]
    with torch.no_grad():
        product = following[..., :, None] * previous[..., None, :]
        covariance = current.new_zeros((b, h, n, 9)).scatter_add(
            2, source[:, None, :, None].expand(b, h, 2*e, 9), product.flatten(-2))
        rotation, diagnostics = _proper_rotations(covariance.reshape(b, h, n, 3, 3))
    selected_rotation = rotation.gather(2, source[:, None, :, None, None].expand(b, h, 2*e, 3, 3))
    # Only R is detached.  The previous prediction here is intentionally live.
    residual = following - (selected_rotation @ previous[..., None]).squeeze(-1)
    squared = residual.square().sum(-1)
    residual_sum = current.new_zeros((b, h, n)).scatter_add(2, source[:, None].expand(b, h, 2*e), squared)
    local = residual_sum/(denominator[:, None] + ARAP_DENOM_EPSILON)
    local = torch.where(center_mask[:, None], local, torch.zeros_like(local))
    arap_h = local.sum(-1)/center_count.clamp_min(1)[:, None]
    arap = arap_h.mean(-1) + connected_zero
    lengths = torch.linalg.vector_norm(following[:, :, :e], dim=-1)
    log_ratio = torch.log((lengths+ISO_LENGTH_EPSILON)/(graph.rest_lengths[:, None]+ISO_LENGTH_EPSILON))
    iso_edge = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=ISO_SMOOTH_L1_BETA, reduction='none')
    iso_edge = torch.where(mask[:, None], iso_edge, torch.zeros_like(iso_edge))
    iso_h = iso_edge.sum(-1)/mask.sum(-1).clamp_min(1)[:, None]
    iso = iso_h.mean(-1) + connected_zero
    valid_centers = center_mask[:, None].expand(b, h, n)
    stats = dict(arap_per_window=arap.detach(), iso_per_window=iso.detach(),
        arap_per_horizon=arap_h.detach(), iso_per_horizon=iso_h.detach(),
        edge_count=mask.sum(-1), center_count=center_count,
        kabsch_fallback_count=(diagnostics['fallback'] & valid_centers).sum(-1),
        kabsch_batch_retry_count=(diagnostics['batch_retry'] & valid_centers).sum(-1),
        kabsch_reflection_correction_count=(diagnostics['reflection_corrected'] & valid_centers).sum(-1))
    for rank in range(4):
        stats[f'kabsch_rank{rank}_count'] = ((diagnostics['rank'] == rank) & valid_centers).sum(-1)
    if include_rotations:
        stats['rotations'] = rotation
    if capture_solver_failures:
        # Opt-in preparation/Smoke evidence only.  Ordinary training avoids
        # sparse-index synchronization and full matrix diagnostic transfers.
        failed = diagnostics['fallback'] & valid_centers
        stats['solver_failures'] = dict(indices=failed.nonzero(),
            covariance=covariance.reshape(b, h, n, 3, 3)[failed].detach(),
            rotation=rotation[failed], singular_values=diagnostics['singular_values'][failed])
    return arap, iso, stats


def combined_regularizers(pred_xyz, initial_xyz, graph, *, capture_solver_failures=False):
    """Return equal-window ARAP, ISO scalars and detached device diagnostics."""
    arap, iso, stats = _compute(pred_xyz, initial_xyz, graph, capture_solver_failures=capture_solver_failures)
    return arap.mean(), iso.mean(), stats


def arap_loss_per_window(pred_xyz, initial_xyz, graph):
    return _compute(pred_xyz, initial_xyz, graph)[0]


def iso_loss_per_window(pred_xyz, initial_xyz, graph):
    return _compute(pred_xyz, initial_xyz, graph)[1]


def arap_loss(pred_xyz, initial_xyz, graph):
    return arap_loss_per_window(pred_xyz, initial_xyz, graph).mean()


def iso_loss(pred_xyz, initial_xyz, graph):
    return iso_loss_per_window(pred_xyz, initial_xyz, graph).mean()


@torch.no_grad()
def local_rigid_rotations(pred_xyz, initial_xyz, graph):
    """Read-only Kabsch diagnostics; maps previous edges to current edges."""
    return _compute(pred_xyz, initial_xyz, graph, include_rotations=True)[2]


@torch.no_grad()
def solver_failure_diagnostics(pred_xyz, initial_xyz, graph):
    """Read-only retry diagnostic; opt-in capture on the original call is exact.

    The returned indices are [batch, zero-based predicted time, GS center].
    A retry may no longer reproduce a transient backend failure; calibration
    and Smoke therefore request ``capture_solver_failures=True`` on their
    original call to retain the actual failing covariance and fallback R.
    """
    return _compute(pred_xyz, initial_xyz, graph, capture_solver_failures=True)[2]['solver_failures']
