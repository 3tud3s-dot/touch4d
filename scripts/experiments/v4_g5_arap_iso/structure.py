"""Weak fixed-initial GS edge-length supervision; no material correspondence.

The candidate graph is the ORIGINAL V4 ``initial_knn(..., k=16)`` graph,
including its backend's deterministic tie choices.  We never add/reorder a GS
identity to bridge holes, infer rope order, or consult a future point set.
Retained undirected edges are stored in canonical (lowest index first) order.

``s_i`` uses the ordinary midpoint median for an even number of valid nonzero
candidate lengths, rather than ``torch.median``'s lower-middle convention.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from scripts.training.model import initial_knn


EPSILON = 1e-6
RATIO_MIN = 0.8
RATIO_MAX = 1.2
SMOOTH_L1_BETA = 0.1


@dataclass(frozen=True)
class StructureGraph:
    edges: torch.Tensor             # [B,E,2], initial GS indices (not Anchors)
    edge_mask: torch.Tensor         # [B,E]
    rest_lengths: torch.Tensor      # [B,E], padded entries are one
    gs_mask: torch.Tensor           # [B,N]
    local_spacing: torch.Tensor     # [B,N], zero if no valid candidate exists
    stats: tuple                    # per-window Python records for provenance


def _validate_initial(initial_xyz, gs_mask):
    if initial_xyz.ndim != 3 or initial_xyz.shape[-1] != 3 or gs_mask.shape != initial_xyz.shape[:2]:
        raise ValueError('Expected initial GS XYZ [B,N,3] and mask [B,N]')
    if not initial_xyz.is_floating_point() or gs_mask.dtype != torch.bool or len(initial_xyz) == 0:
        raise ValueError('Nonempty floating batch and Boolean GS mask required')
    if initial_xyz.device != gs_mask.device:
        raise ValueError('GS XYZ and mask must share a device')
    if not torch.isfinite(initial_xyz[gs_mask]).all():
        raise ValueError('Nonfinite valid initial GS coordinate')


@torch.no_grad()
def build_structure_graph(initial_xyz, gs_mask):
    """Build once per initial window, using only initial valid GS centers.

    Self and lengths <=1e-6 are excluded BEFORE computing each point's local
    spacing.  Then keep mutual, unique edges satisfying the frozen open lower
    and closed upper length bounds.  All returned tensors are detached copies.
    No learned condition, predicted position or GT participates in selection.
    """
    _validate_initial(initial_xyz, gs_mask)
    xyz = torch.where(gs_mask[..., None], initial_xyz.detach(), torch.zeros_like(initial_xyz))
    keep = gs_mask.detach().clone()
    b, n, _ = xyz.shape
    neighbor_ids, neighbor_mask = initial_knn(xyz, keep, 16)
    k = neighbor_ids.shape[-1]
    spacing = xyz.new_zeros((b, n))
    edge_rows, length_rows, records = [], [], []
    for batch in range(b):
        src = torch.arange(n, device=xyz.device)[:, None].expand(n, k)
        dst = neighbor_ids[batch]
        valid = neighbor_mask[batch] & keep[batch, :, None] & keep[batch, dst] & (src != dst)
        lengths = torch.linalg.vector_norm(xyz[batch, dst] - xyz[batch, :, None], dim=-1)
        candidate_count = int(valid.sum())
        nonzero = valid & (lengths > EPSILON)
        if k:
            sorted_lengths = lengths.masked_fill(~nonzero, float('inf')).sort(-1).values
            counts = nonzero.sum(-1)
            lo = ((counts-1).clamp_min(0)//2)[:, None]
            hi = (counts//2)[:, None]
            midpoint = (sorted_lengths.gather(-1, lo) + sorted_lengths.gather(-1, hi)).squeeze(-1)*0.5
            spacing[batch] = torch.where(counts > 0, midpoint, torch.zeros_like(midpoint))
        # Membership via sorted integer keys is O(N*K), without an N-by-N mask.
        source, target = src[nonzero], dst[nonzero]
        keys = source*n + target
        reverse = target*n + source
        sorted_keys = keys.sort().values
        if len(keys):
            positions = torch.searchsorted(sorted_keys, reverse)
            mutual = (positions < len(sorted_keys)) & (sorted_keys[positions.clamp_max(len(sorted_keys)-1)] == reverse)
        else:
            mutual = torch.zeros_like(keys, dtype=torch.bool)
        unique = mutual & (source < target)
        source, target = source[unique], target[unique]
        rest = torch.linalg.vector_norm(xyz[batch, source] - xyz[batch, target], dim=-1)
        local = torch.minimum(spacing[batch, source], spacing[batch, target])
        lower = torch.maximum(local*0.1, local.new_full(local.shape, EPSILON))
        retained = (rest > lower) & (rest <= 2*local)
        edges = torch.stack((source[retained], target[retained]), -1)
        rest = rest[retained]
        if len(edges):
            order = (edges[:, 0]*n + edges[:, 1]).argsort()
            edges, rest = edges[order], rest[order]
        edge_rows.append(edges); length_rows.append(rest)
        mutual_count = int(unique.sum())
        count = len(edges)
        records.append(dict(valid_gs=int(keep[batch].sum()), candidate_directed_edges=candidate_count,
            near_zero_removed_directed=candidate_count-int(nonzero.sum()),
            nonzero_candidate_directed_edges=int(nonzero.sum()),
            nonmutual_removed_directed=int((~mutual).sum()),
            mutual_undirected_before_length=mutual_count,
            length_filtered_undirected=mutual_count-count, retained_edges=count,
            retained_fraction_of_mutual=count/mutual_count if mutual_count else 0.,
            filtered_fraction_of_mutual=(mutual_count-count)/mutual_count if mutual_count else 0.,
            rest_length_min=float(rest.min()) if count else None,
            rest_length_median=float(torch.quantile(rest, .5)) if count else None,
            rest_length_max=float(rest.max()) if count else None,
            epsilon=EPSILON, candidate_knn=16, spacing_median='midpoint_for_even_count',
            candidate_tie_policy='original_initial_knn_backend_topk',
            graph_reference='initial_valid_GS_only', topology='spatial_neighbor_proxy_not_material'))
    slots = max((len(row) for row in edge_rows), default=0)
    edges = torch.zeros((b, slots, 2), dtype=torch.long, device=xyz.device)
    edge_mask = torch.zeros((b, slots), dtype=torch.bool, device=xyz.device)
    rest_lengths = xyz.new_ones((b, slots))
    for batch, (row, rest) in enumerate(zip(edge_rows, length_rows)):
        size = len(row)
        edges[batch, :size] = row
        edge_mask[batch, :size] = True
        rest_lengths[batch, :size] = rest
    return StructureGraph(edges, edge_mask, rest_lengths, keep, spacing, tuple(records))


def _ratios(pred_xyz, graph):
    if pred_xyz.ndim != 4 or pred_xyz.shape[-1] != 3 or pred_xyz.shape[0] != graph.gs_mask.shape[0] or pred_xyz.shape[2] != graph.gs_mask.shape[1] or pred_xyz.shape[1] < 1:
        raise ValueError('Structure penalty requires final GS centers [B,H,N,3] matching initial graph')
    if pred_xyz.device != graph.edges.device or not pred_xyz.is_floating_point():
        raise ValueError('Prediction and initial graph must share a device')
    valid = graph.gs_mask[:, None, :, None]
    safe = torch.where(valid, pred_xyz, torch.zeros_like(pred_xyz))
    if not torch.isfinite(safe).all():
        raise FloatingPointError('Nonfinite valid predicted GS coordinate')
    batch = torch.arange(len(pred_xyz), device=pred_xyz.device)[:, None]
    # Gather on [B,N,H,3], keeping the edge endpoint indices fixed for all H.
    xyz = safe.transpose(1, 2)
    left = xyz[batch, graph.edges[..., 0]].transpose(1, 2)
    right = xyz[batch, graph.edges[..., 1]].transpose(1, 2)
    length = torch.linalg.vector_norm(left-right, dim=-1)
    ratio = length/graph.rest_lengths[:, None]
    ratio = torch.where(graph.edge_mask[:, None], ratio, torch.ones_like(ratio))
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('Nonfinite local GS length ratio')
    return ratio, safe


def _penalties(ratio):
    stretch = (ratio-RATIO_MAX).clamp_min(0)
    compress = (RATIO_MIN-ratio).clamp_min(0)
    return (F.smooth_l1_loss(stretch, torch.zeros_like(stretch), reduction='none', beta=SMOOTH_L1_BETA)
            + F.smooth_l1_loss(compress, torch.zeros_like(compress), reduction='none', beta=SMOOTH_L1_BETA))


def structure_loss_per_window(pred_xyz, graph):
    """Mean over each window's valid edges and H; empty windows contribute 0."""
    ratio, safe = _ratios(pred_xyz, graph)
    penalty = torch.where(graph.edge_mask[:, None], _penalties(ratio), torch.zeros_like(ratio))
    denominator = (graph.edge_mask.sum(-1)*pred_xyz.shape[1]).clamp_min(1).to(ratio.dtype)
    # Even zero-edge batches keep a finite differentiable zero connected to GS.
    connected_zero = safe.reshape(len(safe), -1)[:, :1].sum(-1)*0.
    return penalty.sum((1, 2))/denominator + connected_zero


def structure_loss(pred_xyz, graph):
    """Window-balanced scalar, never an Anchor-displacement or GT-index loss."""
    return structure_loss_per_window(pred_xyz, graph).mean()


@torch.no_grad()
def structure_statistics(pred_xyz, graph):
    """Per-window/H metrics. Undefined empty-edge quantiles are JSON null."""
    ratio, _ = _ratios(pred_xyz, graph)
    records = []
    for batch in range(len(pred_xyz)):
        for horizon in range(pred_xyz.shape[1]):
            values = ratio[batch, horizon, graph.edge_mask[batch]].double()
            count = len(values)
            row = dict(batch=batch, horizon=horizon+1, edges=count,
                stretch_fraction=float((values > RATIO_MAX).double().mean()) if count else 0.,
                compression_fraction=float((values < RATIO_MIN).double().mean()) if count else 0.,
                loss_struct_unweighted=float(_penalties(values).mean()) if count else 0.)
            for name, vector in (('ratio', values), ('abs_ratio_minus_one', (values-1).abs())):
                row[name+'_mean'] = float(vector.mean()) if count else None
                for label, q in (('p05', .05), ('p50', .5), ('p95', .95)):
                    row[name+'_'+label] = float(torch.quantile(vector, q)) if count else None
                row[name+'_min'] = float(vector.min()) if count else None
                row[name+'_max'] = float(vector.max()) if count else None
            records.append(row)
    return records
