"""Geometry-initialized anchor interactions around the UNCHANGED V4 rollout.

Controllers precede anchors in the returned hybrid weights, so the original
V4 forward's alpha[..., :60] remains exactly the external-tool subset.
There is one joint softmax over those keys, other anchors, and the zero null.
"""
import copy
import math
import torch
from torch import nn
from scripts.training.model import masked
from scripts.training.v4_attention import GeometryCrossBlock
from scripts.training.v4_model import V4Stage1Predictor

MODES = {'G0': 'baseline', 'G1': 'pre_a2a', 'G2': 'post_a2a', 'G3': 'hybrid'}


class AnchorInteraction(GeometryCrossBlock):
    """Same learned score and residual/FFN as V4; own-index keys excluded."""
    def attend(self, query, memory, qxyz, kxyz, sigma, qkeep, kkeep, self_offset):
        b, a, width = query.shape
        n = memory.shape[1]
        query, memory = masked(query, qkeep), masked(memory, kkeep)
        q = self.q(self.query_norm(query)).reshape(b, a, self.heads, self.dim)
        k = self.k(self.memory_norm(memory)).reshape(b, n, self.heads, self.dim)
        v = self.v(self.memory_norm(memory)).reshape(b, n, self.heads, self.dim)
        relative = (masked(qxyz, qkeep)[:, :, None] - masked(kxyz, kkeep)[:, None]) / sigma[:, None, None, None]
        geometry = torch.cat((relative, relative.square().sum(-1, keepdim=True)), -1)
        scores = []
        for h in range(self.heads):
            qh = q[:, :, h, None].expand(-1, -1, n, -1)
            kh = k[:, None, :, h].expand(-1, a, -1, -1)
            s = (qh*kh).sum(-1)/math.sqrt(self.dim)
            s = s + (geometry*self.geometry_score[h]).sum(-1)
            scores.append(s + self.pairwise[h](torch.cat((qh, kh, geometry), -1)).squeeze(-1))
        scores = torch.stack(scores, 2)
        allowed = (qkeep[:, :, None] & kkeep[:, None]).clone()
        if self_offset is not None:
            ix = torch.arange(a, device=query.device)
            allowed[:, ix, ix+self_offset] = False
        scores = scores.masked_fill(~allowed[:, :, None], -torch.inf)
        alpha = torch.cat((scores, scores.new_zeros(b, a, self.heads, 1)), -1).softmax(-1)
        update = torch.einsum('bahj,bjhd->bahd', alpha[..., :n], v).reshape(b, a, width)
        result = query + self.projection(update)
        return masked(result + self.ff(self.ff_norm(result)), qkeep), alpha

    def forward(self, query, xyz, sigma, keep):
        return self.attend(query, query, xyz, xyz, sigma, keep, keep, 0)


class PlacedCross(GeometryCrossBlock):
    """Keeps original cross parameter names and per-head controller weights."""
    def __init__(self, placement):
        super().__init__()
        self.placement = placement
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4201)
            self.a2a = nn.ModuleList(AnchorInteraction() for _ in range(2))
        self.collect = False
        self.records = []

    def forward(self, query, tools, xyz, cp, sigma, am, tm):
        weights = []
        if self.placement == 'pre_a2a':
            for block in self.a2a:
                query, alpha = block(query, xyz, sigma, am)
                weights.append(alpha)
        query, controller_alpha = super().forward(query, tools, xyz, cp, sigma, am, tm)
        if self.placement == 'post_a2a':
            for block in self.a2a:
                query, alpha = block(query, xyz, sigma, am)
                weights.append(alpha)
        if self.collect:
            self.records.append(dict(anchor_weights=weights, controller_weights=controller_alpha))
        return query, controller_alpha


class HybridCross(AnchorInteraction):
    def __init__(self):
        super().__init__()
        # Explicit node types, appended after shared V4 modules. Zero at init.
        self.node_types = nn.Parameter(torch.zeros(2, 128))
        self.collect = False
        self.records = []

    def forward(self, query, tools, xyz, cp, sigma, am, tm):
        memory = torch.cat((masked(tools+self.node_types[1], tm),
                            masked(query+self.node_types[0], am)), 1)
        kxyz = torch.cat((cp, xyz), 1)
        keep = torch.cat((tm, am), 1)
        result, alpha = self.attend(query, memory, xyz, kxyz, sigma, am, keep, tools.shape[1])
        if self.collect:
            self.records.append(dict(hybrid_weights=alpha))
        return result, alpha


class InteractionPredictor(V4Stage1Predictor):
    def __init__(self, mode, horizon=10, prediction_interval=1):
        if mode not in ('pre_a2a', 'post_a2a', 'hybrid'):
            raise ValueError('Use the original V4 class for baseline')
        super().__init__(horizon, prediction_interval)
        if mode == 'hybrid':
            self.cross = nn.ModuleList(HybridCross() for _ in range(2))
        else:
            index = 0 if mode == 'pre_a2a' else 1
            self.cross[index] = PlacedCross(mode)
        self.config = dict(self.config, anchor_interaction=mode, a2a_layers=2,
                           interaction_self_keys=False, interaction_initialization='v4-geometry-qzero-pairzero-seed4201')

    def forward(self, x, *, return_trace=False):
        for block in self.cross:
            if hasattr(block, 'collect'):
                block.collect, block.records = return_trace, []
        result = super().forward(x, return_trace=return_trace)
        if return_trace:
            prediction, trace = result
            trace['interactions'] = {str(i): block.records for i, block in enumerate(self.cross)
                                     if hasattr(block, 'records')}
            for block in self.cross:
                if hasattr(block, 'records'): block.records = []
            return prediction, trace
        return result


def paired_models(horizon=10):
    """Copy all same-name/shape tensors; never rely on constructor RNG order."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = V4Stage1Predictor(horizon=horizon, prediction_interval=1)
    state = copy.deepcopy(baseline.state_dict())
    result = {'G0': baseline}
    for name in ('G1', 'G2', 'G3'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4201)
            m = InteractionPredictor(MODES[name], horizon=horizon)
        own = m.state_dict()
        for n, v in state.items():
            if n not in own or own[n].shape != v.shape:
                raise ValueError('Missing common V4 parameter: '+n)
            own[n] = v.clone()
        m.load_state_dict(own, strict=True)
        result[name] = m
    return result
