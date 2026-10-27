"""Scheme B: fixed-identity anchors, live spatial association, unchanged D0."""
import math
import torch
from torch import nn
from scripts.training.model import masked, initial_knn
from scripts.training.gs_decoder import (build_initial_anchor_field,
    pool_anchor_features, interpolate_anchor_displacements)
from .conditions import ToolPredictor, sanitize

ARCH = dict(width=128, heads=4, layers=2, controller_features=7, horizon=15,
    prediction_mode='autoregressive', output_mode='gs_geometry', gs_local_attention='knn',
    gs_local_k=16, gs_decoder='anchors', anchor_count=128, anchor_pool_k=16,
    anchor_bandwidth=1., gs_structure='anchor_attention')


class ToolAnchorAttention(nn.Module):
    """[B,A,D] queries / [B,60,D] tools; fixed zero logit/value null.

    The log Gaussian prior uses live normalized positions and fixed sigma=2*sigma0.
    Bias-free fusion maps an exactly zero observation to exactly zero. This avoids
    a spurious local response when the null receives all attention probability.
    """
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads, self.dim = heads, width//heads
        self.query_norm, self.tool_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.bias = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, heads))
        self.motion_fusion = nn.Sequential(nn.Linear(width+3, width, bias=False), nn.GELU(),
                                           nn.Linear(width, width, bias=False))

    def forward(self, query, centers, anchor_keep, tools, positions, tool_keep,
                displacement, bandwidth, diagnostics=False):
        query, centers = masked(query, anchor_keep), masked(centers, anchor_keep)
        tools, positions = masked(tools, tool_keep), masked(positions, tool_keep)
        displacement = masked(displacement, tool_keep)
        b, a, width = query.shape; count = tools.shape[1]
        if not torch.isfinite(bandwidth).all() or not (bandwidth>0).all():
            raise ValueError('Invalid initial bandwidth')
        q = self.query(self.query_norm(query)).reshape(b,a,self.heads,self.dim)
        z = self.tool_norm(tools)
        key = self.key(z).reshape(b,count,self.heads,self.dim)
        value = self.value(z).reshape_as(key)
        relative = centers[:,:,None]-positions[:,None]
        prior = -.5*(relative/(2*bandwidth[:,None,None,None])).square().sum(-1)
        content = torch.einsum('bahd,bjhd->bahj',q,key)/math.sqrt(self.dim)
        score = content+self.bias(relative).permute(0,1,3,2)+prior[:,:,None]
        allowed=tool_keep[:,None,None] & anchor_keep[:,:,None,None]
        score = score.masked_fill(~allowed,float('-inf'))
        # Null participates in the SAME softmax; tools are never renormalized alone.
        logits = torch.cat((score.new_zeros(b,a,self.heads,1),score),-1)
        alpha = logits.softmax(-1)
        local = alpha[...,1:]
        weighted_value = torch.einsum('bahj,bjhd->bahd',local,value).reshape(b,a,width)
        motion = torch.einsum('baj,bjc->bac',local.mean(2),displacement)
        update = masked(self.motion_fusion(torch.cat((weighted_value,motion),-1)),anchor_keep)
        evidence = dict(alpha=alpha, motion=motion, relative=relative) if diagnostics else None
        return update, evidence


class AnchorBPredictor(ToolPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        expected=dict(gs_decoder='anchors',gs_structure='anchor_attention',
            output_mode='gs_geometry',prediction_mode='autoregressive',gs_local_attention='knn')
        if any(self.config[k]!=v for k,v in expected.items()):
            raise ValueError('B requires the frozen P2 anchor base')
        self.config.update(interaction='tool-anchor-b-v1',tool_bandwidth_multiplier=2.,
                           local_null='fixed-zero-logit-zero-value')
        # Appended AFTER all shared modules, preserving common initial parameters.
        width,heads=self.config['width'],self.config['heads']
        self.tool_attention=ToolAnchorAttention(width,heads)
        self.anchor_to_gs=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))

    def forward(self, inputs, return_trace=False):
        x=sanitize(inputs);keep=x['gs_mask'];b=len(keep);h=self.config['horizon']
        initial=masked(torch.cat((x['gs_xyz'],x['gs_attr']),-1),keep)
        current=initial[...,:3]
        neighbors=initial_knn(current,keep,self.config['gs_local_k'])
        field=build_initial_anchor_field(current,keep,self.config['anchor_count'],
                                        self.config['anchor_pool_k'],self.config['anchor_bandwidth'])
        tokens,valid=self.encode_conditions(x)
        controller=self.controller_inputs(x).reshape(b,h,60,7)
        outputs=[];trace=[];batch=torch.arange(b,device=current.device)[:,None]
        for k in range(h):
            features=self.gs_encoder(masked(torch.cat((current,initial[...,3:]),-1),keep))
            features=self.gs_local(features,current,keep,neighbors)
            features=self.gs_structure(features,current,keep,field)
            query=masked(pool_anchor_features(features,field)+self.prediction_time.weight[k],field.anchor_mask)
            centers=masked(current[batch,field.anchor_indices],field.anchor_mask)
            memory=torch.cat((self.null_token.expand(b,-1,-1),tokens[:,:k+1].flatten(1,2)),1)
            memory_keep=torch.cat((torch.ones(b,1,dtype=torch.bool,device=keep.device),valid[:,:k+1].flatten(1,2)),1)
            global_response=query
            for block in self.blocks:global_response=block(global_response,memory,memory_keep)
            local,evidence=self.tool_attention(query,centers,field.anchor_mask,tokens[:,k,2:62],
                controller[:,k,:,:3],valid[:,k,2:62],controller[:,k,:,3:6],field.bandwidth,return_trace)
            response=masked(self.anchor_to_gs(global_response+local),field.anchor_mask)
            query=masked(features+interpolate_anchor_displacements(response,field,keep),keep)
            delta=self.position_head(pool_anchor_features(query,field))
            current=masked(initial[...,:3]+interpolate_anchor_displacements(delta,field,keep),keep)
            outputs.append(current)
            if return_trace:trace.append(dict(current=current,centers=centers,anchor_delta=delta,**evidence))
        result=dict(xyz=torch.stack(outputs,1),log_scales=initial[:,None,:,3:6].expand(-1,h,-1,-1),
            quaternions=initial[:,None,:,6:10].expand(-1,h,-1,-1),
            opacity_logits=initial[:,None,:,10:11].expand(-1,h,-1,-1),
            sh_dc=initial[:,None,:,11:14].expand(-1,h,-1,-1))
        return (result,trace,field) if return_trace else result
