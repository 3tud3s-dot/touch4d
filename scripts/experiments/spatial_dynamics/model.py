"""One compositional P2 adapter for the preregistered S/D/R/F controls."""
from dataclasses import replace
import torch
from torch import nn
from scripts.training.model import (masked, initial_knn, rotation_vector_to_quaternion,
                                    quaternion_multiply)
from scripts.training.gs_decoder import (build_initial_anchor_field, pool_anchor_features,
                                         interpolate_anchor_displacements)
from scripts.training.gs_structure import StructuralAnchorLayer
from .conditions import ToolPredictor, sanitize

CANDIDATES=('S0','S1','S2-C0','S2-C1','S3-Pre','S3-Post','S4-C0','S4-C1')


def gather(value, ids):
    return value[torch.arange(len(value),device=value.device)[:,None],ids]


def rotate(q, vectors):
    """Unit Hamilton quaternion rotation, broadcasting over Gaussian slots."""
    v=q[...,1:]
    return vectors+2*torch.linalg.cross(v,torch.linalg.cross(v,vectors,dim=-1)+q[...,:1]*vectors,dim=-1)


def se3_decode(initial, field, translation, rotation, quaternions, keep):
    q=rotation_vector_to_quaternion(rotation)
    centers=gather(initial,field.anchor_indices)
    # Chunk over Gaussians to bound intermediate B*N*A*3 graph memory.
    positions=[]
    for start in range(0,initial.shape[1],512):
        relative=initial[:,start:start+512,None]-centers[:,None]
        transformed=rotate(q[:,None].expand(*relative.shape[:-1],4),relative)+centers[:,None]+translation[:,None]
        positions.append((field.weights[:,start:start+512,:,None]*transformed).sum(2))
    xyz=torch.cat(positions,1)
    q=torch.where(q[...,:1]<0,-q,q)
    mixed=torch.bmm(field.weights,q)
    norm=mixed.norm(dim=-1,keepdim=True)
    identity=torch.zeros_like(mixed);identity[...,0]=1
    mixed=torch.where(norm>1e-8,mixed/norm.clamp_min(1e-8),identity)
    axes=quaternion_multiply(mixed,quaternions)
    axes=torch.nn.functional.normalize(axes,dim=-1)
    return masked(xyz,keep),masked(axes,keep)


def local_penalty(xyz, initial, keep, neighbors):
    ids, valid=neighbors
    b=torch.arange(len(initial),device=initial.device)[:,None,None]
    valid=valid & keep[...,None] & (ids!=torch.arange(initial.shape[1],device=initial.device)[None,:,None])
    if not valid.any():return xyz.sum()*0
    d0=(initial[:,:,None]-initial[b,ids]).norm(dim=-1)
    values=[]
    for h in range(xyz.shape[1]):
        current=xyz[:,h]
        dist=(current[:,:,None]-current[b,ids]).norm(dim=-1)
        excess=((dist-d0).abs()-.1*d0).clamp_min(0)
        values.append(torch.nn.functional.smooth_l1_loss(excess[valid],torch.zeros_like(excess[valid]),beta=.01))
    return torch.stack(values).mean()


class SpatialPredictor(ToolPredictor):
    def __init__(self, spatial='S0', decoder='D0', feedback='F0', regularization=0., **kwargs):
        if spatial not in CANDIDATES or decoder not in ('D0','D1','D2') or feedback not in ('F0','F1'):
            raise ValueError('Unknown research mode')
        if regularization not in (0.,.01):raise ValueError('Unregistered R1 weight')
        if decoder=='D1' and not spatial.startswith('S4'):raise ValueError('D1 requires S4')
        if feedback=='F1' and decoder!='D0':raise ValueError('F1 only paired with D0')
        super().__init__(**kwargs)
        if any(self.config[k]!=v for k,v in dict(gs_decoder='anchors',gs_structure='anchor_attention',
                output_mode='gs_geometry',prediction_mode='autoregressive').items()):
            raise ValueError('Spatial research requires unchanged P2 base')
        self.config.update(spatial=spatial,decoder=decoder,feedback=feedback,regularization=regularization)
        width,heads=self.config['width'],self.config['heads']
        # All common parameters precede additional modules, preserving RNG pairing.
        self.spatial_bias=None
        if spatial not in ('S0','S2-C0'):
            self.spatial_bias=nn.ModuleList(nn.Sequential(nn.Linear(3,width),nn.GELU(),nn.Linear(width,heads))
                                            for _ in self.blocks)
            for bias in self.spatial_bias:
                nn.init.zeros_(bias[-1].weight);nn.init.zeros_(bias[-1].bias)
        self.propagation=None
        if spatial.startswith('S3'):
            self.propagation=StructuralAnchorLayer(width,heads)
            self.propagation_output=nn.Linear(width,width)
            nn.init.zeros_(self.propagation_output.weight);nn.init.zeros_(self.propagation_output.bias)
        if decoder in ('D1','D2'):
            self.deformation_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,3))
            nn.init.zeros_(self.deformation_head[-1].weight);nn.init.zeros_(self.deformation_head[-1].bias)
        self.trace=[]

    def tool_geometry(self,x,k):
        xyz=x['controller_points'][:,k].reshape(len(x['gs_xyz']),60,3)
        keep=x['controller_mask'][:,k].reshape(len(xyz),60)
        mean,maximum,coverage,age,obs,_=self.aggregate(x)
        strength=(mean[:,k]+maximum[:,k]).reshape(len(xyz),60).clamp_min(0)
        alpha=(coverage[:,k].reshape_as(strength)*strength/(1+strength))
        alpha=alpha*obs[:,k].reshape_as(keep)*keep
        return xyz,keep,alpha

    def cross(self,query,positions,memory,valid,tools,tool_keep,k):
        b,n,_=query.shape
        for index,block in enumerate(self.blocks):
            if self.spatial_bias is None:
                query=block(query,memory,valid)
                continue
            # Only latest 60 tool slots get live geometry. Older tokens have zero bias.
            relative=tools[:,None]-positions[:,:,None]
            bias=self.spatial_bias[index](relative).permute(0,3,1,2)
            bias=bias*tool_keep[:,None,None].to(bias.dtype)
            prefix=1+66*k+2
            additive=torch.cat((bias.new_zeros(b,self.config['heads'],n,prefix),bias,
                               bias.new_zeros(b,self.config['heads'],n,4)),dim=-1)
            additive=additive.masked_fill(~valid[:,None,None],float('-inf'))
            mem=block.memory_norm(memory)
            update,_=block.attention(block.query_norm(query),mem,mem,
                                    attn_mask=additive.reshape(b*self.config['heads'],n,-1),need_weights=False)
            query=query+update
            query=query+block.ff(block.ff_norm(query))
        return query

    def propagate(self,query,positions,keep):
        return masked(query+self.propagation_output(self.propagation(query,positions,keep)),keep)

    def dynamic_field(self,current,keep,fixed,tools,tool_keep,alpha):
        """Hard deterministic IDs; feature pools and positions retain gradients."""
        with torch.no_grad():
            scores=torch.exp(-.5*torch.cdist(current,tools).square()/.1**2)
            scores=scores*tool_keep[:,None]
            if self.config['spatial']=='S4-C1':scores=scores*alpha[:,None]
            relevance=scores.amax(-1)*keep
            ids=fixed.anchor_indices.clone();mask=fixed.anchor_mask.clone()
            count=ids.shape[1];global_count=min(96,max(1,3*count//4))
            active=relevance.amax(-1)>0
            for b in range(len(current)):
                if not active[b]:continue
                candidates=keep[b].clone()
                candidates[ids[b,:global_count][mask[b,:global_count]]]=False
                remaining=candidates.nonzero(as_tuple=True)[0]
                ranked=remaining[relevance[b,remaining].argsort(descending=True,stable=True)]
                used=min(count-global_count,len(ranked))
                mask[b,global_count:]=False;ids[b,global_count:]=0
                ids[b,global_count:global_count+used]=ranked[:used]
                mask[b,global_count:global_count+used]=True
            # Gather neighbor indices only; selected encoded features remain live.
            distances=torch.cdist(gather(current,ids),current).masked_fill(~keep[:,None],float('inf'))
            pool=distances.argsort(dim=-1,stable=True)[...,:fixed.pool_indices.shape[-1]]
            pool_keep=gather(keep[...,None],pool.reshape(len(current),-1)).squeeze(-1).reshape_as(pool)
            pool_keep=pool_keep & mask[...,None]
        return replace(fixed,anchor_indices=ids,anchor_mask=mask,pool_indices=pool,pool_mask=pool_keep),relevance,active,global_count

    def forward(self,inputs):
        x=sanitize(inputs);keep=x['gs_mask'];initial=masked(torch.cat((x['gs_xyz'],x['gs_attr']),-1),keep)
        current=initial[...,:3];quats=initial[...,6:10]
        neighbors=initial_knn(current,keep,self.config['gs_local_k'])
        fixed=build_initial_anchor_field(current,keep,self.config['anchor_count'],self.config['anchor_pool_k'],self.config['anchor_bandwidth'])
        tokens,token_valid=self.encode_conditions(x)
        b=len(current);positions=[];rotations=[];self.trace=[];total=None
        for k in range(self.config['horizon']):
            attrs=torch.cat((initial[...,3:6],quats,initial[...,10:]),-1)
            features=self.gs_encoder(masked(torch.cat((current,attrs),-1),keep))
            features=self.gs_local(features,current,keep,neighbors)
            features=self.gs_structure(features,current,keep,fixed)
            tools,tool_keep,alpha=self.tool_geometry(x,k)
            field=fixed;relevance=None;active=None;global_count=0
            if self.config['spatial'].startswith('S4'):
                field,relevance,active,global_count=self.dynamic_field(current,keep,fixed,tools,tool_keep,alpha)
            per_gs=self.config['spatial'] in ('S0','S1')
            query=features if per_gs else pool_anchor_features(features,field)
            centers=current if per_gs else masked(gather(current,field.anchor_indices),field.anchor_mask)
            query=query+self.prediction_time.weight[k]
            if self.config['spatial']=='S3-Pre':query=self.propagate(query,centers,field.anchor_mask)
            memory=torch.cat((self.null_token.expand(b,-1,-1),tokens[:,:k+1].flatten(1,2)),1)
            valid=torch.cat((torch.ones(b,1,dtype=torch.bool,device=keep.device),token_valid[:,:k+1].flatten(1,2)),1)
            query=self.cross(query,centers,memory,valid,tools,tool_keep,k)
            if self.config['spatial']=='S3-Post':query=self.propagate(query,centers,field.anchor_mask)
            anchors=pool_anchor_features(query,fixed) if per_gs else masked(query,field.anchor_mask)
            dynamic_features=anchors
            if relevance is not None:
                fixed_pos=gather(current,fixed.anchor_indices)
                dynamic_pos=gather(current,field.anchor_indices)
                logits=-.5*(torch.cdist(fixed_pos,dynamic_pos)/fixed.bandwidth[:,None,None]).square()
                logits=logits.masked_fill(~field.anchor_mask[:,None],float('-inf'))
                logits=torch.where(field.anchor_mask.any(-1)[:,None,None],logits,0.)
                mapping=logits.softmax(-1).masked_fill(~field.anchor_mask[:,None],0.)
                anchors=masked(torch.bmm(mapping,anchors),fixed.anchor_mask)
                # A no-contact fallback is exactly the fixed-anchor path.
                anchors=torch.where(active[:,None,None],anchors,dynamic_features)
            displacement=masked(self.position_head(anchors),fixed.anchor_mask)
            if self.config['feedback']=='F1':
                total=displacement if total is None else total+displacement
                displacement=total
            if self.config['decoder']=='D2':
                current,quats=se3_decode(initial[...,:3],fixed,displacement,
                    masked(self.deformation_head(anchors),fixed.anchor_mask),initial[...,6:10],keep)
            else:
                current=masked(initial[...,:3]+interpolate_anchor_displacements(displacement,fixed,keep),keep)
                if self.config['decoder']=='D1':
                    local_pos=gather(current,field.anchor_indices)[:,global_count:]
                    local_keep=field.anchor_mask[:,global_count:]
                    score=torch.exp(-.5*torch.cdist(current,tools).square()/.1**2)*alpha[:,None]*tool_keep[:,None]
                    gate=score.amax(-1)*keep
                    logit=-.5*(torch.cdist(current,local_pos)/fixed.bandwidth[:,None,None]).square()
                    logit=logit.masked_fill(~local_keep[:,None],float('-inf'))
                    logit=torch.where(local_keep.any(-1)[:,None,None],logit,0.)
                    weights=logit.softmax(-1).masked_fill(~local_keep[:,None],0.)
                    delta=masked(self.deformation_head(dynamic_features[:,global_count:]),local_keep)
                    current=masked(current+gate[...,None]*torch.bmm(weights,delta),keep)
            positions.append(current);rotations.append(quats)
            # Detached diagnostics never feed the model. Full prediction graph stays live.
            self.trace.append(dict(anchor_ids=field.anchor_indices.detach(),anchor_mask=field.anchor_mask.detach(),
                displacement=displacement.detach(),fixed_ids=fixed.anchor_indices.detach(),fixed_mask=fixed.anchor_mask.detach(),
                current=current,bandwidth=fixed.bandwidth.detach()))
        xyz=torch.stack(positions,1);h=self.config['horizon']
        return dict(xyz=xyz,log_scales=initial[:,None,:,3:6].expand(-1,h,-1,-1),
            quaternions=torch.stack(rotations,1),opacity_logits=initial[:,None,:,10:11].expand(-1,h,-1,-1),
            sh_dc=initial[:,None,:,11:14].expand(-1,h,-1,-1))
