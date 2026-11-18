import torch
import torch.nn as nn
import math
import dgl
import dgl.function as fn

from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList
from mmgs.models.utils import FFN, GroupFFN

from .rotation_layer import RotFFN
from ..builder import ATTENTION
from .initialization import kaiming_uniform_, kaiming_normal_
from .norm_func import mean_std_attn, mean_std_attn_dense, mean_std_attn_dense_rotation, mean_std_attn_dense_rotation_obstacle, mean_std_attn_dense_rotation_unify


@ATTENTION.register_module()
class AttentionTIE(BaseModule):
    """
    Equal to SiT without relation_emb in relation encoder
    num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
    """
    def __init__(self, dim, mesh_heads=4, world_heads=4, dropout=0.0, bias=False, 
                 act_cfg=dict(type='ReLU', inplace=False), norm_cfg=dict(type='LN'),
                 eps=1e-7, add_residual=True, attr_dim=None, **kwargs):
        super(AttentionTIE, self).__init__()
        num_heads = mesh_heads + world_heads
        assert dim % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {dim} and {num_heads}.'
        self.num_heads = num_heads
        self.mesh_heads = mesh_heads
        self.world_heads = world_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.eps = eps
        self.add_residual = add_residual
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = FFN([dim, dim], final_act=False, bias=bias, act_cfg=act_cfg)
        self.norms = ModuleList()
        # self.norms.append(build_norm_layer(norm_cfg, dim)[1])
        # self.norms.append(build_norm_layer(norm_cfg, dim)[1])
        # self.norms.append(build_norm_layer(norm_cfg, dim)[1])
        self.norms.append(nn.Identity())
        self.norms.append(nn.Identity())
        self.norms.append(nn.Identity())
        self.proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)
        self.r_proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)
        self.s_proj = FFN([dim, dim], final_act=False, bias=True, act_cfg=act_cfg)

        # For equally layer normalizations
        self.eps = eps
        self.n_weight = nn.Parameter(torch.Tensor(1, dim))
        self.n_bias = nn.Parameter(torch.Tensor(1, dim))
        nn.init.ones_(self.n_weight)
        nn.init.zeros_(self.n_bias)

        # Attribute
        attr_in = dim if attr_dim is None else attr_dim
        self.attr_weight = FFN(
            [attr_in, dim],
            final_act=True, bias=False)
        self.attr_bias = FFN(
            [attr_in, dim],
            final_act=True, bias=True)

    def head_nodes(self, input_field, outm_field, outw_field):
        '''
            Split into heads and mesh/world splits
        '''
        def func(nodes):
            features = nodes.data[input_field]
            # After: n_verts, num_heads, head_dim
            emb = features.reshape(-1, self.num_heads, self.head_dim)
            mesh_emb = emb[:, :self.mesh_heads]
            world_emb = emb[:, self.mesh_heads:]
            return {outm_field: mesh_emb, outw_field: world_emb}
        return func
    
    def concat_nodes(self, input_field, out_field):
        def func(nodes):
            # n_verts, n_heads, head_dim
            features = nodes.data[input_field]
            features = features.reshape(-1, self.dim)
            return {out_field: features}
        return func

    def forward_feature(self, field):
        if field == 'receiver':
            mlp_func = self.r_proj
            norm_func = self.norms[1]
        elif field == 'sender':
            mlp_func = self.s_proj
            norm_func = self.norms[2]
        else:
            mlp_func = self.proj
            norm_func = self.norms[0]
        def func(nodes):
            feature = nodes.data[field]
            emb = mlp_func(norm_func(feature))
            return {field: emb}
        return func

    def interact_feature(self, src_field, dst_field, out_field):
        '''
            src: sender
            dst: receiver
        '''
        # TODO: check the src/dst. src should be sender
        def func(edges):
            # n_edge, n_head, head_dim
            sender = edges.src[src_field]
            receiver = edges.dst[dst_field]
            l2_norm = torch.linalg.norm(receiver-sender, dim=-1, keepdim=True)
            f = (receiver + sender) / l2_norm
            return {out_field: f}
        return func

    def interact_score(self, node_field, edge_field, out_field):
        def func(edges):
            # n_edge, n_head, head_dim
            query = edges.src[node_field]
            key = edges.data[edge_field]
            # Attention
            score = torch.sum(query*key, dim=-1, keepdim=True)
            score *= self.scale
            # Softmax
            ## Exp here only. Aggregate the normalizer in next step
            score = torch.exp(score.clamp(-5, 5)) # TODO: here is different from previous, previous clamp after norm, which is no clamp
            return {out_field: score}
        return func
    
    def interact_weighted(self, efeature_field, escore_field, out_field):
        def func(edges):
            e_feature = edges.data[efeature_field]
            e_score = edges.data[escore_field]
            weighted_feature = e_feature * e_score
            return {out_field: weighted_feature}
        return func

    def norm_interaction(self, feature_field, score_field, out_field, eps=1e-7):
        def func(nodes):
            feature, z = nodes.data[feature_field], nodes.data[score_field]
            norm_feature = feature / (z + eps)
            return {out_field: norm_feature}
        return func
        
    def merge_mesh_world(self, mesh_field, world_field, out_field):
        def func(nodes):
            # n_verts, n_heads, head_dim
            mesh_f, world_f = nodes.data[mesh_field], nodes.data[world_field]
            feature = torch.cat([mesh_f, world_f], dim=-2)
            return {out_field: feature}
        return func

    def propagate_attention(self, g, receiver_nids, sender_nids, mesh_eids, world_eids):
        # reshape into heads
        g.apply_nodes(self.head_nodes('query', 'q_hm', 'q_hw'))
        g.apply_nodes(self.head_nodes('receiver', 'r_hm', 'r_hw'), receiver_nids)
        g.apply_nodes(self.head_nodes('sender', 's_hm', 's_hw'), sender_nids)

        # Interactions
        ## Mesh domain
        g.apply_edges(self.interact_feature('s_hm', 'r_hm', 'f_hm'), mesh_eids)
        g.apply_edges(self.interact_score('q_hm', 'f_hm', 'score_m'), mesh_eids)
        g.apply_edges(self.interact_weighted('f_hm', 'score_m', 'wf_hm'), mesh_eids)
        ## World domain
        g.apply_edges(self.interact_feature('s_hw', 'r_hw', 'f_hw'), world_eids)
        g.apply_edges(self.interact_score('q_hw', 'f_hw', 'score_w'), world_eids)
        g.apply_edges(self.interact_weighted('f_hw', 'score_w', 'wf_hw'), world_eids)

        # Aggregate
        ## Mesh domain
        g.send_and_recv(mesh_eids, fn.copy_e('wf_hm', 'wf_hm'), fn.sum('wf_hm', 'feature_m'))
        g.send_and_recv(mesh_eids, fn.copy_e('score_m', 'score_m'), fn.sum('score_m', 'z_m'))
        g.apply_nodes(self.norm_interaction('feature_m', 'z_m', 'feature_m'), receiver_nids)
        ## World domain
        g.send_and_recv(world_eids, fn.copy_e('wf_hw', 'wf_hw'), fn.sum('wf_hw', 'feature_w'))
        g.send_and_recv(world_eids, fn.copy_e('score_w', 'score_w'), fn.sum('score_w', 'z_w'))
        g.apply_nodes(self.norm_interaction('feature_w', 'z_w', 'feature_w'), receiver_nids)
        ## Merge mesh & world
        g.apply_nodes(self.merge_mesh_world('feature_m', 'feature_w', 'feature_cat'), receiver_nids)

        # Reshape back
        g.apply_nodes(self.concat_nodes('feature_cat', 'feature'), receiver_nids)
        ## Sender only nodes (outer forces) are not changed
        return g

    def forward(self,
            g, receiver_nids, sender_nids, mesh_eids, world_eids,
            residual=None,
            attr_weight=None, attr_bias=None,
            **kwargs):
        '''
        Args:
            x: n_particles, bs, embed_dim
            attn_mask: bs, headnum, n_particles, n_particles
            receiver_val_res: bs, n_particles, embed_dim Need norm before into this!!! state_receiver
            receiver_val_res: n_particles, bs, embed_dim
            key_padding_mask: bs, num_key,
            mask: bs, N_r, garment+human
            attr_mask: bs, N_r, attr
            invisable_mask: bs, N_r, invisable_mask
        '''
        if attr_weight is not None:
            raise NotImplementedError()
            attr_weight_hidden = self.attr_weight(attr_weight)
            x_receiver *= (1+attr_weight_hidden)
            x_sender[:, :N_r] *= (1+attr_weight_hidden)
        if attr_bias is not None:
            raise NotImplementedError()
            attr_bias_hidden = self.attr_bias(attr_bias)
            x_receiver += attr_bias_hidden
            x_sender[:, :N_r] += attr_bias_hidden

        # Get query
        g.apply_nodes(lambda x: dict(query=self.q_proj(x.data['feature'])), receiver_nids)

        g = self.propagate_attention(g, receiver_nids, sender_nids, mesh_eids, world_eids)
        # Only the garment part is updated
        g.apply_nodes(lambda x: dict(feature=x.data['feature']*self.n_weight+self.n_bias), receiver_nids)
        # x = x.transpose(1, 2).reshape(B, N_r, C)
        # x = x * self.n_weight.reshape(1,1,-1) + self.n_bias.reshape(1,1,-1)
        
        # Since outer forces didn't change here, thus no need update them
        g.apply_nodes(self.forward_feature('feature'), receiver_nids)
        # This can be regarded as a 2 layer MLP on receiver and sender
        g.apply_nodes(self.forward_feature('receiver'), receiver_nids)
        g.apply_nodes(self.forward_feature('sender'), sender_nids)

        return g, None

@ATTENTION.register_module()
class AttentionTIERot(AttentionTIE):
    """
    Equal to SiT without relation_emb in relation encoder
    """
    def __init__(self, dim, mesh_heads=4, world_heads=4, dropout=0.0, bias=False, 
                 act_cfg=dict(type='ReLU', inplace=False), norm_cfg=dict(type='LN'),
                 eps=1e-7, add_residual=True, attr_dim=None, **kwargs):
        super(AttentionTIERot, self).__init__(
            dim=dim, mesh_heads=mesh_heads, world_heads=world_heads, dropout=dropout, bias=bias,
            act_cfg=act_cfg, norm_cfg=norm_cfg,
            eps=eps, add_residual=add_residual, attr_dim=attr_dim, **kwargs)

        mask_type = 2
        # Since the mask is composed of world(4)+mesh(4)
        self.world_rot_mapping = RotFFN([3, self.head_dim], num_groups=self.num_heads//mask_type, final_act=False, bias=False, act_cfg=act_cfg)
        # self.mesh_rot_mapping = RotFFN([3, self.head_dim], num_groups=self.num_heads//mask_type, final_act=False, bias=False, act_cfg=act_cfg)
        self.world_rs_mlp = GroupFFN([self.head_dim, self.head_dim], num_groups=self.num_heads//mask_type, final_act=False, bias=True, act_cfg=act_cfg)
        # self.mesh_rs_mlp = GroupFFN([self.head_dim, self.head_dim], num_groups=self.num_heads//mask_type, final_act=False, bias=True, act_cfg=act_cfg)

    def head_rotations(self, input_field, outw_field):
        '''
            Split into heads and mesh/world splits
            in: rot,
            out: h_rotm, h_rotw

            Here, the mesh rotation is no use, just for easy extend; i.e. add random rotations for mesh to improve robustness
        '''
        def func(nodes):
            f_rotations = nodes.data[input_field]
            # After: n_verts, num_heads, hr_dim, hr_dim
            h_rotations = self.world_rot_mapping(f_rotations)
            world_rot = h_rotations
            # mesh_rot = h_rotations[:, self.mesh_heads:]
            return {outw_field: world_rot}
        return func

    def interact_feature(self, src_field, dst_field, out_field, rot_field=None, mask_field=None):
        '''
            src: sender
            dst: receiver
        '''
        # TODO: check the src/dst. src should be sender
        def func(edges):
            # n_edge, n_head, head_dim
            sender = edges.src[src_field]
            receiver = edges.dst[dst_field]
            l2_norm = torch.linalg.norm(receiver-sender, dim=-1, keepdim=True)
            f = (receiver + sender) / l2_norm

            if mask_field is not None and rot_field is not None:
                ## n_edge, n_head, head_dim, head_dim
                sender_rot = edges.src[rot_field]
                ## n_edge, 1 -> n_edge, 1(head), 1
                force_mask = edges.src[mask_field].unsqueeze(1)
                ## n_edge, n_head, head_dim
                f_r = torch.matmul(sender_rot, f.unsqueeze(-1)).squeeze(-1)
                f_r = self.world_rs_mlp(f_r) # TODO: check, each head give one
                f_r_back = torch.matmul(sender_rot.transpose(-1, -2), f_r.unsqueeze(-1)).squeeze(-1)
                f_comb = f * (1-force_mask) + f_r_back * force_mask
                f = f_comb
            return {out_field: f}
        return func

    def propagate_attention(self, g, receiver_nids, sender_nids, mesh_eids, world_eids):
        # reshape into heads
        g.apply_nodes(self.head_nodes('query', 'q_hm', 'q_hw'))
        g.apply_nodes(self.head_nodes('receiver', 'r_hm', 'r_hw'), receiver_nids)
        g.apply_nodes(self.head_nodes('sender', 's_hm', 's_hw'), sender_nids)
        ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        g.apply_nodes(self.head_rotations('rot', 'rot_hw'), sender_nids) # Only cal world rot
        ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Interactions
        ## Mesh domain
        ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        g.apply_edges(self.interact_feature('s_hm', 'r_hm', 'f_hm'), mesh_eids)
        ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        g.apply_edges(self.interact_score('q_hm', 'f_hm', 'score_m'), mesh_eids)
        g.apply_edges(self.interact_weighted('f_hm', 'score_m', 'wf_hm'), mesh_eids)
        ## World domain
        ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        g.apply_edges(self.interact_feature('s_hw', 'r_hw', 'f_hw', rot_field='rot_hw', mask_field='external'), world_eids)
        ## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        g.apply_edges(self.interact_score('q_hw', 'f_hw', 'score_w'), world_eids)
        g.apply_edges(self.interact_weighted('f_hw', 'score_w', 'wf_hw'), world_eids)

        # Aggregate
        ## Mesh domain
        g.send_and_recv(mesh_eids, fn.copy_e('wf_hm', 'wf_hm'), fn.sum('wf_hm', 'feature_m'))
        g.send_and_recv(mesh_eids, fn.copy_e('score_m', 'score_m'), fn.sum('score_m', 'z_m'))
        g.apply_nodes(self.norm_interaction('feature_m', 'z_m', 'feature_m'), receiver_nids)
        ## World domain
        g.send_and_recv(world_eids, fn.copy_e('wf_hw', 'wf_hw'), fn.sum('wf_hw', 'feature_w'))
        g.send_and_recv(world_eids, fn.copy_e('score_w', 'score_w'), fn.sum('score_w', 'z_w'))
        g.apply_nodes(self.norm_interaction('feature_w', 'z_w', 'feature_w'), receiver_nids)
        ## Merge mesh & world
        g.apply_nodes(self.merge_mesh_world('feature_m', 'feature_w', 'feature_cat'), receiver_nids)

        # Reshape back
        g.apply_nodes(self.concat_nodes('feature_cat', 'feature'), receiver_nids)
        ## Sender only nodes (outer forces) are not changed
        return g

    def forward(self, g, receiver_nids, sender_nids, mesh_eids, world_eids, **kwargs):
        g, _ = super().forward(g, receiver_nids, sender_nids, mesh_eids, world_eids, **kwargs)
        rot_weight = self.world_rot_mapping.get_weight()
        return g, rot_weight



@ATTENTION.register_module()
class AttentionTIERotV2(AttentionTIE):
    """
    Equal to SiT without relation_emb in relation encoder
    """
    def __init__(self, dim, num_heads=8, dropout=0.0, bias=False, activate=False, 
                 act_cfg=dict(type='ReLU', inplace=False), norm_cfg=dict(type='LN'),
                 eps=1e-7, add_residual=True, integ_weight=False, rel_type='NP', dense=False, attr_dim=None, **kwargs):
        super(AttentionTIERotV2, self).__init__(
            dim=dim, num_heads=num_heads, dropout=dropout, bias=bias, activate=activate, 
            act_cfg=act_cfg, norm_cfg=norm_cfg,
            eps=eps, add_residual=add_residual, integ_weight=integ_weight, rel_type=rel_type, dense=dense, **kwargs)
        
        head_dim = dim//self.num_heads
        mask_type = 2
        # Since the mask is composed of world(4)+mesh(4)
        # the input for now is 3D rotation
        self.rot_mapping = RotFFN([3, head_dim], num_groups=self.num_heads//mask_type, final_act=False, bias=False, act_cfg=act_cfg)
        self.rs_mlp = GroupFFN([head_dim, head_dim], num_groups=self.num_heads//mask_type, final_act=True, bias=True, act_cfg=act_cfg)
        # Attribute
        ## Previously, MLP(attr) first, here dim=128;
        ## But performance is much worse, thus directly MLP from the raw attr for each layer
        attr_in = dim if attr_dim is None else attr_dim
        self.attr_weight = FFN(
            [attr_in, dim],
            final_act=True, bias=False)
        self.attr_bias = FFN(
            [attr_in, dim],
            final_act=True, bias=True)

    def forward(self,
            x_state, x_receiver, x_sender,
            h_state, h_sender,
            h_rotation,
            rs_mask=None, h_rs_mask=None,
            # Residual parts
            residual_r=None, residual_s=None, residual_hs=None,
            attr_weight=None, attr_bias=None,
            **kwargs):
        '''
        Args:
            x: n_particles, bs, embed_dim
            attn_mask: bs, headnum, n_particles, n_particles
            receiver_val_res: bs, n_particles, embed_dim Need norm before into this!!! state_receiver
            receiver_val_res: n_particles, bs, embed_dim
            key_padding_mask: bs, num_key,
            mask: bs, N_r, garment+human
            attr_mask: bs, N_r, attr
            invisable_mask: bs, N_r, invisable_mask
            h_rs_mask: bs, N_r, 16
            h_rotation: bs, num_verts, dim, dim
            attr_weight, attr_bias: bs, N_r, dim
        '''
        B, N, C = x_state.shape
        _, _, num_neighbor, _ = h_sender.shape
        N_r = x_receiver.shape[1]
        N_s = x_sender.shape[1]
        N_hs = h_rotation.shape[1]
        mask_type = rs_mask.shape[1]
        head_dim = C // self.num_heads            

        r_mem = self.m_activate(self.memory_weight(x_receiver))
        s_mem = self.m_activate(self.memory_weight(x_sender))
        hs_mem = self.m_activate(self.memory_weight(h_sender))
        x_receiver = self.r_activate(self.receiver_weight(x_state[:, :N_r])) + r_mem
        x_sender = self.s_activate(self.sender_weight(x_state)) + s_mem
        h_sender = self.s_activate(self.sender_weight(h_state)) + hs_mem
        
        if self.add_residual:
            x_receiver += residual_r
            x_sender += residual_s
            h_sender += residual_hs
        
        if attr_weight is not None:
            attr_weight_hidden = self.attr_weight(attr_weight)
            x_receiver *= (1+attr_weight_hidden)
            x_sender[:, :N_r] *= (1+attr_weight_hidden)
        if attr_bias is not None:
            attr_bias_hidden = self.attr_bias(attr_bias)
            x_receiver += attr_bias_hidden
            x_sender[:, :N_r] += attr_bias_hidden

        # After: B, num_heads, N, C//num_heads
        x_receiver = x_receiver.reshape(B, N_r, self.num_heads, head_dim).permute(0, 2, 1, 3)
        x_receiver = x_receiver.reshape(B, self.num_heads // mask_type, mask_type, N_r, head_dim)
        x_sender = x_sender.reshape(B, N_s, self.num_heads, head_dim).permute(0, 2, 1, 3)
        x_sender = x_sender.reshape(B, self.num_heads // mask_type, mask_type, N_s, head_dim)
        h_sender = h_sender.reshape(B, N_r, num_neighbor, self.num_heads, head_dim).permute(0, 3, 1, 2, 4)
        h_sender = h_sender.reshape(B, self.num_heads // mask_type, mask_type, N_r, num_neighbor, head_dim)
        ## Only receiver need query
        q = self.q_proj(x_state[:, :N_r]).reshape(B, N_r, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(B, self.num_heads // mask_type, mask_type, N_r, head_dim)
        # q: B, num_heads, N_r, C//num_heads

        # bs, num_heads / mask_type, mask_type, N_r, N_s
        # bs, num_heads / mask_type, 1, N_r, K, 1
        self_attn_std, inter_attn_std = mean_std_attn_dense_rotation_unify(q, x_receiver, x_sender, h_sender, rs_mask, h_rs_mask, self.scale, eps=self.eps,  **kwargs)

        # Split related part
        ## self part
        ## bs, num_heads/mask_type, mask_type, N_r, dim
        self_x = torch.matmul(self_attn_std, x_sender) + torch.sum(self_attn_std, dim=-1, keepdim=True) * x_receiver
        self_x = self_x.reshape(B, self.num_heads, N_r, head_dim)
        # Only the garment part is updated
        self_x = self_x.transpose(1, 2).reshape(B, N_r, C)

        ## receiver, human part
        ## bs, num_heads/mask_type, mask_type, N_r, num_neighbor, 1
        ### mask_type 0: world mask; 1: mesh mask
        # h_attn = mean_std_attn_dense_rotation_obstacle(
        #     q[:, :, 0:1], x_receiver[:, :, 0:1], h_sender[:, :, 0:1],
        #     h_rs_mask, self.scale, eps=self.eps, **kwargs)
        ## bs, _, _, N_r, num_neighbor, dim
        orig_rot_rs = x_receiver.unsqueeze(4) + h_sender
        rot_rs = orig_rot_rs[:, :, 0:1]
        # mesh_rs = orig_rot_rs[:, :, 1:2]
        h_rotation_hidden = self.rot_mapping(h_rotation, group_input=False)
        h_rotation_hidden = h_rotation_hidden.unsqueeze(2)
        rot_rs = torch.matmul(h_rotation_hidden, rot_rs.unsqueeze(-1)).squeeze(-1)
        assert rot_rs.shape[2] == 1
        rot_rs = self.rs_mlp(rot_rs)
        # Rot each head back to original space
        rot_rs = torch.matmul(h_rotation_hidden.transpose(-1, -2), rot_rs.unsqueeze(-1)).squeeze(-1)
        rot_rs *= inter_attn_std
        rot_rs = rot_rs.reshape(B, self.num_heads//mask_type, N_r, num_neighbor, head_dim)
        rot_rs = rot_rs.permute(0, 2, 3, 1, 4).reshape(B, N_r, num_neighbor, C // mask_type)
        rot_rs *= h_rs_mask.unsqueeze(-1)
        rot_x = torch.sum(rot_rs, dim=-2)

        # mesh_rs = mesh_rs.reshape(B, self.num_heads//mask_type, N_r, num_neighbor, head_dim).permute(0, 2, 3, 1, 4).reshape(B, N_r, num_neighbor, C // mask_type)
        # mesh_rs *= h_rs_mask.unsqueeze(-1)
        # mesh_x = torch.sum(mesh_rs, dim=-2)
        # Since the interact will not happen between human and garment
        mesh_x = torch.zeros_like(rot_x).to(rot_x)
        obstacle_x = torch.cat([rot_x, mesh_x], dim=-1)
        
        ### B, N_r, dim
        x = self_x + obstacle_x
        x = x * self.n_weight.reshape(1,1,-1) + self.n_bias.reshape(1,1,-1)

        x_receiver = x_receiver.reshape(B, self.num_heads, N_r, head_dim).permute(0, 2, 1, 3).reshape(B, N_r, C)
        x_sender = x_sender.reshape(B, self.num_heads, N_s, head_dim).permute(0, 2, 1, 3).reshape(B, N_s, C)
        h_sender = h_sender.reshape(B, self.num_heads, N_r, num_neighbor, head_dim).permute(0, 2, 3, 1, 4).reshape(B, N_r, num_neighbor, C)
        # >>>
        
        x = self.norms[0](x)
        x_receiver = self.norms[1](x_receiver)
        x_sender = self.norms[2](x_sender)
        h_sender = self.norms[2](h_sender)
        x = self.proj(x)
        x_receiver = self.r_proj(x_receiver)
        x_sender = self.s_proj(x_sender)
        h_sender = self.s_proj(h_sender)

        # TODO: check when to update the sender tokens
        x = torch.cat([x, x_state[:, N_r:]], dim=1)
        # if it's the same input as residual later, then after normalization, the embeddings are the same. Which means the sender only didn't change after this block.
        rot_weight = self.rot_mapping.get_weight()
        return x, x_receiver, x_sender, h_state, h_sender, h_rotation, rot_weight
    
@ATTENTION.register_module()
class AttentionLocalEdge(BaseModule):
    """
    Equal to SiT without relation_emb in relation encoder
    num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0, bias=False, 
                 act_cfg=dict(type='ReLU', inplace=False), norm_cfg=dict(type='LN'),
                 eps=1e-7, add_residual=True, **kwargs):
        super(AttentionLocalEdge, self).__init__()
        assert dim % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {dim} and {num_heads}.'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.eps = eps
        self.add_residual = add_residual
        self.dropout = nn.Dropout(dropout)
        
        # self.qkv_proj = nn.Parameter(torch.empty(3*dim, dim))
        # nn.init.xavier_uniform_(self.qkv_proj)
        # self.out_proj = nn.Parameter(torch.empty(dim, dim))
        # nn.init.xavier_uniform_(self.out_proj)
        self.qkv_proj = FFN([dim, 3*dim], final_act=False, bias=bias, act_cfg=None)
        self.out_proj = FFN([dim, dim], final_act=False, bias=bias, act_cfg=None)

    def head_nodes(self, input_field, out_q, out_k, out_v):
        '''
            Split into heads and mesh/world splits
        '''
        def func(nodes):
            features = nodes.data[input_field]
            # n_verts, 3*dim
            proj = self.qkv_proj(features)
            proj = proj.reshape(-1, 3, self.num_heads, self.head_dim)
            # n_verts, n_head, n_hdim
            q = proj[:, 0]
            k = proj[:, 1]
            v = proj[:, 2]
            return {out_q: q, out_k: k, out_v: v}
        return func
    
    def concat_nodes(self, input_field, out_field):
        def func(nodes):
            # n_verts, n_heads, head_dim
            features = nodes.data[input_field]
            features = features.reshape(-1, self.dim)
            features = self.out_proj(features)
            return {out_field: features}
        return func

    def interact_feature(self, q_field, k_field, v_field, out_score, out_feature):
        def func(edges):
            # n_edge, n_head, head_dim
            query = edges.dst[q_field]
            key = edges.src[k_field]
            value = edges.src[v_field]
            # Attention
            score = torch.sum(query*key, dim=-1, keepdim=True)
            score *= self.scale
            # Softmax
            ## Exp here only. Aggregate the normalizer in next step
            score = torch.exp(score.clamp(-5, 5)) # TODO: here is different from previous, previous clamp after norm, which is no clamp
            weighted_feature = score * value

            return {out_score: score, out_feature: weighted_feature}
        return func

    def propagate_attention(self, g, nids, eids, out_verts, suffix):
        query_n = 'Q' + suffix
        key_n = 'K' + suffix
        value_n = 'V' + suffix
        score_n = 'score' + suffix
        sum_score_n = 'sum_score'+suffix
        weighted_feature_n = 'weighted_feature' + suffix
        # Interactions
        g.apply_edges(self.interact_feature(query_n, key_n, value_n, score_n, weighted_feature_n), eids)
        # Aggregate
        ## Aggregate the scores
        g.send_and_recv(eids, fn.copy_e(score_n, score_n), fn.sum(score_n, sum_score_n))
        ## Aggregate the features
        g.send_and_recv(eids, fn.copy_e(weighted_feature_n, weighted_feature_n), fn.sum(weighted_feature_n, weighted_feature_n))
        ## Normalize the feature
        ## One reason that here may occur torch.nan is that, the node is isolated
        ## To avoid potential unstable, we still add eps
        g.apply_nodes(lambda x: {weighted_feature_n: x.data[weighted_feature_n]/(x.data[sum_score_n]+self.eps)}, nids)

        # Reshape back and transform
        g.apply_nodes(self.concat_nodes(weighted_feature_n, out_verts), nids)
        return g

    def forward(self, g, nids, eids, out_vert_for_edge, residual=None, suffix='', **kwargs):
        '''
        Args:
            x: n_particles, bs, embed_dim
            attn_mask: bs, headnum, n_particles, n_particles
            receiver_val_res: bs, n_particles, embed_dim Need norm before into this!!! state_receiver
            receiver_val_res: n_particles, bs, embed_dim
            key_padding_mask: bs, num_key,
            mask: bs, N_r, garment+human
            attr_mask: bs, N_r, attr
            invisable_mask: bs, N_r, invisable_mask
        '''
        # Get query
        g.apply_nodes(self.head_nodes(out_vert_for_edge, 'Q'+suffix, 'K'+suffix, 'V'+suffix), nids)
        # Attention part
        g = self.propagate_attention(g, nids, eids, out_vert_for_edge, suffix)
        if residual is not None:
            g.nodes[nids].data[out_vert_for_edge] += residual
        return g
    
@ATTENTION.register_module()
class AttentionLocalNode(BaseModule):
    """
    Equal to SiT without relation_emb in relation encoder
    num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0, bias=False, 
                 act_cfg=dict(type='ReLU', inplace=False), norm_cfg=dict(type='LN'),
                 eps=1e-7, add_residual=True, **kwargs):
        super(AttentionLocalNode, self).__init__()
        assert dim % num_heads == 0, 'embed_dims must be ' \
            f'divisible by num_heads. got {dim} and {num_heads}.'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.eps = eps
        self.add_residual = add_residual
        self.dropout = nn.Dropout(dropout)

        # self.q_proj = nn.Parameter(torch.empty(dim, dim))
        # self.k_proj = nn.Parameter(torch.empty(dim, dim))
        # self.v_proj = nn.Parameter(torch.empty(dim, dim))
        # nn.init.xavier_uniform_(self.q_proj)
        # nn.init.xavier_uniform_(self.k_proj)
        # nn.init.xavier_uniform_(self.v_proj)
        # self.out_proj = nn.Parameter(torch.empty(dim, dim))
        # nn.init.xavier_uniform_(self.out_proj)
        self.q_proj = FFN([dim, dim], final_act=False, bias=bias, act_cfg=None)
        self.k_proj = FFN([dim, dim], final_act=False, bias=bias, act_cfg=None)
        self.v_proj = FFN([dim, dim], final_act=False, bias=bias, act_cfg=None)
        self.out_proj = FFN([dim, dim], final_act=False, bias=bias, act_cfg=None)

    def head_verts(self, input_field, out_q):
        '''
            Split into heads and mesh/world splits
        '''
        def func(nodes):
            features = nodes.data[input_field]
            # n_verts, 3*dim
            q_proj = self.q_proj(features)
            # n_verts, n_head, n_hdim
            q_proj = q_proj.reshape(-1, self.num_heads, self.head_dim)
            return {out_q: q_proj}
        return func
    
    def head_edges(self, input_field, out_k, out_v):
        '''
            Split into heads and mesh/world splits
        '''
        def func(nodes):
            features = nodes.data[input_field]
            # n_verts, 3*dim
            k_proj = self.k_proj(features)
            v_proj = self.v_proj(features)
            # n_verts, n_head, n_hdim
            k_proj = k_proj.reshape(-1, self.num_heads, self.head_dim)
            v_proj = v_proj.reshape(-1, self.num_heads, self.head_dim)
            return {out_k: k_proj, out_v: v_proj}
        return func
    
    def concat_nodes(self, input_field, out_field):
        def func(nodes):
            # n_verts, n_heads, head_dim
            features = nodes.data[input_field]
            features = features.reshape(-1, self.dim)
            features = self.out_proj(features)
            return {out_field: features}
        return func

    def interact_feature(self, q_field, k_field, v_field, out_score, out_feature):
        def func(edges):
            # n_edge, n_head, head_dim
            query = edges.dst[q_field]
            key = edges.data[k_field]
            value = edges.data[v_field]
            # Attention
            score = torch.sum(query*key, dim=-1, keepdim=True)
            score *= self.scale
            # Softmax
            ## Exp here only. Aggregate the normalizer in next step
            score = torch.exp(score.clamp(-5, 5)) # TODO: here is different from previous, previous clamp after norm, which is no clamp
            weighted_feature = score * value

            return {out_score: score, out_feature: weighted_feature}
        return func

    def propagate_attention(self, g, nids, eids, out_verts, out_edges, suffix):
        query_n = 'Q' + suffix
        key_n = 'K' + suffix
        value_n = 'V' + suffix
        score_n = 'score' + suffix
        sum_score_n = 'sum_score'+suffix
        weighted_feature_n = 'weighted_feature' + suffix
        # TODO: check whether the name is conflict
        # Interactions
        g.apply_edges(self.interact_feature(query_n, key_n, value_n, score_n, weighted_feature_n), eids)
        # Aggregate
        ## Aggregate the scores
        g.send_and_recv(eids, fn.copy_e(score_n, score_n), fn.sum(score_n, sum_score_n))
        ## Aggregate the features
        g.send_and_recv(eids, fn.copy_e(weighted_feature_n, weighted_feature_n), fn.sum(weighted_feature_n, weighted_feature_n))
        ## Normalize the feature
        ## The sum_score could be 0 if there's no in_edges, such as the attr node; though can be avoided by nids, just in case so still add it
        g.apply_nodes(lambda x: {weighted_feature_n: x.data[weighted_feature_n]/(x.data[sum_score_n]+self.eps)}, nids)

        # Reshape back and transform
        g.apply_nodes(self.concat_nodes(weighted_feature_n, out_verts), nids)
        return g

    def forward(self, g, nids, eids, f_eids, out_vert, out_edge, residual=None, suffix='', **kwargs):
        '''
        Args:
            x: n_particles, bs, embed_dim
            attn_mask: bs, headnum, n_particles, n_particles
            receiver_val_res: bs, n_particles, embed_dim Need norm before into this!!! state_receiver
            receiver_val_res: n_particles, bs, embed_dim
            key_padding_mask: bs, num_key,
            mask: bs, N_r, garment+human
            attr_mask: bs, N_r, attr
            invisable_mask: bs, N_r, invisable_mask
        '''
        all_eids = torch.cat([eids, f_eids], dim=-1)
        # Get query
        g.apply_nodes(self.head_verts(out_vert, 'Q'+suffix), nids)
        g.apply_edges(self.head_edges(out_edge, 'K'+suffix, 'V'+suffix), all_eids)
        # Attention part
        g = self.propagate_attention(g, nids, all_eids, out_vert, out_edge, suffix)
        if residual is not None:
            # TODO: check the residaul is changed
            g.nodes[nids].data[out_vert] += residual
        return g

