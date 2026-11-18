import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_norm_layer)
from mmcv.runner import BaseModule, ModuleList
from mmcv.ops import QueryAndGroup, grouping_operation

from .. import builder
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
# from mmgd.models.utils import (FFN, AttentionTIE, kaiming_uniform_, kaiming_normal_)
from mmgs.models.utils import (FFN, AttentionTIE, TimeEmbedding, TimeResidualFFN, Normalizer)
from mmgs.datasets.utils import normalize
from mmgs.utils import face_normals_batched, vertex_normal_batched, rotation_from_normals
from mmgs.core import multi_apply
import dgl
import dgl.function as fn


class MeshGraphNetEncoderLayer(BaseModule):
    def __init__(self,
                embed_dims,
                dropout=0.0,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN'),
                pre_norm=False,
                **kwargs):
        super(MeshGraphNetEncoderLayer, self).__init__()
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.pre_norm = pre_norm
        
        self.norms = ModuleList()
        if self.pre_norm:
            self.norms.append(build_norm_layer(norm_cfg, embed_dims*3)[1])
            self.norms.append(build_norm_layer(norm_cfg, embed_dims*2)[1])
        else:
            self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
            self.norms.append(build_norm_layer(norm_cfg, embed_dims)[1])
        
        # For receiver and sender
        self.edge_weight = FFN([embed_dims*3, embed_dims, embed_dims, embed_dims], bias=True, act_cfg=act_cfg, add_residual=True)
        self.node_weight = FFN([embed_dims*2, embed_dims, embed_dims, embed_dims], bias=True, act_cfg=act_cfg, add_residual=True)
    
    def interact_feature(self, mlp_func, norm_func, edge_field, src_field, dst_field, out_field):
        """
        src: sender
        dst: receiver

        edge_field: world/mesh
        src/dst_field: n_feature,
        out_field: n_feature
        """
        def func(edges):
            sender = edges.src[src_field]
            receiver = edges.dst[dst_field]
            interactions = edges.data[edge_field]
            in_emb = torch.cat([interactions, receiver, sender], dim=-1)
            if self.pre_norm:
                in_emb = norm_func(in_emb)
            f = mlp_func(in_emb, residual=interactions)
            if not self.pre_norm:
                f = norm_func(f)
            return {out_field: f}
        return func
    
    def node_feature(self, mlp_func, norm_func, node_field, edge_field, out_field):
        def func(nodes):
            node_f = nodes.data[node_field]
            mesh_f = nodes.data[edge_field]
            in_emb = torch.cat([node_f, mesh_f], dim=-1)
            if self.pre_norm:
                in_emb = norm_func(in_emb)
            f = mlp_func(in_emb, residual=node_f)
            if not self.pre_norm:
                f = norm_func(f)
            return {out_field: f}
        return func
    
    def forward(self, g, out_node_field, out_edge_field):
        g.apply_edges(self.interact_feature(self.edge_weight, self.norms[0], out_edge_field, out_node_field, out_node_field, out_edge_field))
        g.send_and_recv(g.edges(), fn.copy_e(out_edge_field, out_edge_field), fn.sum(out_edge_field, out_edge_field))
        if g.num_edges() == 0:
            # Nodes are too far away
            g.ndata[out_edge_field] = torch.zeros_like(g.ndata[out_node_field]).to(g.ndata[out_node_field])
        # Update nodes
        g.apply_nodes(self.node_feature(self.node_weight, self.norms[1], out_node_field, out_edge_field, out_node_field))

        return g

class MeshGraphNetEncoder(BaseModule):
    def __init__(self,
                num_layers,
                embed_dims,
                dropout=0.0,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN'),
                pre_norm=False,
                **kwargs):
        super(MeshGraphNetEncoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.dropout = dropout
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.layers = ModuleList()
        self.pre_norm = pre_norm

        if pre_norm:
            self.emb_norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        for _ in range(num_layers):
            self.layers.append(
                MeshGraphNetEncoderLayer(embed_dims, dropout, act_cfg, norm_cfg, pre_norm=pre_norm, **kwargs))
    
    def forward(self, g, out_node_field, out_edge_field):
        for layer in self.layers:
            g = layer(g, out_node_field, out_edge_field)
        return g

@BACKBONES.register_module()
class SkeletonGraphNet(BaseBackbone):
    def __init__(self, 
                attr_dim = 5,
                state_dim=6,
                position_dim=3,
                num_frames=1+1,  #?   
                embed_dims=128,
                num_encoder_layers=4,
                dropout=0.0,
                eps=1e-7,
                num_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg = dict(type='LN'),
                pre_norm=False,
                dt=1/30,
                norm_acc_steps=None,
                attr_mode='cat',
                edge_mode='ratio',
                edge_theta=False,
                fix_bug=False,
                **kwargs):
        super(SkeletonGraphNet, self).__init__()
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        self.position_dim = position_dim
        self.embed_dims = embed_dims
        self.eps = eps
        self.num_frames = num_frames
        assert num_frames == 2
        self.dt = dt
        self.norm_acc_steps = norm_acc_steps
        self.attr_mode = attr_mode
        self.edge_mode = edge_mode
        self.edge_theta = edge_theta
        self.fix_bug = fix_bug

        # Attribute
        ##attr_dim
        if attr_mode == 'mul':
            # Multiply
            self.attr_encoder = FFN(
                [attr_dim] + [embed_dims for i in range(num_fcs)],
                final_act=True, bias=True)
        else:
            assert attr_mode == 'cat'
            self.attr_encoder = nn.Identity()
        # self.attr_normalizer = Normalizer(attr_dim) if norm_acc_steps is None else Normalizer(attr_dim, max_accumulations=norm_acc_steps)
        self.attr_normalizer = nn.Identity()

        # States embeddings
        ## vel
        node_dim = 2*position_dim+attr_dim if attr_mode == 'cat' else 2*position_dim # Only for input the encoder
        self.node_encoder = FFN(
            [node_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=True, bias=True)
        self.node_normalizer = Normalizer(position_dim) if norm_acc_steps is None else Normalizer(position_dim, max_accumulations=norm_acc_steps)
        self.anchor_normalizer = Normalizer(position_dim) if norm_acc_steps is None else Normalizer(position_dim, max_accumulations=norm_acc_steps)
        self.node_norm = build_norm_layer(norm_cfg, embed_dims)[1] if not pre_norm else nn.Identity()

        # Edge deltaX, |deltaX|; for current state and template
        if self.edge_mode == 'ordinary':
            mesh_in_dim = (3+1)*2
        elif self.edge_mode == 'ratio':
            mesh_in_dim = 3+1 # only direction, 1 change of ratio
        else:
            assert False, "Not yet"
        if fix_bug:
            # Include extra velocity
            mesh_in_dim += 3
        mesh_in_enc_dim = mesh_in_dim
        if edge_theta:
            mesh_in_enc_dim += 2 # cos, sin
        self.edge_encoder = FFN(
            [mesh_in_enc_dim] + [embed_dims for i in range(num_fcs)], 
            final_act=True, bias=True)
        self.edge_normalizer = Normalizer(mesh_in_dim) if norm_acc_steps is None else Normalizer(mesh_in_dim, max_accumulations=norm_acc_steps)
        self.edge_norm = build_norm_layer(norm_cfg, embed_dims)[1] if not pre_norm else nn.Identity()

        self.encoder = MeshGraphNetEncoder(num_encoder_layers, embed_dims, dropout, act_cfg, norm_cfg, pre_norm=pre_norm, **kwargs)

    def _edge_theta(self, recv_state, send_state):
        recv_vec = recv_state # - anchor_recv_state
        send_vec = send_state # - anchor_send_state
        if self.fix_bug:
            normed_recv = recv_vec / torch.linalg.norm(recv_vec, dim=-1, keepdim=True)
            normed_send = send_vec / torch.linalg.norm(send_vec, dim=-1, keepdim=True)
            cos = torch.sum(normed_recv * normed_send, dim=-1, keepdim=True)
            sin = torch.linalg.norm(torch.cross(normed_recv, normed_send, dim=-1), dim=-1, keepdim=True)
        else:
            cos = torch.sum(recv_vec * send_vec, dim=-1, keepdim=True)
            sin = torch.linalg.norm(torch.cross(recv_vec, send_vec, dim=-1), dim=-1, keepdim=True)
        return cos, sin

    def init_node_features(self, 
                            prev_state_field, cur_state_field,
                            attr_field, external_field, pin_mask_field,
                            node_out_field):
        def func(nodes):
            # Static info
            attr = nodes.data[attr_field]
            pin_mask = nodes.data[pin_mask_field]
            external_forces = nodes.data[external_field]

            prev_state = nodes.data[prev_state_field]
            cur_state = nodes.data[cur_state_field]
            vel = (cur_state - prev_state) / self.dt * torch.logical_not(pin_mask)
            
            norm_vel = self.node_normalizer(vel)
            norm_attr = self.attr_normalizer(attr)

            norm_external_forces = self.anchor_normalizer(external_forces)
            in_dynamic = torch.cat([norm_external_forces, norm_vel], dim=-1)
            if self.attr_mode == 'cat':
                in_raw = torch.cat([in_dynamic, norm_attr], dim=-1)
                in_emb = self.node_encoder(in_raw)
            else:
                assert self.attr_mode == 'mul'
                in_vel = self.node_encoder(in_dynamic)
                in_attr = self.attr_encoder(norm_attr)
                in_emb = in_vel * (1+in_attr)
            in_feature = self.node_norm(in_emb)
            return {node_out_field: in_feature}
        return func
    
    def init_edge_features(self,
                            cur_state_field, template_state_field, edge_out_field):
        def func(edges):
            # n_edges, 3
            recv_state = edges.dst[cur_state_field]
            send_state = edges.src[cur_state_field]
            recv_tem   = edges.dst[template_state_field]
            send_tem   = edges.src[template_state_field]

            # n_edges, 3
            delta_state = (recv_state - send_state)
            delta_pos   = delta_state
            # n_edges, 1
            norm_delta_pos = torch.linalg.norm(delta_pos, dim=-1, keepdim=True)

            delta_tem = recv_tem - send_tem
            norm_delta_tem = torch.linalg.norm(delta_tem, dim=-1, keepdim=True)

            if self.edge_mode == 'ordinary':
                in_emb = self.edge_normalizer(
                    torch.cat([delta_pos, norm_delta_pos, delta_tem, norm_delta_tem], dim=-1)
                )
            else:
                assert self.edge_mode == 'ratio'
                in_emb = self.edge_normalizer(
                    torch.cat([delta_pos/norm_delta_pos, norm_delta_pos/norm_delta_tem], dim=-1)
                )
            
            if self.edge_theta:
                tem_cos, tem_sin = self._edge_theta(recv_tem  , send_tem  )
                cur_cos, cur_sin = self._edge_theta(recv_state, send_state)
                # cos(A-B) = cosAcosB + sinAsinB
                # sin(A-B) = sinAcosB - cosAsinB
                delta_cos = cur_cos * tem_cos + cur_sin * tem_sin
                delta_sin = cur_sin * tem_cos - cur_cos * tem_sin
                in_emb = torch.cat([in_emb, delta_cos, delta_sin], dim=-1)
            
            out_emb = self.edge_norm(self.edge_encoder(in_emb))
            return {edge_out_field: out_emb}
        
        return func
    
    def init_edge_features_fixbug(self,
                                    cur_state_field, prev_state_field, orig_template_state_field, edge_out_field):
        def func(edges):
            # n_edges, 3
            recv_state = edges.dst[cur_state_field]
            send_state = edges.src[cur_state_field]
            recv_prev = edges.dst[prev_state_field]
            send_prev = edges.src[prev_state_field]

            recv_orig_tem = edges.dst[orig_template_state_field]
            send_orig_tem = edges.src[orig_template_state_field]

            # n_edges, 3
            delta_state = recv_state - send_state
            delta_pos = delta_state
            delta_prev = recv_prev - send_prev

            delta_rel_vel = (delta_state - delta_prev) / self.dt
            norm_delta_pos = torch.linalg.norm(delta_pos, dim=-1, keepdim=True)

            delta_orig_tem = recv_orig_tem - send_orig_tem
            norm_delta_orig_tem = torch.linalg.norm(delta_orig_tem, dim=-1, keepdim=True)

            if self.edge_mode == 'ordinary':
                in_emb = self.edge_normalizer(
                    torch.cat([delta_pos, norm_delta_pos, delta_orig_tem, norm_delta_orig_tem, delta_rel_vel], dim=-1)
                )
            else:
                assert self.edge_mode == 'ratio'
                in_emb = self.edge_normalizer(
                    torch.cat([delta_pos/norm_delta_pos, norm_delta_pos/norm_delta_orig_tem, delta_rel_vel], dim=-1)
                )
            
            if self.edge_theta:
                tem_cos, tem_sin = self._edge_theta(recv_orig_tem, send_orig_tem)
                cur_cos, cur_sin = self._edge_theta(recv_state, send_state)
                # cos(a-b) = cosAcosB + sinAsinB
                # sin(A-B) = sinAcosB - cosAsinB
                delta_cos = cur_cos * tem_cos + cur_sin * tem_sin
                delta_sin = cur_sin * tem_cos - cur_cos * tem_sin
                in_emb = torch.cat([in_emb, delta_cos, delta_sin], dim=-1)
            
            out_emb = self.edge_norm(self.edge_encoder(in_emb))
            return {edge_out_field: out_emb}
        return func

    def init_features(self, graph,
                        prev_state_field, cur_state_field,
                        template_state_field, orig_template_state_field,
                        attr_field, external_field, pin_mask_field,
                        node_out_field, edge_out_field):
        # Apply to all nodes
        graph.apply_nodes(self.init_node_features(
            prev_state_field, cur_state_field,
            attr_field, external_field, pin_mask_field,
            node_out_field))
        # Apply to all edges
        if self.fix_bug:
            graph.apply_edges(self.init_edge_features_fixbug(
                cur_state_field, prev_state_field, orig_template_state_field, edge_out_field))
        else:
            graph.apply_edges(self.init_edge_features(
                cur_state_field, template_state_field, edge_out_field))
        return graph

    def forward(self, skeleton_graph, **kwargs):
        # Embed features
        out_node_field = 'out_node'
        out_edge_field = 'out_edge'
        #node keys: ['attr', 'prev_state', 'cur_state', 'template_state', 'orig_template_state', 'pin_mask', 'external']
        skeleton_graph = self.init_features(
            skeleton_graph,
            'prev_state', 'cur_state',
            'template_state', 'orig_template_state',
            'attr', 'external', 'pin_mask',
            out_node_field, out_edge_field)
        # Propagate
        g_enc = self.encoder(skeleton_graph, out_node_field=out_node_field, out_edge_field=out_edge_field)
        return g_enc