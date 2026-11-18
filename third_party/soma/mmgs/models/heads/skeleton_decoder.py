import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops import QueryAndGroup, grouping_operation

from ..builder import HEADS
from .sim_head import SimHead
from mmgs.models.utils import FFN, Normalizer
from mmgs.datasets.utils import denormalize
# from mmgs.datasets.utils.hood_common import NodeType
from mmgs.models.losses import L2Loss
from mmgs.core import multi_apply

import numpy as np
from mmgs.datasets.utils import to_numpy_detach
from mmgs.models.utils.dgl_graph import VERT_ID, MESH_EDGE, CROSS_EDGE, MESH_OBJ_ID, DIRECT_FORCE_ID, density_activation
from mmgs.models.utils.deformation_gradient import DeformationGradient
from mmgs.models.utils.render import render_gaussian, PipelineParams
import cv2

@HEADS.register_module()
class SkeletonDecoder(SimHead):
    def __init__(self,
                 out_channels=4+3+4, # rot, scale, rot
                 in_channels=128,
                 dt=1/30,
                 add_residual=True, # since not the same size
                 init_cfg=None,
                 eps=1e-7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 pre_norm=False,
                 render_pipe_cfg=dict(white_bg=False, convert_SHs_python=True, compute_cov3D_python=False, debug=False),
                 scalar_max=5,
                 scalar_min=-5,
                 init_quant=1.0,
                 norm_volumn=True,
                 *args,
                 **kwargs):
        super(SkeletonDecoder, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dt = dt
        self.eps = eps
        self.loss_dim = (0, 3)
        self.pre_norm = pre_norm
        self.node_norm = build_norm_layer(norm_cfg, in_channels)[1] if pre_norm else nn.Identity()
        self.scalar_max = scalar_max
        self.scalar_min = scalar_min
        self.init_quant = init_quant
        self.norm_volumn = norm_volumn

        if self.out_channels <= 0:
            raise ValueError(
                f'num_classes={out_channels} must be a positive integer')

        self.dynamic_proj = FFN([in_channels, in_channels//2, in_channels//4, out_channels], final_act=False, act_cfg=act_cfg, add_residual=False)
        self.scalar_activation = torch.exp
        self.scalar_constrain = self.scalar_incompressible_constrain
        self.density_activation = density_activation()
    
    def scalar_incompressible_constrain(self, scalar):
        '''
            The scalar must be activated before this func
            Assume the density is constant: incompressible 
        '''
        assert scalar.shape[-1] == 3
        mult = torch.pow(torch.prod(scalar, dim=-1, keepdim=True), 1/3)
        normed_scalar = scalar / mult
        return normed_scalar

    def node_pred_pos_dg(self,
                        feature_field,
                        cur_state_field, pin_mask_field,
                        pos_field, apply_pin):
        def func(nodes):
            vert_emb = nodes.data[feature_field]
            feature = self.node_norm(vert_emb)
            # Pred deformation gradient
            pred_vel = self.dynamic_proj(feature)
            pin_mask = nodes.data[pin_mask_field]
            cur_pos = nodes.data[cur_state_field]
            pred_pos = cur_pos + pred_vel * self.dt
            if apply_pin:
                pred_pos = pred_pos * torch.logical_not(pin_mask) + nodes.data[pos_field] * pin_mask
            return {pos_field: pred_pos}
        return func
    
    def predict(self, skeleton_graph, apply_pin=False):
        skeleton_graph.apply_nodes(
            self.node_pred_pos_dg(
                'out_node',
                'cur_state', 'pin_mask',
                'pred_pos', apply_pin=apply_pin
            )
        )

        pred_pos    = skeleton_graph.ndata['pred_pos']
        # pred_dg     = skeleton_graph.ndata['pred_dg']
        # pred_dg_mat = skeleton_graph.ndata['pred_dg_mat']
        return pred_pos, skeleton_graph

    def pre_predict(self, skeleton_graph, **kwargs):
        pred_pos, skeleton_graph = self.predict(skeleton_graph)
        cluster_nids = skeleton_graph.ndata['cluster_nids']
        joint_nids = skeleton_graph.ndata['joint_nids']
        pred_cluster_pos = pred_pos[cluster_nids==1]
        pred_joints_pos = pred_pos[joint_nids==1]
        return pred_cluster_pos, pred_joints_pos, skeleton_graph

    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)
    
    def simple_test(self,
            img_list, gt_label, test_cfg=None, **kwargs):
        """Test without augmentation."""

        rst = dict()
        acc_dict = self.evaluate(
            [pi.permute(1,2,0) for pi in img_list], [gt.permute(1,2,0) for gt in gt_label],
            term_filter=['render'],
            **kwargs
        )
        rst.update(dict(acc=acc_dict))
        return rst

    
    def forward_train_skeleton(self, pred_cluster_pos, pred_joints_pos, input_graph, joint_ids, is_detach=True, **kwargs):
        losses = dict()
        #compute joint pos in hierarchical graph
        joint_ids_list = list(joint_ids.values())
        joint_pos = input_graph[0].ndata['cur_state'][joint_ids_list]

        cluster_pos = input_graph[-1].ndata['cur_state']
        if is_detach:
            joint_pos = joint_pos.detach().clone()
            cluster_pos = cluster_pos.detach().clone()

        pred_pos = torch.concat([pred_cluster_pos, pred_joints_pos], dim=0)
        hier_pos = torch.concat([cluster_pos, joint_pos], dim=0)

        #compute loss
        loss_skeleton = self.loss(pred_pos, hier_pos, term_filter=['skeleton'], **kwargs)
        losses.update(loss_skeleton)
        
        return losses

    def forward_train_regularize(self,
            pred_pos, pred_dg, base_graph, connect_graph, **kwargs):
        '''
            Both train and test
        '''
        losses = dict()
        # Compute the momentum
        anchor_pos = base_graph.ndata['anchor_next_state']
        relative_pos = pred_pos - anchor_pos
        density = self.density_activation(base_graph.ndata['attr'][:, 0:1])
        mass = density * base_graph.ndata['diag_volume']
        momentum = (relative_pos * mass).sum(dim=0, keepdim=True).clamp(-100.0, 100.0)
        assert not torch.any(torch.isnan(relative_pos))
        assert not torch.any(torch.isnan(density))
        assert not torch.any(torch.isnan(mass))
        assert not torch.any(torch.isnan(momentum))
        loss_momentum = self.loss(
            momentum, torch.zeros_like(momentum).to(momentum),
            term_filter=['momentum'], **kwargs)
        losses.update(loss_momentum)

        return losses