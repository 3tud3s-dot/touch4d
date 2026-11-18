from pickle import FALSE
import numpy as np

import torch
import torch.nn as nn

from mmcv.runner import BaseModule, ModuleList
from mmcv.ops import QueryAndGroup, grouping_operation

import dgl
import dgl.function as fn
from mmgs.core import multi_apply
from mmgs.utils import rotation_from_normals, vertex_normal_batched
# from mmgs.datasets.utils.data_format import NodeType
# from mmgs.models.heads.acc_decoder import density_activation

from functools import partial
from collections import OrderedDict

from ..builder import PREPROCESSOR


def density_activation():
    return torch.exp

def density_deactivation():
    return torch.log

RECEIVER_ID = 'receiver'
SENDER_ID = 'sender'

MESH_OBJ_ID = 'vert_mesh'
ATTR_ID = 'vert_attr'
DIRECT_FORCE_ID = 'vert_invisable'

VERT_ID = 'vertices'
MESH_EDGE = 'vert2vert'
CROSS_EDGE = 'cross_e'

FORCE_ID = 'forces'
FORCE_EDGE = 'force_e'
DISTRIBUTE_EDGE = 'tov_e'

# For compute only
CPT_EDGE_NEIGH = 'e_neigh'

REL_DELTA_L2 = [0, 1]
REL_EDGE_DIR = [1, 4]
REL_VEL_L2 = [4, 5]
REL_VEL_DIR = [5, 8]

# For hood graph
HOOD_HIE_V0 = 'vert_level0'
HOOD_HIE_V1 = 'vert_level1'
HOOD_HIE_V2 = 'vert_level2'
HOOD_HIE_E0 = 'edge_level0'
HOOD_HIE_E1 = 'edge_level1'
HOOD_HIE_E2 = 'edge_level2'

# For gs sim
P2C_EDGE_ID = 'p2c_mask'
C2P_EDGE_ID = 'c2p_mask'
POINT_VERT_ID = 'point_mask'
CLUSTER_VERT_ID = 'cluster_mask'


@PREPROCESSOR.register_module()
class GsDynamicDGLProcessor(object):
    def __init__(self, radius, group_cfg, eps=1e-7, **kwargs) -> None:
        self.radius = radius
        self.grouper = QueryAndGroup(**group_cfg)
        self.eps = eps
        self.graph_builder = BuildGsDGLGraph(**kwargs)

    def _preprocess(self,
                    dynamic_edges, prev_state, cur_state, template_state, attr, pin_mask=None):
        assert attr is not None
        node_wise_dict = {
            'attr': attr,
            'prev_state': prev_state,
            'cur_state': cur_state,
            'template_state': template_state,}
        if pin_mask is not None:
            node_wise_dict['pin_mask'] = pin_mask
        
        # Build Graph
        g = self.graph_builder.build_graph(
            node_wise_dict, dynamic_edges)
        return g

    def _dynamic_edges(self, receiver_pos, sender_pos, radius):
        '''
            return tuple of: (receiver, sender) -> (dst, src)
            if between garment and garment, this is dual directoin; since 0 will find 54, and 54 will find 0, thus dual edge. No need further duplicate edges.
        '''
        # Return: bs, 3, n_verts, sample_num; bs, n_verts, sample_num
        group_xyz_diff, group_idx = self.grouper(
            sender_pos.contiguous(),
            receiver_pos.contiguous())
        # bs, n_verts, sample_num, 3
        group_xyz_diff = group_xyz_diff.permute(0, 2, 3, 1)
        group_xyz_l2 = torch.sqrt(torch.sum(group_xyz_diff**2, dim=-1, keepdim=True))
        group_xyz_l2_mask = group_xyz_l2 < radius
        # n_verts, sample_num
        group_xyz_l2_mask = group_xyz_l2_mask.squeeze(0).squeeze(-1)
        group_idx = group_idx.squeeze(0)
        valid_idx = torch.where(group_xyz_l2_mask == True)
        valid_neighbor = group_idx[valid_idx].to(torch.int64)
        # n_edges, 2
        relation_pair = torch.unique(torch.stack([valid_idx[0], valid_neighbor], dim=-1), dim=0)
        # Remove self-loop
        relation_non_loop = torch.where(relation_pair[:, 0] != relation_pair[:, 1])
        relation_pair = relation_pair[relation_non_loop]
        return relation_pair
    
    def batch_preprocess(self,
                         prev_state, cur_state, template_state, attr=None, pin_mask=None):
        '''
            bs == 1 case
            prev_state: n_point, 3
            cur_state: n_point, 3
            attr: n_point, attr_dim
            pin_mask: n_point, 1
        '''
        # TODO (not high priority): May add external forces to make it more generalizable

        # Build the initial graph first
        # 1. Get dynamic edges
        ## Query balls, not guareented for dual directed, since some point may choose other balls
        dynamic_edges = self._dynamic_edges(cur_state.unsqueeze(0), cur_state.unsqueeze(0))
        # 2. Register the data field
        g = self._preprocess(dynamic_edges, prev_state, cur_state, template_state, attr, pin_mask)
        return g
    
@PREPROCESSOR.register_module()
class GsHieDynamicDGLProcessor(object):
    def __init__(self, radius, group_cfg, anchor_prefix='anchor_', eps=1e-7, **kwargs) -> None:
        self.radius = radius
        if isinstance(group_cfg, list):
            assert len(group_cfg) == len(radius) # layer = len(radius)+1, 1 is extra layer for manually fixed root. Thus, hie-layer is len(radius)
            self.grouper = [
                QueryAndGroup(**g_cfg) for g_cfg in group_cfg]
        else:
            assert isinstance(group_cfg, dict)
            self.grouper = QueryAndGroup(**group_cfg)
        self.eps = eps
        self.graph_builder = BuildGsDGLGraph(**kwargs)
        self.anchor_prefix = anchor_prefix
        self.density_activation = density_activation()
        self.density_deactivation = density_deactivation()

    def _preprocess(self,
                    prev_state, cur_state, template_state, attr, diag_volume, external_forces, pin_mask, dynamic_radius, grouper, dynamic_base):

        if dynamic_base:
            assert dynamic_base is False, "Need debug"
            static_edges = self._dynamic_edges(template_state.unsqueeze(0), template_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper)
            dynamic_edges = self._dynamic_edges(cur_state.unsqueeze(0), cur_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=static_edges)
            dynamic_edges = torch.cat([static_edges, dynamic_edges], dim=0)
        else:
            dynamic_edges = None

        assert attr is not None
        n_nodes = prev_state.shape[0]
        if external_forces.shape[0] == 1:
            in_external = external_forces.expand(n_nodes, -1)
        else:
            assert external_forces.shape[0] == n_nodes
            in_external = external_forces
        node_wise_dict = {
            'attr': attr,
            'diag_volume': diag_volume,
            'prev_state': prev_state,
            'cur_state': cur_state,
            'template_state': template_state,
            'orig_template_state': template_state.detach().clone(),
            'pin_mask': pin_mask,
            'external': in_external}
        
        # Build Graph
        g = self.graph_builder.base_graph(
            node_wise_dict, dynamic_edges)
        return g

    def _preprocess_hierarchy(self, prev_graph, connect_edges, node_attr_scheme, dynamic_radius, grouper):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]
        n_clusters = torch.max(connect_edges)+1

        # Move the offset
        offseted_connect_edges = connect_edges+n_points
        # n_edges, 2; dst, src
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        # Build the connection graph first
        ## Node first, cluster next
        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        # p2c_edge_mask = torch.zeros((p2c_edge_pair.shape[0]*2, 1))
        # p2c_edge_mask[:p2c_edge_pair.shape[0]] = 1.0
        # connect_edge_wise_dict = {
        #     P2C_EDGE_ID: p2c_edge_mask,
        #     C2P_EDGE_ID: 1-p2c_edge_mask,}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)
        # Transfer the data
        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        p2c_eids = torch.nonzero(connect_graph.edata[P2C_EDGE_ID][:, 0], as_tuple=False).squeeze()
        ## Copying, broadcasting, extracting
        hie_node_wise_dict = dict()
        for key, mode in node_attr_scheme.items():
            val = prev_graph.ndata[key]
            connect_graph.nodes[point_nids].data[key] = val
            if key == 'attr':
                density_key = 'density'
                volumn_key = 'diag_volume'
                # Specifically deal with this
                # Get mean first
                connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                # Calculate the density and merge
                connect_graph.apply_nodes(lambda x: {density_key: self.density_activation(x.data['attr'][:, :1])})
                connect_graph.send_and_recv(p2c_eids, lambda x: {density_key: x.src[density_key]*x.src[volumn_key]/x.dst[volumn_key]}, fn.sum(density_key, density_key))
                connect_graph.apply_nodes(lambda x: {key: torch.cat([self.density_deactivation(x.data[density_key]), x.data[key][:, 1:]], dim=-1)}, cluster_nids)
            elif isinstance(mode, dict):
                # weighted measure
                weighted_key = mode['key']
                assert mode['reduct'] == 'mean', "Currently only support mean"
                assert weighted_key in connect_graph.ndata.keys()
                connect_graph.send_and_recv(p2c_eids, lambda x: {key: x.src[key]*x.src[weighted_key]}, fn.sum(key, key))
                if mode['reduct'] == 'mean':
                    connect_graph.nodes[cluster_nids].data[key] = connect_graph.nodes[cluster_nids].data[key] / connect_graph.nodes[cluster_nids].data[weighted_key]
            else:
                assert isinstance(mode, str)
                if mode == 'mean':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                elif mode == 'or':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.max(key, key))
                elif mode == 'sum':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.sum(key, key))
                else:
                    assert False, "Not support yet"
            hie_node_wise_dict[key] = connect_graph.nodes[cluster_nids].data[key]
        # Build hierarchical graph
        ## Static edges
        ### n_edges, 2
        hie_static_edges = self._dynamic_edges(hie_node_wise_dict['template_state'].unsqueeze(0), hie_node_wise_dict['template_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper)
        ## Dynamic edges
        hie_dynamic_edges = self._dynamic_edges(hie_node_wise_dict['cur_state'].unsqueeze(0), hie_node_wise_dict['cur_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=hie_static_edges)
        hie_edges = torch.cat([hie_static_edges, hie_dynamic_edges], dim=0)
        ## Build graph
        hie_g = self.graph_builder.base_graph(
            hie_node_wise_dict, hie_edges)
        return hie_g, connect_graph

    def _preprocess_abs_hierarchy(self, prev_graph, connect_edges, pinned_node_wise_dict):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]
        n_clusters = torch.max(connect_edges)+1

        # Move the offset
        offseted_connect_edges = connect_edges+n_points
        # n_edges, 2; dst, src
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        # Build the connection graph first
        ## Node first, cluster next
        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)

        # Different from here <<< 
        # Transfer the data
        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        ## Copying, broadcasting, extracting
        for key in pinned_node_wise_dict.keys():
            #  Very weird here, previously need int32, here int64
            connect_graph.nodes[point_nids.long()].data[key] = prev_graph.ndata[key]
            # for key, val in pinned_node_wise_dict.items():
            connect_graph.nodes[cluster_nids.long()].data[key] = pinned_node_wise_dict[key]
        return connect_graph

    def _dynamic_edges(self, receiver_pos, sender_pos, radius, grouper, non_intersect_relations=None):
        '''
            return tuple of: (receiver, sender) -> (dst, src)
            if between garment and garment, this is dual directoin; since 0 will find 54, and 54 will find 0, thus dual edge. No need further duplicate edges.
        '''
        # Return: bs, 3, n_verts, sample_num; bs, n_verts, sample_num
        group_xyz_diff, group_idx = grouper(
            sender_pos.contiguous(),
            receiver_pos.contiguous())
        # bs, n_verts, sample_num, 3
        group_xyz_diff = group_xyz_diff.permute(0, 2, 3, 1)
        group_xyz_l2 = torch.sqrt(torch.sum(group_xyz_diff**2, dim=-1, keepdim=True))
        group_xyz_l2_mask = group_xyz_l2 < radius
        # n_verts, sample_num
        group_xyz_l2_mask = group_xyz_l2_mask.squeeze(0).squeeze(-1)
        group_idx = group_idx.squeeze(0)
        valid_idx = torch.where(group_xyz_l2_mask == True)
        valid_neighbor = group_idx[valid_idx].to(torch.int64)
        # n_edges, 2
        relation_pair = torch.unique(torch.stack([valid_idx[0], valid_neighbor], dim=-1), dim=0)
        # Remove self-loop
        relation_non_loop = torch.where(relation_pair[:, 0] != relation_pair[:, 1])
        relation_pair = relation_pair[relation_non_loop]
        # Here may occur no edges:
        # Clear unique pairs
        if non_intersect_relations is not None:
            # e_w+e_m, 2
            candidates = torch.cat([relation_pair.transpose(-1, -2), non_intersect_relations.transpose(-1, -2), non_intersect_relations.transpose(-1, -2)], dim=-1).transpose(-1, -2)
            r_uniques, r_counts = torch.unique(candidates, dim=0, return_counts=True)
            r_pairs = r_uniques[r_counts == 1]
            relation_pair = r_pairs
        return relation_pair
    
    def _postprocess_hie(self, base_g, connect_g, forward_keys, prefix='anchor_'):
        point_nids = torch.nonzero(connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        # cluster_nids = torch.nonzero(connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        c2p_eids = torch.nonzero(connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
        for key in forward_keys:
            connect_g.send_and_recv(c2p_eids, fn.copy_u(key, key), fn.mean(key, prefix+key))
            base_g.ndata[prefix+key] = connect_g.nodes[point_nids].data[prefix+key]
        return base_g, connect_g

    def batch_preprocess(self,
                         prev_state, cur_state, template_state, attr, diag_volume, cur_cov, external_forces,
                         p2c_mapping=None, pin_mask=None, dynamic_base=True):
        '''
            bs == 1 case
            prev_state: n_point, 3
            cur_state: n_point, 3
            attr: n_point, attr_dim
            pin_mask: n_point, 1
            p2c_mapping: last one is the relation towards the pinned verts
        '''
        # TODO (not high priority): May add external forces to make it more generalizable
        assert len(p2c_mapping) == len(self.radius)
        graph_list = []
        connect_graph_list = []
        # Build the initial graph first
        base_g = self._preprocess(
            prev_state, cur_state, template_state, attr, diag_volume, external_forces, pin_mask,
            dynamic_radius=self.radius[0],
            grouper=self.grouper[0] if isinstance(self.grouper, list) else self.grouper,
            dynamic_base=dynamic_base)
        graph_list.append(base_g)

        node_attr_scheme = OrderedDict(
            # Static
            diag_volume='sum',
            attr='mean', # TODO: Only for selfsup loss. For now no need to correct. Check the mass. the mass is wrong cuz the assumtion is the density is constant by here density is vertex-wise.
            pin_mask='or',
            external='mean', # TODO: Here is wrong. Should be sum and multiply the mass and then divide by the total mass. Here external is acceleration as gravity.
            # Dynamic
            prev_state=dict(reduct='mean', key='diag_volume'),
            cur_state=dict(reduct='mean', key='diag_volume'),
            template_state=dict(reduct='mean', key='diag_volume'),
            orig_template_state=dict(reduct='mean', key='diag_volume'),
        )

        # hierarchical connect (from detail-level (particle) to coarse-level (cluster) )
        for cluster_idx in range(len(p2c_mapping)-1):
            p2c_map = p2c_mapping[cluster_idx]
            # Build connection graph
            connect_edges = p2c_map.long()
            prev_graph = graph_list[-1]
            hie_g, connect_g = self._preprocess_hierarchy(prev_graph, connect_edges, node_attr_scheme, dynamic_radius=self.radius[cluster_idx+1], grouper=self.grouper[cluster_idx+1] if isinstance(self.grouper, list) else self.grouper)
            connect_graph_list.append(connect_g)
            graph_list.append(hie_g)
        
        # Append a connect graph for pinned verts interact
        pinned_nodes = base_g.nodes[torch.where(pin_mask.bool()[:, 0])[0]]
        pinned_node_wise_dict = {
            key: val
            for key, val in pinned_nodes.data.items()}
        pin_connect_g = self._preprocess_abs_hierarchy(graph_list[-1], p2c_mapping[-1], pinned_node_wise_dict)
        connect_graph_list.append(pin_connect_g)
        # pinned_nids = torch.where(pinned_mask > 0)

        # Post process
        hie_forward_keys = [
            'prev_state',
            'cur_state',
            'template_state',
            'orig_template_state',]
        for i in range(len(graph_list)):
            # Broadcast the anchor's info to each point
            cur_base_g, cur_connect_g = graph_list[i], connect_graph_list[i]
            update_base_g, update_connect_g = self._postprocess_hie(cur_base_g, cur_connect_g, hie_forward_keys, prefix=self.anchor_prefix)
            graph_list[i], connect_graph_list[i] = update_base_g, update_connect_g

        # Append the cov for only the pc
        graph_list[0].ndata['cur_cov'] = cur_cov
        return graph_list, connect_graph_list


@PREPROCESSOR.register_module()
class GsHieEmbodiedDGLProcessor(object):
    def __init__(self, radius, group_cfg, anchor_prefix='anchor_', eps=1e-7, **kwargs) -> None:
        self.radius = radius + [radius[-1]*10]
        if isinstance(group_cfg, list):
            assert len(group_cfg) == len(radius) # layer = len(radius)+1, 1 is extra layer for manually fixed root. Thus, hie-layer is len(radius)
            self.grouper = [
                QueryAndGroup(**g_cfg) for g_cfg in group_cfg]
        else:
            assert isinstance(group_cfg, dict)
            self.grouper = QueryAndGroup(**group_cfg)
        self.eps = eps
        self.graph_builder = BuildGsDGLGraph(**kwargs)
        self.anchor_prefix = anchor_prefix
        self.density_activation = density_activation()
        self.density_deactivation = density_deactivation()

    def _preprocess(self,
                    prev_state, cur_state, template_state, attr, diag_volume, external_forces, pin_mask, dynamic_radius, grouper, dynamic_base):

        if dynamic_base:
            assert dynamic_base is False, "Need debug"
            static_edges = self._dynamic_edges(template_state.unsqueeze(0), template_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper)
            dynamic_edges = self._dynamic_edges(cur_state.unsqueeze(0), cur_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=static_edges)
            dynamic_edges = torch.cat([static_edges, dynamic_edges], dim=0)
        else:
            dynamic_edges = None

        assert attr is not None
        n_nodes = prev_state.shape[0]
        if external_forces.shape[0] == 1:
            in_external = external_forces.expand(n_nodes, -1)
        else:
            assert external_forces.shape[0] == n_nodes
            in_external = external_forces
        node_wise_dict = {
            'attr': attr,
            'diag_volume': diag_volume,
            'prev_state': prev_state,
            'cur_state': cur_state,
            'template_state': template_state,
            'orig_template_state': template_state.detach().clone(),
            'pin_mask': pin_mask,
            'external': in_external}
        
        # Build Graph
        g = self.graph_builder.base_graph(
            node_wise_dict, dynamic_edges)
        return g

    def _preprocess_hierarchy(self, prev_graph, connect_edges, node_attr_scheme, dynamic_radius, grouper):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]
        n_clusters = torch.max(connect_edges)+1

        # Move the offset
        offseted_connect_edges = connect_edges+n_points
        # n_edges, 2; dst, src
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        # Build the connection graph first
        ## Node first, cluster next
        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        # p2c_edge_mask = torch.zeros((p2c_edge_pair.shape[0]*2, 1))
        # p2c_edge_mask[:p2c_edge_pair.shape[0]] = 1.0
        # connect_edge_wise_dict = {
        #     P2C_EDGE_ID: p2c_edge_mask,
        #     C2P_EDGE_ID: 1-p2c_edge_mask,}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)
        # Transfer the data
        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        p2c_eids = torch.nonzero(connect_graph.edata[P2C_EDGE_ID][:, 0], as_tuple=False).squeeze()
        ## Copying, broadcasting, extracting
        hie_node_wise_dict = dict()
        for key, mode in node_attr_scheme.items():
            val = prev_graph.ndata[key]
            connect_graph.nodes[point_nids].data[key] = val
            if key == 'attr':
                density_key = 'density'
                volumn_key = 'diag_volume'
                # Specifically deal with this
                # Get mean first
                connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                # Calculate the density and merge
                connect_graph.apply_nodes(lambda x: {density_key: self.density_activation(x.data['attr'][:, :1])})
                connect_graph.send_and_recv(p2c_eids, lambda x: {density_key: x.src[density_key]*x.src[volumn_key]/x.dst[volumn_key]}, fn.sum(density_key, density_key))
                connect_graph.apply_nodes(lambda x: {key: torch.cat([self.density_deactivation(x.data[density_key]), x.data[key][:, 1:]], dim=-1)}, cluster_nids)
            elif isinstance(mode, dict):
                # weighted measure
                weighted_key = mode['key']
                assert mode['reduct'] == 'mean', "Currently only support mean"
                assert weighted_key in connect_graph.ndata.keys()
                connect_graph.send_and_recv(p2c_eids, lambda x: {key: x.src[key]*x.src[weighted_key]}, fn.sum(key, key))
                if mode['reduct'] == 'mean':
                    connect_graph.nodes[cluster_nids].data[key] = connect_graph.nodes[cluster_nids].data[key] / connect_graph.nodes[cluster_nids].data[weighted_key]
            else:
                assert isinstance(mode, str)
                if mode == 'mean':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                elif mode == 'or':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.max(key, key))
                elif mode == 'sum':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.sum(key, key))
                else:
                    assert False, "Not support yet"
            hie_node_wise_dict[key] = connect_graph.nodes[cluster_nids].data[key]
        # Build hierarchical graph
        ## Static edges
        ### n_edges, 2
        hie_static_edges = self._dynamic_edges(hie_node_wise_dict['template_state'].unsqueeze(0), hie_node_wise_dict['template_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper)
        ## Dynamic edges
        hie_dynamic_edges = self._dynamic_edges(hie_node_wise_dict['cur_state'].unsqueeze(0), hie_node_wise_dict['cur_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=hie_static_edges)
        hie_edges = torch.cat([hie_static_edges, hie_dynamic_edges], dim=0)
        ## Build graph
        hie_g = self.graph_builder.base_graph(
            hie_node_wise_dict, hie_edges)
        return hie_g, connect_graph

    def _preprocess_abs_hierarchy(self, prev_graph, connect_edges, pinned_node_wise_dict):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]   
        n_clusters = torch.max(connect_edges)+1

        # Move the offset
        offseted_connect_edges = connect_edges+n_points
        # n_edges, 2; dst, src
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        # Build the connection graph first
        ## Node first, cluster next
        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)

        # Different from here <<< 
        # Transfer the data
        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        ## Copying, broadcasting, extracting
        for key in pinned_node_wise_dict.keys():
            #  Very weird here, previously need int32, here int64
            connect_graph.nodes[point_nids.long()].data[key] = prev_graph.ndata[key]
            # for key, val in pinned_node_wise_dict.items():
            connect_graph.nodes[cluster_nids.long()].data[key] = pinned_node_wise_dict[key]
        return connect_graph

    def _dynamic_edges(self, receiver_pos, sender_pos, radius, grouper, non_intersect_relations=None):
        '''
            return tuple of: (receiver, sender) -> (dst, src)
            if between garment and garment, this is dual directoin; since 0 will find 54, and 54 will find 0, thus dual edge. No need further duplicate edges.
        '''
        # Return: bs, 3, n_verts, sample_num; bs, n_verts, sample_num
        group_xyz_diff, group_idx = grouper(
            sender_pos.contiguous(),
            receiver_pos.contiguous())
        # bs, n_verts, sample_num, 3
        group_xyz_diff = group_xyz_diff.permute(0, 2, 3, 1)
        group_xyz_l2 = torch.sqrt(torch.sum(group_xyz_diff**2, dim=-1, keepdim=True))
        group_xyz_l2_mask = group_xyz_l2 < radius
        # n_verts, sample_num
        group_xyz_l2_mask = group_xyz_l2_mask.squeeze(0).squeeze(-1)
        group_idx = group_idx.squeeze(0)
        valid_idx = torch.where(group_xyz_l2_mask == True)
        valid_neighbor = group_idx[valid_idx].to(torch.int64)
        # n_edges, 2
        relation_pair = torch.unique(torch.stack([valid_idx[0], valid_neighbor], dim=-1), dim=0)
        # Remove self-loop
        relation_non_loop = torch.where(relation_pair[:, 0] != relation_pair[:, 1])
        relation_pair = relation_pair[relation_non_loop]
        # Here may occur no edges:
        # Clear unique pairs
        if non_intersect_relations is not None:
            # e_w+e_m, 2
            candidates = torch.cat([relation_pair.transpose(-1, -2), non_intersect_relations.transpose(-1, -2), non_intersect_relations.transpose(-1, -2)], dim=-1).transpose(-1, -2)
            r_uniques, r_counts = torch.unique(candidates, dim=0, return_counts=True)
            r_pairs = r_uniques[r_counts == 1]
            relation_pair = r_pairs
        return relation_pair
    
    def _postprocess_hie(self, base_g, connect_g, forward_keys, prefix='anchor_'):
        point_nids = torch.nonzero(connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        # cluster_nids = torch.nonzero(connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        c2p_eids = torch.nonzero(connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
        for key in forward_keys:
            connect_g.send_and_recv(c2p_eids, fn.copy_u(key, key), fn.mean(key, prefix+key))
            base_g.ndata[prefix+key] = connect_g.nodes[point_nids].data[prefix+key]
        return base_g, connect_g

    def batch_preprocess(self,
                         prev_state, cur_state, template_state, attr, diag_volume, cur_cov, external_forces,
                         p2c_mapping=None, pin_mask=None, dynamic_base=True):
        '''
            bs == 1 case
            prev_state: n_point, 3
            cur_state: n_point, 3
            attr: n_point, attr_dim
            pin_mask: n_point, 1
            p2c_mapping: last one is the relation towards the pinned verts
        '''
        # TODO (not high priority): May add external forces to make it more generalizable
        assert len(p2c_mapping) == len(self.radius)
        graph_list = []
        connect_graph_list = []
        # Build the initial graph first
        base_g = self._preprocess(
            prev_state, cur_state, template_state, attr, diag_volume, external_forces, pin_mask,
            dynamic_radius=self.radius[0],
            grouper=self.grouper[0] if isinstance(self.grouper, list) else self.grouper,
            dynamic_base=dynamic_base)
        graph_list.append(base_g)

        node_attr_scheme = OrderedDict(
            # Static
            diag_volume='sum',
            attr='mean', # TODO: Only for selfsup loss. For now no need to correct. Check the mass. the mass is wrong cuz the assumtion is the density is constant by here density is vertex-wise.
            pin_mask='or',
            external='mean', # TODO: Here is wrong. Should be sum and multiply the mass and then divide by the total mass. Here external is acceleration as gravity.
            # Dynamic
            prev_state=dict(reduct='mean', key='diag_volume'),
            cur_state=dict(reduct='mean', key='diag_volume'),
            template_state=dict(reduct='mean', key='diag_volume'),
            orig_template_state=dict(reduct='mean', key='diag_volume'),
        )

        # hierarchical connect (from detail-level (particle) to coarse-level (cluster) )
        for cluster_idx in range(len(p2c_mapping)-1):
            p2c_map = p2c_mapping[cluster_idx]
            # Build connection graph
            connect_edges = p2c_map.long()
            prev_graph = graph_list[-1]
            hie_g, connect_g = self._preprocess_hierarchy(prev_graph, connect_edges, node_attr_scheme, dynamic_radius=self.radius[cluster_idx+1], grouper=self.grouper[cluster_idx+1] if isinstance(self.grouper, list) else self.grouper)
            connect_graph_list.append(connect_g)
            graph_list.append(hie_g)
        
        # # Append a connect graph for pinned verts interact
        # pinned_nodes = base_g.nodes[torch.where(pin_mask.bool()[:, 0])[0]]
        # pinned_node_wise_dict = {
        #     key: val
        #     for key, val in pinned_nodes.data.items()}
        # pin_connect_g = self._preprocess_abs_hierarchy(graph_list[-1], p2c_mapping[-1], pinned_node_wise_dict)
        # connect_graph_list.append(pin_connect_g)
        # # pinned_nids = torch.where(pinned_mask > 0)

        # Post process
        hie_forward_keys = [
            'prev_state',
            'cur_state',
            'template_state',
            'orig_template_state',]
        for i in range(len(graph_list)-1):
            # Broadcast the anchor's info to each point
            cur_base_g, cur_connect_g = graph_list[i], connect_graph_list[i]
            update_base_g, update_connect_g = self._postprocess_hie(cur_base_g, cur_connect_g, hie_forward_keys, prefix=self.anchor_prefix)
            graph_list[i], connect_graph_list[i] = update_base_g, update_connect_g

        # Append the cov for only the pc
        graph_list[0].ndata['cur_cov'] = cur_cov
        return graph_list, connect_graph_list

@PREPROCESSOR.register_module()
class GsHieAlignDGLProcessor(object):
    def __init__(self, radius, group_cfg, anchor_prefix='anchor_', eps=1e-7, **kwargs) -> None:
        self.radius = radius + [radius[-1]*10]
        if isinstance(group_cfg, list):
            assert len(group_cfg) == len(radius) # layer = len(radius)+1, 1 is extra layer for manually fixed root. Thus, hie-layer is len(radius)
            self.grouper = [
                QueryAndGroup(**g_cfg) for g_cfg in group_cfg]
        else:
            assert isinstance(group_cfg, dict)
            self.grouper = QueryAndGroup(**group_cfg)
        self.eps = eps
        self.graph_builder = BuildGsDGLGraph(**kwargs)
        self.anchor_prefix = anchor_prefix
        self.density_activation = density_activation()
        self.density_deactivation = density_deactivation()

    def _preprocess(self,
                    cur_state, template_state, attr, diag_volume, dynamic_radius, grouper, dynamic_base):

        if dynamic_base:
            assert dynamic_base is False, "Need debug"
            static_edges = self._dynamic_edges(template_state.unsqueeze(0), template_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper)
            dynamic_edges = self._dynamic_edges(cur_state.unsqueeze(0), cur_state.unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=static_edges)
            dynamic_edges = torch.cat([static_edges, dynamic_edges], dim=0)
        else:
            dynamic_edges = None

        assert attr is not None
        node_wise_dict = {
            'attr': attr,
            'diag_volume': diag_volume,
            'cur_state': cur_state,
            'template_state': template_state,
            'orig_template_state': template_state.detach().clone()
            }
        
        # Build Graph
        g = self.graph_builder.base_graph(
            node_wise_dict, dynamic_edges)
        return g

    def _preprocess_hierarchy(self, prev_graph, connect_edges, node_attr_scheme, dynamic_radius, grouper):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]
        n_clusters = torch.max(connect_edges)+1

        # Move the offset
        offseted_connect_edges = connect_edges+n_points
        # n_edges, 2; dst, src
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        # Build the connection graph first
        ## Node first, cluster next
        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        # p2c_edge_mask = torch.zeros((p2c_edge_pair.shape[0]*2, 1))
        # p2c_edge_mask[:p2c_edge_pair.shape[0]] = 1.0
        # connect_edge_wise_dict = {
        #     P2C_EDGE_ID: p2c_edge_mask,
        #     C2P_EDGE_ID: 1-p2c_edge_mask,}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)
        # Transfer the data
        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        p2c_eids = torch.nonzero(connect_graph.edata[P2C_EDGE_ID][:, 0], as_tuple=False).squeeze()
        ## Copying, broadcasting, extracting
        hie_node_wise_dict = dict()
        for key, mode in node_attr_scheme.items():
            val = prev_graph.ndata[key]
            connect_graph.nodes[point_nids].data[key] = val
            if key == 'attr':
                density_key = 'density'
                volumn_key = 'diag_volume'
                # Specifically deal with this
                # Get mean first
                connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                # Calculate the density and merge
                connect_graph.apply_nodes(lambda x: {density_key: self.density_activation(x.data['attr'][:, :1])})
                connect_graph.send_and_recv(p2c_eids, lambda x: {density_key: x.src[density_key]*x.src[volumn_key]/x.dst[volumn_key]}, fn.sum(density_key, density_key))
                connect_graph.apply_nodes(lambda x: {key: torch.cat([self.density_deactivation(x.data[density_key]), x.data[key][:, 1:]], dim=-1)}, cluster_nids)
            elif isinstance(mode, dict):
                # weighted measure
                weighted_key = mode['key']
                assert mode['reduct'] == 'mean', "Currently only support mean"
                assert weighted_key in connect_graph.ndata.keys()
                connect_graph.send_and_recv(p2c_eids, lambda x: {key: x.src[key]*x.src[weighted_key]}, fn.sum(key, key))
                if mode['reduct'] == 'mean':
                    connect_graph.nodes[cluster_nids].data[key] = connect_graph.nodes[cluster_nids].data[key] / connect_graph.nodes[cluster_nids].data[weighted_key]
            else:
                assert isinstance(mode, str)
                if mode == 'mean':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.mean(key, key))
                elif mode == 'or':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.max(key, key))
                elif mode == 'sum':
                    connect_graph.send_and_recv(p2c_eids, fn.copy_u(key, key), fn.sum(key, key))
                else:
                    assert False, "Not support yet"
            hie_node_wise_dict[key] = connect_graph.nodes[cluster_nids].data[key]
        # Build hierarchical graph
        ## Static edges
        ### n_edges, 2
        hie_static_edges = self._dynamic_edges(hie_node_wise_dict['template_state'].unsqueeze(0), hie_node_wise_dict['template_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper)
        ## Dynamic edges
        hie_dynamic_edges = self._dynamic_edges(hie_node_wise_dict['cur_state'].unsqueeze(0), hie_node_wise_dict['cur_state'].unsqueeze(0), radius=dynamic_radius, grouper=grouper, non_intersect_relations=hie_static_edges)
        hie_edges = torch.cat([hie_static_edges, hie_dynamic_edges], dim=0)
        ## Build graph
        hie_g = self.graph_builder.base_graph(
            hie_node_wise_dict, hie_edges)
        return hie_g, connect_graph

    def _preprocess_abs_hierarchy(self, prev_graph, connect_edges, pinned_node_wise_dict):
        n_points = prev_graph.num_nodes()
        assert n_points == connect_edges.shape[0]   
        n_clusters = torch.max(connect_edges)+1

        # Move the offset
        offseted_connect_edges = connect_edges+n_points
        # n_edges, 2; dst, src
        p2c_edge_pair = torch.stack([offseted_connect_edges, torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges)], dim=-1)
        c2p_edge_pair = torch.stack([torch.arange(offseted_connect_edges.shape[0]).to(offseted_connect_edges), offseted_connect_edges], dim=-1)

        # Build the connection graph first
        ## Node first, cluster next
        point_vert_mask = torch.zeros((n_points+n_clusters, 1)).float().to(offseted_connect_edges.device)
        point_vert_mask[:n_points] = 1.0
        connect_node_wise_dict = {
            POINT_VERT_ID: point_vert_mask,
            CLUSTER_VERT_ID: 1-point_vert_mask}
        connect_graph = self.graph_builder.connect_graph(n_points, n_clusters, p2c_edge_pair, c2p_edge_pair, connect_node_wise_dict)

        # Different from here <<< 
        # Transfer the data
        point_nids = torch.nonzero(connect_graph.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        cluster_nids = torch.nonzero(connect_graph.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        ## Copying, broadcasting, extracting
        for key in pinned_node_wise_dict.keys():
            #  Very weird here, previously need int32, here int64
            connect_graph.nodes[point_nids.long()].data[key] = prev_graph.ndata[key]
            # for key, val in pinned_node_wise_dict.items():
            connect_graph.nodes[cluster_nids.long()].data[key] = pinned_node_wise_dict[key]
        return connect_graph

    def _dynamic_edges(self, receiver_pos, sender_pos, radius, grouper, non_intersect_relations=None):
        '''
            return tuple of: (receiver, sender) -> (dst, src)
            if between garment and garment, this is dual directoin; since 0 will find 54, and 54 will find 0, thus dual edge. No need further duplicate edges.
        '''
        # Return: bs, 3, n_verts, sample_num; bs, n_verts, sample_num
        group_xyz_diff, group_idx = grouper(
            sender_pos.contiguous(),
            receiver_pos.contiguous())
        # bs, n_verts, sample_num, 3
        group_xyz_diff = group_xyz_diff.permute(0, 2, 3, 1)
        group_xyz_l2 = torch.sqrt(torch.sum(group_xyz_diff**2, dim=-1, keepdim=True))
        group_xyz_l2_mask = group_xyz_l2 < radius
        # n_verts, sample_num
        group_xyz_l2_mask = group_xyz_l2_mask.squeeze(0).squeeze(-1)
        group_idx = group_idx.squeeze(0)
        valid_idx = torch.where(group_xyz_l2_mask == True)
        valid_neighbor = group_idx[valid_idx].to(torch.int64)
        # n_edges, 2
        relation_pair = torch.unique(torch.stack([valid_idx[0], valid_neighbor], dim=-1), dim=0)
        # Remove self-loop
        relation_non_loop = torch.where(relation_pair[:, 0] != relation_pair[:, 1])
        relation_pair = relation_pair[relation_non_loop]
        # Here may occur no edges:
        # Clear unique pairs
        if non_intersect_relations is not None:
            # e_w+e_m, 2
            candidates = torch.cat([relation_pair.transpose(-1, -2), non_intersect_relations.transpose(-1, -2), non_intersect_relations.transpose(-1, -2)], dim=-1).transpose(-1, -2)
            r_uniques, r_counts = torch.unique(candidates, dim=0, return_counts=True)
            r_pairs = r_uniques[r_counts == 1]
            relation_pair = r_pairs
        return relation_pair
    
    def _postprocess_hie(self, base_g, connect_g, forward_keys, prefix='anchor_'):
        point_nids = torch.nonzero(connect_g.ndata[POINT_VERT_ID][:, 0], as_tuple=False).squeeze()
        # cluster_nids = torch.nonzero(connect_g.ndata[CLUSTER_VERT_ID][:, 0], as_tuple=False).squeeze()
        c2p_eids = torch.nonzero(connect_g.edata[C2P_EDGE_ID][:, 0], as_tuple=False).squeeze()
        for key in forward_keys:
            connect_g.send_and_recv(c2p_eids, fn.copy_u(key, key), fn.mean(key, prefix+key))
            base_g.ndata[prefix+key] = connect_g.nodes[point_nids].data[prefix+key]
        return base_g, connect_g

    def batch_preprocess(self,
                        cur_state, template_state, attr, diag_volume, cur_cov,
                         p2c_mapping=None, dynamic_base=False):
        '''
            bs == 1 case
            prev_state: n_point, 3
            cur_state: n_point, 3
            attr: n_point, attr_dim
            pin_mask: n_point, 1
            p2c_mapping: last one is the relation towards the pinned verts
        '''
        # TODO (not high priority): May add external forces to make it more generalizable
        assert len(p2c_mapping) == len(self.radius)
        graph_list = []
        connect_graph_list = []
        # Build the initial graph first
        base_g = self._preprocess(
            cur_state, template_state, attr, diag_volume,
            dynamic_radius=self.radius[0],
            grouper=self.grouper[0] if isinstance(self.grouper, list) else self.grouper,
            dynamic_base=dynamic_base)
        graph_list.append(base_g)

        node_attr_scheme = OrderedDict(
            # Static
            diag_volume='sum',
            attr='mean', # TODO: Only for selfsup loss. For now no need to correct. Check the mass. the mass is wrong cuz the assumtion is the density is constant by here density is vertex-wise.
            # pin_mask='or',
            # external='mean', # TODO: Here is wrong. Should be sum and multiply the mass and then divide by the total mass. Here external is acceleration as gravity.
            # Dynamic
            # prev_state=dict(reduct='mean', key='diag_volume'),
            cur_state=dict(reduct='mean', key='diag_volume'),
            template_state=dict(reduct='mean', key='diag_volume'),
            orig_template_state=dict(reduct='mean', key='diag_volume'),
        )

        # hierarchical connect (from detail-level (particle) to coarse-level (cluster) )
        for cluster_idx in range(len(p2c_mapping)-1):
            p2c_map = p2c_mapping[cluster_idx]
            # Build connection graph
            connect_edges = p2c_map.long()
            prev_graph = graph_list[-1]
            hie_g, connect_g = self._preprocess_hierarchy(prev_graph, connect_edges, node_attr_scheme, dynamic_radius=self.radius[cluster_idx+1], grouper=self.grouper[cluster_idx+1] if isinstance(self.grouper, list) else self.grouper)
            connect_graph_list.append(connect_g)
            graph_list.append(hie_g)
        
        # # Append a connect graph for pinned verts interact
        # pinned_nodes = base_g.nodes[torch.where(pin_mask.bool()[:, 0])[0]]
        # pinned_node_wise_dict = {
        #     key: val
        #     for key, val in pinned_nodes.data.items()}
        # pin_connect_g = self._preprocess_abs_hierarchy(graph_list[-1], p2c_mapping[-1], pinned_node_wise_dict)
        # connect_graph_list.append(pin_connect_g)
        # # pinned_nids = torch.where(pinned_mask > 0)

        # Post process
        hie_forward_keys = [
            'cur_state',
            'template_state',
            'orig_template_state',]
        for i in range(len(graph_list)-1):
            # Broadcast the anchor's info to each point
            cur_base_g, cur_connect_g = graph_list[i], connect_graph_list[i]
            update_base_g, update_connect_g = self._postprocess_hie(cur_base_g, cur_connect_g, hie_forward_keys, prefix=self.anchor_prefix)
            graph_list[i], connect_graph_list[i] = update_base_g, update_connect_g

        # Append the cov for only the pc
        graph_list[0].ndata['cur_cov'] = cur_cov
        return graph_list, connect_graph_list

class BuildGsDGLGraph:
    def __init__(self, receiver_id=None, sender_id=None) -> None:
        self.receiver_id = VERT_ID
        self.sender_id = VERT_ID
        if receiver_id is not None:
            self.receiver_id = receiver_id
        if sender_id is not None:
            self.sender_id = sender_id

    def build_graph(self,
            node_wise_dict, dynamic_edges):
        '''
            Node order: garment verts; garment patches; human verts;
        '''
        # Static graph
        base_g = self.base_graph(node_wise_dict, dynamic_edges)
        return base_g

    def base_graph(self, node_wise_dict, dynamic_edges):
        """
            verts_state: n_verts, dim
            verts_mass: n_verts, 1
            Order: mesh, patch
            Direction: patch -> mesh only
            is_target: True only when no hierarchical structure
            # need to generate data before call this func; eg, noise: noise+state; vertices: torch.ones((num_verts, 1)).to(verts_state)
        """
        num_verts = list(node_wise_dict.values())[0].shape[0]
        device = list(node_wise_dict.values())[0].device
        # TODO: check src, dst the direction
        mesh_rel = dynamic_edges
        ## Build graph: src, dst
        if mesh_rel is not None:
            g = dgl.graph((mesh_rel[:, 1], mesh_rel[:, 0]), num_nodes=num_verts, idtype=torch.int64)
        else:
            g = dgl.graph(((), ()), num_nodes=num_verts, idtype=torch.int64).to(device)
        # No need for now, since all nodes are reciever and sender
        # if VERT_ID not in node_wise_dict.keys():
        #     node_wise_dict[VERT_ID] = torch.ones((num_verts, 1)).to(mesh_rel.device)
        for key, val in node_wise_dict.items():
            assert val.shape[0] == num_verts
            g.ndata[key] = val
        return g
    
    def connect_graph(self, num_points, num_clusters, p2c_edges, c2p_edges, connect_node_wise_dict):
        ## Build graph: src, dst
        device = None
        g = dgl.graph((p2c_edges[:, 1], p2c_edges[:, 0]), num_nodes=num_points+num_clusters, idtype=torch.int64)
        for key, val in connect_node_wise_dict.items():
            assert val.shape[0] == num_points+num_clusters
            g.ndata[key] = val
            if device == None:
                device = val.device
        g.edata[P2C_EDGE_ID] = torch.ones((p2c_edges.shape[0], 1)).float().to(device)
        g.add_edges(
            c2p_edges[:, 1], c2p_edges[:, 0],
            data={C2P_EDGE_ID: torch.ones((c2p_edges.shape[0], 1)).float().to(device)})
        return g


@PREPROCESSOR.register_module()
class GsHieSkeletonDynamicDGLProcessor(object):
    def __init__(self, **kwargs):
        pass

    def set_device(self, device):
        self.device = device

    def batch_preprocess(self, 
        input_graph, connect_graph, original_mov_state, attr, diag_volume, cur_cov, external_forces,
        joint_ids, cluster_nids, joint_nids, root_ids, edge_list):
        device = input_graph[0].device
        dtype = input_graph[0].ndata['cur_state'].dtype

        joint_ids_list = list(joint_ids.values())
        root_nids = torch.zeros_like(cluster_nids)
        root_nids[root_ids] = 1

        num_nodes = cluster_nids.shape[0]
        senders   = [edge[0] for edge in edge_list]
        receivers = [edge[1] for edge in edge_list]
        particle_level_graph = input_graph[0]
        cluster_level_graph = input_graph[-1]
        
        #build graph
        g = dgl.graph((senders, receivers), num_nodes=num_nodes, idtype=torch.int64, device=device)

        #build node data
        g.ndata['cluster_nids'] = cluster_nids.to(device)
        g.ndata['joint_nids'] = joint_nids.to(device)
        g.ndata['root_ids'] = root_nids.to(device)
        #how the 'attr' generated?
        node_keys = ['attr', 'prev_state', 'cur_state', 'template_state', 'orig_template_state', 'pin_mask', 'external']
        for key in node_keys:
            #get cluster_nodes
            value_cluster = cluster_level_graph.ndata[key]
            value_joint   = particle_level_graph.ndata[key][joint_ids_list]
            value = torch.cat([value_joint, value_cluster], dim=0)
            g.ndata[key] = value

        #build edge data
        edge_attr = {
            'dist': [],
            'pos': [],
        }
        for edge in edge_list:
            sender, receiver = edge[0], edge[1]
            cur_state_sender, cur_state_receiver = g.ndata['cur_state'][sender], g.ndata['cur_state'][receiver]
            pos = cur_state_receiver - cur_state_sender
            dist = torch.norm(pos, dim=-1)
            edge_attr['dist'].append(dist)
            edge_attr['pos'].append(pos)
        edge_attr['dist'] = torch.tensor(edge_attr['dist'], device=device, dtype=dtype).reshape(-1, 1)
        edge_attr['pos'] = torch.stack(edge_attr['pos'], dim=0)
        edge_attr['attr'] = torch.cat([edge_attr['dist'], edge_attr['pos']], dim=-1)
        g.edata['attr'] = edge_attr['attr']
            
        return g
    
    def update_graph(self, g, pred_joints_pos, pred_cluster_pos, edge_list):
        cluster_nids = g.ndata['cluster_nids']
        joint_nids = g.ndata['joint_nids']
        device = pred_joints_pos.device
        dtype = pred_joints_pos.dtype

        # Update points
        g.ndata['prev_state'] = g.ndata['cur_state']
        g.ndata['cur_state'] = torch.concatenate([pred_cluster_pos, pred_joints_pos], dim=0)
        # g.ndata['cur_state'][joint_nids] = pred_joints_pos
        # g.ndata['cur_state'][cluster_nids] = pred_cluster_pos

        # Update edges
        edge_attr = {
            'dist': [],
            'pos': [],
        }
        for edge in edge_list:
            sender, receiver = edge[0], edge[1]
            cur_state_sender, cur_state_receiver = g.ndata['cur_state'][sender], g.ndata['cur_state'][receiver]
            pos = cur_state_receiver - cur_state_sender
            dist = torch.norm(pos, dim=-1)
            edge_attr['dist'].append(dist)
            edge_attr['pos'].append(pos)
        edge_attr['dist'] = torch.tensor(edge_attr['dist'], device=device, dtype=dtype).reshape(-1, 1)
        edge_attr['pos'] = torch.stack(edge_attr['pos'], dim=0)
        edge_attr['attr'] = torch.cat([edge_attr['dist'], edge_attr['pos']], dim=-1)
        g.edata['attr'] = edge_attr['attr']

        return g

# @PREPROCESSOR.register_module()
# class GSHieEmbodiedDGLProcessor(object):
#     def __init__(self, radius, group_cfg, anchor_prefix='anchor_', eps=1e-7, **kwargs) -> None:
#         self.radius = radius
#         if isinstance(group_cfg, list):
#             assert len(group_cfg) == len(radius) # layer = len(radius)+1, 1 is extra layer for manually fixed root. Thus, hie-layer is len(radius)
#             self.grouper = [
#                 QueryAndGroup(**g_cfg) for g_cfg in group_cfg]
#         else:
#             assert isinstance(group_cfg, dict)
#             self.grouper = QueryAndGroup(**group_cfg)
#         self.eps = eps
#         self.graph_builder = BuildGsDGLGraph(**kwargs)
#         self.anchor_prefix = anchor_prefix
#         self.density_activation = density_activation()
#         self.density_deactivation = density_deactivation()
    
#     def batch_preprocess(self,
#                     prev_state,cur_state, template_state, attr, diag_volume, cur_cov, external_forces, 
#                     p2c_mapping=None, dynamic_base=True):
#         '''
#             bs == 1 case
#             cur_state: n_point, 3
#             template_state: n_point, 3
#             attr: n_point, attr_dim
#             diag_volume: n_point, 1
#             external_forces: n_point, 3
#             p2c_mapping: last one is the relation towards the pinned verts
#         '''
#         assert len(p2c_mapping) == len(self.radius)
#         connect_graph_list = []
#         base_g = self._preprocess(
#             cur_state, template_state, attr, 
#         )