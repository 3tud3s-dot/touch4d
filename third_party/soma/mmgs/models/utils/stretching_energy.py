import einops
import torch
from torch import nn
from ..builder import SIMULATORS
from mmgs.datasets.utils.phys import deformation_gradient, green_strain_tensor, gather_triangles, make_pervertex_tensor_from_lens
from mmgs.core import multi_apply
from functools import partial


class Criterion(nn.Module):
    def __init__(self, weight=1.0, thickness=4.7e-4):
        super().__init__()
        self.weight = weight
        self.thickness = thickness
        self.name = 'stretching_energy'

    def create_stack(self, triangles_list, param):
        '''
            triangles_list: bs, n_faces, 3(vert), 3(dim)
            param: bs, 1
            return: bs*n_faces
        '''
        # bs, n_faces
        lens = [x.shape[0] for x in triangles_list]
        # bs*n_faces
        stack = make_pervertex_tensor_from_lens(lens, param)[:, 0]
        return stack

    def forward(self, pred_verts, faces, Dm_inv, f_area, lame_mu):
        '''
            f_area: bs*n_faces, 1
        '''
        f_area = f_area[None, ..., 0]
        device = Dm_inv.device

        B = len(pred_verts)

        # bs, n_faces, 3(vert), 3(dim)
        triangles_list = []
        for i in range(B):
            # n_verts, 3
            v = pred_verts[i]
            # n_faces, 3
            f = faces[i].T
            # n_faces, 3(vert), 3(dim)
            triangles = gather_triangles(v.unsqueeze(0), f)[0]
            triangles_list.append(triangles)

        # bs*n_faces
        lame_mu_stack = self.create_stack(triangles_list, sample['cloth'].lame_mu)
        # bs*n_faces
        lame_lambda_stack = self.create_stack(triangles_list, sample['cloth'].lame_lambda)
        # bs*n_faces, 3, 3
        triangles = torch.cat(triangles_list, dim=0)

        # bs*n_faces, 3, 2
        F = deformation_gradient(triangles, Dm_inv)
        # bs*n_faces, 2, 2
        G = green_strain_tensor(F)

        I = torch.eye(2).to(device)
        # bs*n_faces, 2, 2
        I = einops.repeat(I, 'm n -> k m n', k=G.shape[0])
        
        # bs*n_faces
        ## Sun the diagonal
        G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        
        # bs*n_faces, 2, 2
        S = lame_mu_stack[:, None, None] * G + 0.5 * lame_lambda_stack[:, None, None] * G_trace[:, None, None] * I
        # bs*n_faces, 2, 2
        energy_density_matrix = S.permute(0, 2, 1) @ G
        # bs*n_faces
        energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        # bs*n_faces
        f_area = f_area[0]
        # bs*n_faces
        energy = f_area * self.thickness * energy_density
        loss = energy.sum() / B

        return dict(loss=loss)

@SIMULATORS.register_module()
class StretchingPotentialPrior(nn.Module):
    def __init__(self, weight=1.0, thickness=4.7e-4, lame_mu_override=None, lame_lambda_override=None):
        super().__init__()
        self.weight = weight
        self.thickness = thickness
        self.name = 'stretching_energy'
        self.lame_mu_override = lame_mu_override
        self.lame_lambda_override = lame_lambda_override

    def create_stack(self, triangles_list, param):
        '''
            triangles_list: bs, n_faces, 3(vert), 3(dim)
            param: bs, 1
            return: bs*n_faces
        '''
        # bs, n_faces
        lens = [x.shape[0] for x in triangles_list]
        # bs*n_faces
        stack = make_pervertex_tensor_from_lens(lens, param)[:, 0]
        return stack
    
    def init_weights(self):
        '''
        To align with the pipeline
        '''
        pass
    
    def inference(self, inputs, gt_label=None, **kwargs):
        '''
        To align with the pipeline
        '''
        # Only need pos
        pred_verts = [vert[:, :3] for vert in inputs['dynamic']['state']]
        static_inputs = inputs['static']
        # Can do something using self.test_cfg
        output = self.forward(pred_verts, **static_inputs)
        
        return dict(pred_energy=output)
    
    def forward(self, pred_verts, faces, Dm_inv, f_area, lame_mu_raw, lame_lambda_raw, **kwargs):
        '''
            pred_verts: bs, n_verts, 3+x
            faces: bs, n_faces, 3
            Dm_inv: bs, 2, 2
            f_area: bs, n_faces, 1
            lame_mu: bs, 1 (constant, can be randomized)
            lame_lambda: bs, 1 (constant: 44400)
            TODO: f_area, Dm_inv can pre compute
        '''
        device = pred_verts[0].device
        bs = len(pred_verts)

        potential_prior_list = []
        # bs, n_faces, 3(vert), 3(dim)
        for i in range(bs):
            triangles_list = []
            # n_verts, 3
            v = pred_verts[i]
            # n_faces, 3
            f = faces[i]
            # n_faces, 3(vert), 3(dim)
            triangles = gather_triangles(v.unsqueeze(0), f)[0]
            triangles_list.append(triangles)
            # n_faces
            if self.lame_mu_override is not None:
                lame_mu_raw_override = torch.full_like(lame_mu_raw[i], self.lame_mu_override).to(lame_mu_raw[i])
                lame_mu_stack = self.create_stack(triangles_list, lame_mu_raw_override)
            else:
                lame_mu_stack = self.create_stack(triangles_list, lame_mu_raw[i])
            # n_faces
            if self.lame_lambda_override is not None:
                lame_lambda_raw_override = torch.full_like(lame_lambda_raw[i], self.lame_lambda_override).to(lame_lambda_raw[i])
                lame_lambda_stack = self.create_stack(triangles_list, lame_lambda_raw_override)
            else:
                lame_lambda_stack = self.create_stack(triangles_list, lame_lambda_raw[i])
            # n_faces, 3, 2
            F = deformation_gradient(triangles, Dm_inv[i])
            # n_faces, 2, 2
            G = green_strain_tensor(F)
            I = torch.eye(2).to(device)
            # n_faces, 2, 2
            I = einops.repeat(I, 'm n -> k m n', k=G.shape[0])
            # n_faces
            ## Sun the diagonal
            G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
            # n_faces, 2, 2
            S = lame_mu_stack[:, None, None] * G + 0.5 * lame_lambda_stack[:, None, None] * G_trace[:, None, None] * I
            # n_faces, 2, 2
            energy_density_matrix = S.permute(0, 2, 1) @ G
            # n_faces
            energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
            # n_faces
            f_s = f_area[i][:, 0]
            # n_faces
            energy = f_s * self.thickness * energy_density
            potential_prior_list.append(energy.unsqueeze(-1))
        # TODO: energy dim is n_faces
        return potential_prior_list
    

@SIMULATORS.register_module()
class StretchingPotentialPriorNewMark(nn.Module):
    def __init__(self, weight=1.0, thickness=4.7e-4, lame_mu_override=None, lame_lambda_override=None, **kwargs):
        super().__init__()
        self.weight = weight
        self.thickness = thickness
        self.name = 'stretching_energy'
        self.lame_mu_override = lame_mu_override
        self.lame_lambda_override = lame_lambda_override

    def create_stack(self, triangles_list, param):
        '''
            triangles_list: bs, n_faces, 3(vert), 3(dim)
            param: bs, 1
            return: bs*n_faces
        '''
        # bs, n_faces
        lens = [x.shape[0] for x in triangles_list]
        # bs*n_faces
        stack = make_pervertex_tensor_from_lens(lens, param)[:, 0]
        return stack
    
    def init_weights(self):
        '''
        To align with the pipeline
        '''
        pass
    
    def inference(self, inputs, is_training=False, register_norm=False, out_energy=True, out_dev=False, **kwargs):
        '''
        To align with the pipeline
        '''
        # Only need pos
        pred_verts = [vert[:, :3] for vert in inputs['dynamic']['state']]
        static_inputs = inputs['static']

        faces = static_inputs['faces']
        Dm_inv = static_inputs['Dm_inv']
        f_area = static_inputs['f_area']
        lame_mu_raw = static_inputs['lame_mu_raw']
        lame_lambda_raw = static_inputs['lame_lambda_raw']
        vert_mask = static_inputs['vert_mask']

        assert out_energy or out_dev
        rst = dict()
        if out_energy:
            output = multi_apply(
                self.encode_decode,
                pred_verts, faces, Dm_inv, f_area, lame_mu_raw, lame_lambda_raw)[0]
            rst['pred'] = output
        
        if out_dev:
            output_dev = multi_apply(
                self.dev_encode_decode,
                pred_verts, faces, Dm_inv, f_area, lame_mu_raw, lame_lambda_raw, vert_mask, is_training=is_training)[0]
            rst['pred_dev'] = output_dev
        
        return rst
    
    def encode_decode(self, pred_verts, faces, Dm_inv, f_area, lame_mu_raw, lame_lambda_raw, **kwargs):
        '''
            pred_verts: n_verts, 3+x
            faces: n_faces, 3
            Dm_inv: 2, 2
            f_area: n_faces, 1
            lame_mu: 1 (constant, can be randomized)
            lame_lambda: 1 (constant: 44400)
            TODO: f_area, Dm_inv can pre compute
        
        '''
        device = pred_verts[0].device

        # n_verts, 3
        v = pred_verts
        # n_faces, 3
        f = faces
        # n_faces, 3(vert), 3(dim)
        triangles = gather_triangles(v.unsqueeze(0), f)[0]
        # n_faces
        if self.lame_mu_override is not None:
            lame_mu_raw_override = torch.full_like(lame_mu_raw, self.lame_mu_override).to(lame_mu_raw)
            lame_mu_stack = self.create_stack([triangles], lame_mu_raw_override)
        else:
            lame_mu_stack = self.create_stack([triangles], lame_mu_raw)
        # n_faces
        if self.lame_lambda_override is not None:
            lame_lambda_raw_override = torch.full_like(lame_lambda_raw, self.lame_lambda_override).to(lame_lambda_raw)
            lame_lambda_stack = self.create_stack([triangles], lame_lambda_raw_override)
        else:
            lame_lambda_stack = self.create_stack([triangles], lame_lambda_raw)
        # n_faces, 3, 2
        F = deformation_gradient(triangles, Dm_inv)
        # n_faces, 2, 2
        G = green_strain_tensor(F)
        I = torch.eye(2).to(device)
        # n_faces, 2, 2
        I = einops.repeat(I, 'm n -> k m n', k=G.shape[0])
        # n_faces
        ## Sun the diagonal
        G_trace = G.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        # n_faces, 2, 2
        S = lame_mu_stack[:, None, None] * G + 0.5 * lame_lambda_stack[:, None, None] * G_trace[:, None, None] * I
        # n_faces, 2, 2
        energy_density_matrix = S.permute(0, 2, 1) @ G
        # n_faces
        energy_density = energy_density_matrix.diagonal(dim1=-1, dim2=-2).sum(-1)  # trace
        # n_faces
        f_s = f_area[:, 0]
        # n_faces
        energy = f_s * self.thickness * energy_density
        # TODO: energy dim is n_faces
        # Return n_faces, 1
        return energy.unsqueeze(-1),
    
    def dev_encode_decode(self, pred_verts, faces, Dm_inv, f_area, lame_mu_raw, lame_lambda_raw, vert_mask, is_training=False):
        dev_func = partial(self.encode_decode,
                faces=faces, Dm_inv=Dm_inv, f_area=f_area, lame_mu_raw=lame_mu_raw, lame_lambda_raw=lame_lambda_raw)
        def _func(x):
            return dev_func(x)[0].sum(dim=0)
        # n_verts, 3
        dev_out = torch.autograd.functional.jacobian(_func, pred_verts, create_graph=is_training).squeeze(0)
        dev_out = dev_out * vert_mask
        return dev_out,
