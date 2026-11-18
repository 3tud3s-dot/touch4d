import torch
from torch import nn as nn
import torch.nn.functional as F
# import scipy.sparse as sp
# from scipy.sparse import vstack, csr_matrix
# from scipy.sparse.linalg import spsolve
# from scipy.spatial.transform import Rotation as R
import numpy as np

from mmcv.ops import grouping_operation
# from mmgd.datasets.utils import to_numpy_detach
# from pytorch3d.transforms import matrix_to_quaternion
# from mmgd.datasets.utils.hood_common import gather, unsorted_segment_sum


def vertex_area(vertices, faces):
    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v12 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
    face_areas = np.linalg.norm(np.cross(v01, v12), axis=-1)
    vertex_areas = np.zeros((vertices.shape[0],), np.float32)
    for i, face in enumerate(faces):
        vertex_areas[face] += face_areas[i]
    vertex_areas *= 1 / 6
    total_area = vertex_areas.sum()
    return vertex_areas, face_areas, total_area
	
def face_normals(verts, faces):
	face_normals = torch.cross(
		verts[faces[:, 2]] - verts[faces[:, 1]],
		verts[faces[:, 0]] - verts[faces[:, 1]],
		dim=1,
	)
	return face_normals

def face_normals_batched(verts, faces, normalized=True, eps=1e-7, with_face_area=False):
	'''
		bs, num_verts, 3
		bs, num_faces, 3
	'''
	# face_verts: bs, 3(coordinate), num_faces, 3(face)
	face_verts = grouping_operation(verts.transpose(-1, -2), faces.int()).permute(0, 2, 3, 1)
	face_normals = torch.cross(
		face_verts[:, :, 2] - face_verts[:, :, 1],
		face_verts[:, :, 0] - face_verts[:, :, 1],
		dim=-1,
	)
	# face_normals = torch.cross(
	#     torch.gather(verts, 1, faces[..., 2:3].repeat(1, 1, 3)) - torch.gather(verts, 1, faces[..., 1:2].repeat(1, 1, 3)),
	#     torch.gather(verts, 1, faces[..., 0:1].repeat(1, 1, 3)) - torch.gather(verts, 1, faces[..., 1:2].repeat(1, 1, 3)),
	#     dim=-1,
	# )

	face_area = torch.norm(face_normals, dim=-1, keepdim=True)
	if normalized:
		face_normals = face_normals / (face_area + eps)
	if with_face_area:
		return face_normals, face_area
	else:
		return face_normals

def vertex_normal_batched(verts, faces, v2f_mask_sparse, normalized=True, eps=1e-7):
	bs, v_num, f_num = v2f_mask_sparse.shape
	face_normals = face_normals_batched(verts, faces, normalized=normalized, eps=eps)
	vert_normals = torch.bmm(v2f_mask_sparse, face_normals)
	assert normalized
	if normalized:
		vert_normals = vert_normals / (torch.linalg.norm(vert_normals, dim=-1, ord=2, keepdim=True) + eps)
	return vert_normals

def vertex_normal_batched_simple(vertices, faces):
	'''
		v: bs, n_verts, 3dim
		f: bs, n_face, 3nodes
		return: bs, n_verts, 3dim
	'''
	v = vertices
	f = faces

	# bs, n_face, 3nodes, 3dim
	triangles = gather(v, f, 1, 2, 2)

	# Compute face normals
	v0, v1, v2 = torch.unbind(triangles, dim=-2)
	e0 = v1 - v0
	e1 = v2 - v1
	e2 = v0 - v2
	face_normals = torch.linalg.cross(e0, e1) + torch.linalg.cross(e1, e2) + torch.linalg.cross(e2, e0)  # F x 3

	vn = unsorted_segment_sum(face_normals, f, 1, 2, 2)

	vn = F.normalize(vn, dim=-1)
	return vn

def remove_interpenetration_np(mesh_verts, base_verts, base_normals, l_mask, grouper, min_distance=0.01, weight=2.0, eps=1e-7):
	"""Deforms `mesh` to remove its interpenetration from `base`.
	This is posed as least square optimization problem which can be solved
	faster with sparse solver.

	mesh_verts: [N, 3]. Single batch
	l_mask: didn't minus diag, it's normalized one
	"""
	bs, nverts, dim = mesh_verts.shape
	assert bs == 1

	grouped_results = grouper(base_verts, mesh_verts, base_normals)
	grouped_normals, grouped_xyz = grouped_results
	grouped_diff = mesh_verts.transpose(1, 2).unsqueeze(-1) - grouped_xyz  # relative offsets
	grouped_normals = grouped_normals.permute(0, 2, 3, 1)
	grouped_diff = grouped_diff.permute(0, 2, 3, 1)
	direction = torch.sum(grouped_diff * grouped_normals, dim=-1)

	indices = torch.where(direction < 0)

	# pentgt_points = grouped_xyz[indices] - mesh_verts[indices]
	grouped_xyz = grouped_xyz.permute(0, 2, 3, 1)
	pentgt_points = -grouped_diff[indices]
	pentgt_points = grouped_xyz[indices] \
					+ min_distance * pentgt_points / (eps + torch.norm(pentgt_points, dim=-1, keepdim=True))
	fixed_direction = torch.sum(grouped_normals[indices] * (pentgt_points - grouped_xyz[indices]), dim=-1)

	# TODO: need futher check if need grad: clone/copy_
	tgt_points = mesh_verts.clone().unsqueeze(2)
	tgt_points[indices] = weight * pentgt_points

	# rc = torch.arange(nverts).to(mesh_verts)
	# data = torch.ones((bs, nverts, 1)).to(mesh_verts)
	# data[indices] *= weight
	# I = torch.sparse_coo_tensor(torch.stack([rc, rc], dim=0), data, size=(nverts, nverts)).coalesce()
	
	# # Dense case
	# # l_mask -= xxx, will modify inplace
	# l_mask_normalized = l_mask - torch.eye(nverts).to(l_mask).unsqueeze(0) # Have decimation 1e-7
	# I = torch.eye(nverts).to(mesh_verts).reshape(-1, nverts, nverts, 1)
	# I[indices] *= weight
	# I = I.squeeze(-1)
	# # bs, 2N, N
	# A = torch.cat([l_mask_normalized, I], dim=1)
	# # bs, 2N, 3
	# b = torch.cat([
	# 	torch.bmm(l_mask_normalized, mesh_verts),
	# 	tgt_points.squeeze(2)], dim=1)
	# sys_A = torch.bmm(A.transpose(-1, -2), A)
	# bias_B = torch.bmm(A.transpose(-1, -2), b)
	# res_verts = spsolve(sys_A[0].detach().cpu().numpy(), bias_B[0].detach().cpu().numpy())

	# Sparse case
	l_mask_normalized = l_mask.coalesce()
	l_mask_indices = l_mask_normalized.indices().detach().cpu().numpy()
	l_mask_values = l_mask_normalized.values().detach().cpu().numpy()
	l_mask_size = l_mask_normalized.size()
	laplacian = csr_matrix(
		(l_mask_values, (l_mask_indices[0], l_mask_indices[1])),
		shape=(nverts, nverts)) - sp.eye(nverts)
	I_data = torch.ones(nverts).reshape(1, -1, 1).to(mesh_verts)
	I_data[indices] *= weight
	I_data = I_data.squeeze(2).squeeze(0).detach().cpu().numpy()
	I = csr_matrix(
		(I_data, (np.arange(nverts), np.arange(nverts))),
		shape=(nverts, nverts))
	A = vstack([laplacian, I])
	b = np.vstack((
		laplacian.dot(mesh_verts[0].detach().cpu().numpy()),
		tgt_points.squeeze(2)[0].detach().cpu().numpy()
	))
	res_verts = spsolve(A.T.dot(A), A.T.dot(b))

	res_verts = torch.from_numpy(res_verts).to(mesh_verts)
	return res_verts

def random_3D_vector(shape, device, eps=1e-7):
	'''
		shape: bs, n_verts, 3
	'''
	random_x = torch.zeros(*shape).to(device)
	random_x[:, :, 0] = 1
	zero_position = torch.where(torch.abs(data) < eps)
	random_x[zero_position] = 1

def rotation_from_normals(data, eps=1e-7):
	# bs, n_verts, dim
	bs, n_verts, dim = data.shape
	# Get random vector
	random_x = torch.randn_like(data).to(data)
	random_x[:, :, 0] = 1
	zero_position = torch.where(torch.abs(data) < eps)
	random_x[zero_position] = 1

	# Calculate orthogonal vector
	## v' = v - dot(v, r) / |r| * r/|r|
	orth_x = random_x - torch.sum(data * random_x, dim=-1, keepdim=True) * data
	orth_x = orth_x / (torch.linalg.norm(orth_x, dim=-1, ord=2, keepdim=True) + eps)

	## the rest rotation
	## y = z cross x
	orth_y = torch.cross(data, orth_x)

	## rot_M is from obj to world
	rot_M = torch.stack([orth_x, orth_y, data], dim=-1)
	## result need from world to obj. Thus transpose
	return rot_M.transpose(-1, -2)

def rotation_from_quats_np(data, eps=1e-7):
	# seq_len, history, dim
	seq_len, n_history, dim = data.shape
	data = data.reshape(-1, dim)

	# Wind q: wxyz format
	# Func need xyzw
	data_q = np.zeros_like(data)
	w = data[:, 0]
	data_q[:, 0:3] = data[:, 1:4]
	data_q[:, 3] = w
	# Check not all 0
	sum_q = np.sum(data_q == 0, axis=-1)
	zero_q = np.where(sum_q == dim)
	data_q[zero_q, 3] = 1
	rot_q = R.from_quat(data_q).as_matrix()
	rot_dim = rot_q.shape[-1]
	rot_q = rot_q.reshape(seq_len, n_history, rot_dim, rot_dim)
	
	return rot_q


def extract_rotation_with_padding(rotation_M, group_idx, mask=None, pad_dim=3):
	'''
		rotation_M: bs, src_num, 3, 3
		group_idx: bs, tar_num, 1
		mask: bs, tar_num, 1
	'''
	bs = rotation_M.shape[0]
	src_num = rotation_M.shape[1]
	tar_num = group_idx.shape[1]

	# bs, tar_num, 1, 3, 3
	grouped_rotation = grouping_operation(rotation_M.reshape(bs, src_num, -1).transpose(-1, -2), group_idx).permute(0, 2, 3, 1).reshape(bs, tar_num, -1, 3, 3)
	assert grouped_rotation.shape[2] == 1
	# bs, tar_num, 3, 3
	grouped_rotation = grouped_rotation.squeeze(2)
	if mask is not None:
		grouped_rotation *= mask.unsqueeze(-1)
		# Rotation is identity matrix
		grouped_rotation[torch.where(mask.squeeze(-1) == False)] = torch.eye(pad_dim).to(grouped_rotation)
	
	return grouped_rotation


def normal_cos_sin(vec1, vec2, eps=1e-7):
	'''
		vec1: ..., 3
	'''
	# Inputs are unit normal, so no need
	# norm_1 = torch.linalg.norm(vec1, dim=-1, keepdim=True)
	# norm_2 = torch.linalg.norm(vec2, dim=-1, keepdim=True)
	dot_p = torch.sum(vec1*vec2, dim=-1, keepdim=True)
	# cos = dot_p / (norm_1*norm_2+self.eps)
	cos = dot_p
	cross_p = torch.cross(vec1, vec2, dim=-1)
	# sin_n = cross_p / (norm_1*norm_2+self.eps)
	# sin = torch.linalg.norm(sin_n, dim=-1, keepdim=True)
	sin = torch.linalg.norm(cross_p, dim=-1, keepdim=True)
	return cos, sin

def normal_theta(vec1, vec2, eps=1e-7):
	cos, sin = normal_cos_sin(vec1, vec2, eps=eps)
	theta = torch.atan2(sin, cos)
	return theta

def check_zero_coord(coord, c_norm):
	'''
		normal: n_edge, 2
		n_norm: n_edge, 1
	'''
	### Check zeros place
	zero_idx = torch.where(c_norm == 0)
	if len(zero_idx[0]) > 0:
		coord[zero_idx] = 1.0
		c_norm[zero_idx] = 1.0
	return coord, c_norm

def rot_by_x_yz_toz(x_local_yz, device):
	# Since to z, thus
	#  y' = cos(pi/2-a) = sin(a) = z
	#  z' = sin(pi/2-a) = cos(a) = y
	x_local_yz_norm = torch.linalg.norm(x_local_yz, dim=-1, keepdim=True)
	x_local_yz, x_local_yz_norm = check_zero_coord(x_local_yz, x_local_yz_norm)
	x_local_cossin = x_local_yz / x_local_yz_norm
	x_first_line = torch.cat([torch.ones_like(x_local_cossin[:, 0:1]).to(device), torch.zeros_like(x_local_cossin).to(device)], dim=-1)
	x_second_line = torch.cat([torch.zeros_like(x_local_cossin[:, 0:1]).to(device), x_local_cossin[:, 1:2], -1*x_local_cossin[:, 0:1]], dim=-1)
	x_third_line = torch.cat([torch.zeros_like(x_local_cossin[:, 0:1]).to(device), x_local_cossin[:, 0:1], x_local_cossin[:, 1:2]], dim=-1)
	x_Rot = torch.stack([x_first_line, x_second_line, x_third_line], dim=1)
	return x_Rot

def rot_by_y_zx_toz(y_local_zx, device):
	# Since to z
	y_local_zx_norm = torch.linalg.norm(y_local_zx, dim=-1, keepdim=True)
	y_local_zx, y_local_zx_norm = check_zero_coord(y_local_zx, y_local_zx_norm)
	y_local_cossin = y_local_zx / y_local_zx_norm
	y_first_line = torch.cat([y_local_cossin[:, 0:1], torch.zeros_like(y_local_cossin[:, 0:1]).to(device), y_local_cossin[:, 1:2]], dim=-1)
	y_second_line = torch.cat([torch.zeros_like(y_local_cossin[:, 0:1]).to(device), torch.ones_like(y_local_cossin[:, 0:1]).to(device), torch.zeros_like(y_local_cossin[:, 0:1]).to(device)], dim=-1)
	y_third_line = torch.cat([-1*y_local_cossin[:, 1:2], torch.zeros_like(y_local_cossin[:, 0:1]).to(device), y_local_cossin[:, 0:1]], dim=-1)
	y_Rot = torch.stack([y_first_line, y_second_line, y_third_line], dim=1)
	return y_Rot

def get_rel_rotation(edge_dir, template_normal, vec_normal, eps=1e-7):
	z_axis = template_normal
	x_axis = torch.cross(edge_dir, z_axis)
	x_axis = x_axis / (torch.linalg.norm(x_axis, dim=-1, keepdim=True)+eps)
	y_axis = torch.cross(z_axis, x_axis)
	y_axis = y_axis / (torch.linalg.norm(y_axis, dim=-1, keepdim=True)+eps)

	# n_edge, stack(3), 3
	base_axis = torch.stack([x_axis, y_axis, z_axis], dim=1)
	# n_edge, 3
	local_dst_normal = torch.bmm(base_axis, vec_normal.unsqueeze(-1)).squeeze(-1)

	# Check rotation
	## Rot by y
	device = local_dst_normal.device
	y_local_zx = torch.cat([local_dst_normal[:, 2:3], local_dst_normal[:, 0:1]], dim=-1)
	# y_Rot_T.T: dst -> 0, y, 1. if x == 0, the out is 0, y, z
	# torch.bmm(y_Rot_T.transpose(-1, -2), local_dst_normal.unsqueeze(-1)).squeeze(-1) -> 0, y, 1
	y_Rot_T = rot_by_y_zx_toz(y_local_zx, device)
	local_dst_normal_yRot = torch.bmm(y_Rot_T.transpose(-1, -2), local_dst_normal.unsqueeze(-1)).squeeze(-1)
	x_local_yz = local_dst_normal_yRot[:, 1:3]
	# x_Rot: dst, -> x, 0, 1
	# torch.bmm(x_Rot, local_dst_normal_yRot.unsqueeze(-1)).squeeze(-1) -> 0, 0, 1
	x_Rot = rot_by_x_yz_toz(x_local_yz, device)
	# dst -> 0, 0, 1
	# torch.bmm(dst_rot_mat, local_dst_normal.unsqueeze(-1)).squeeze(-1) -> 0, 0, 1
	dst_rot_mat = torch.bmm(x_Rot, y_Rot_T.transpose(-1, -2))

	# Back into original 3D space!!!
	orig_mat = torch.bmm(torch.bmm(base_axis.transpose(-1, -2), dst_rot_mat), base_axis)
	# quant = matrix_to_quaternion(dst_rot_mat)
	orig_quant = matrix_to_quaternion(orig_mat)
	return orig_quant

def get_rel_rotation_edge_vn(edge_dir, dst_normal, src_normal, eps=1e-7):
	''' 
		Must manually align with the above func
	'''
	y_axis = edge_dir
	x_axis = torch.cross(y_axis, src_normal)
	x_axis = x_axis / (torch.linalg.norm(x_axis, dim=-1, keepdim=True)+eps)
	z_axis = torch.cross(x_axis, y_axis)
	z_axis = z_axis / (torch.linalg.norm(z_axis, dim=-1, keepdim=True)+eps)

	# n_edge, stack(3), 3
	base_axis = torch.stack([x_axis, y_axis, z_axis], dim=1)
	# n_edge, 3
	local_dst_normal = torch.bmm(base_axis, dst_normal.unsqueeze(-1)).squeeze(-1)
	local_src_normal = torch.bmm(base_axis, src_normal.unsqueeze(-1)).squeeze(-1)

	# Check rotation
	## dst to z
	device = local_dst_normal.device
	y_local_zx = torch.cat([local_dst_normal[:, 2:3], local_dst_normal[:, 0:1]], dim=-1)
	# y_Rot_T.T: dst -> 0, y, 1
	# torch.bmm(y_Rot_T.transpose(-1, -2), local_dst_normal.unsqueeze(-1)).squeeze(-1) -> 0, y, 1
	y_Rot_T = rot_by_y_zx_toz(y_local_zx, device)
	local_dst_normal_yRot = torch.bmm(y_Rot_T.transpose(-1, -2), local_dst_normal.unsqueeze(-1)).squeeze(-1)
	x_local_yz = local_dst_normal_yRot[:, 1:3]
	# x_Rot: dst, -> x, 0, 1
	# torch.bmm(x_Rot, local_dst_normal_yRot.unsqueeze(-1)).squeeze(-1) -> 0, 0, 1
	x_Rot = rot_by_x_yz_toz(x_local_yz, device)
	# dst -> 0, 0, 1
	# torch.bmm(dst_rot_mat, local_dst_normal.unsqueeze(-1)).squeeze(-1) -> 0, 0, 1
	dst_rot_mat = torch.bmm(x_Rot, y_Rot_T.transpose(-1, -2))
	
	## src rot to z
	## Here, theta is pi/2 - theta_by_yz, which need zy as input
	x_local_yz_src = local_src_normal[:, 1:3]
	# src -> x, 0, 1, x==0
	# torch.bmm(x_Rot_src, local_src_normal.unsqueeze(-1)).squeeze(-1) -> x(0), 0, 1
	x_Rot_src = rot_by_x_yz_toz(x_local_yz_src, device)

	# torch.bmm(rot_mas, src): src -> 0, 0, 1 -> dst
	# torch.bmm(rot_mat, local_src_normal.unsqueeze(-1)).squeeze(-1) - local_dst_normal -> (0, 0, 0)
	rot_mat = torch.bmm(dst_rot_mat.transpose(-1, -2), x_Rot_src)

	# Back into original 3D space!!!
	orig_mat = torch.bmm(torch.bmm(base_axis.transpose(-1, -2), rot_mat), base_axis)
	# quant = matrix_to_quaternion(rot_mat)
	orig_quant = matrix_to_quaternion(orig_mat)
	return orig_quant
	