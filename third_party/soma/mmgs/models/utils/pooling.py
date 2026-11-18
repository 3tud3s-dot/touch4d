from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule


class MaxPoolingConcat(BaseModule):
	def __init__(self, init_cfg=None):
		super(MaxPoolingConcat, self).__init__(init_cfg)

	def forward(self, x, indices, **kwargs):
		'''
			X: bs, num_verts, dim
			indices (list of tensor): num_outfit + 1 + pad_dim
		'''
		# TODO: try to use mask to accelorate
		bs, num_verts, dim = x.shape
		batched_X = []
		for b_outfit, b_ind in zip(x, indices):
			# b_outfit: num_sample, dim
			# b_ind: num_outfit+1
			_X = []
			for i in range(1, len(b_ind)):
				s, e = b_ind[i-1], b_ind[i]
				# should be dim
				pooling_feature = F.max_pool1d(b_outfit[s:e].transpose(0,1), kernel_size=e.item()-s.item()).reshape(1, -1)
				pooling_feature = torch.tile(pooling_feature, (e-s, 1))
				_X.append(pooling_feature)
			_X = torch.concat(_X, dim=0)
			# Padding
			# After: num_verts, dim
			_X = torch.concat([_X, torch.zeros((num_verts-_X.shape[0], dim)).cuda()], dim=0)
			batched_X.append(_X)
		# bs, num_verts, dim
		batched_X = torch.stack(batched_X, axis=0)

		return batched_X