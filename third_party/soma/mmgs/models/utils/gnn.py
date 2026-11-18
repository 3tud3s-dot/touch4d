import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer

from .feedforward_networks import FFN
from mmcv.runner import BaseModule


class GCNLayer(BaseModule):
	def __init__(self,
				 out_channels=3,
				 in_channels=0,
				 act_cfg=dict(type='ReLU', inplace=True),
				 init_cfg=None,
				 **kargs):
		super(GCNLayer, self).__init__(init_cfg)
		self.receiver_proj = FFN(
			[in_channels, out_channels],
			act_cfg=None,
			bias=False,)
		self.sender_proj = FFN(
			[in_channels, out_channels],
			act_cfg=None,
			bias=True,)
		self.activate = build_activation_layer(act_cfg)

	def forward(self, X, laplacian_matrix, **kwargs):
		'''
			X: bs, num_sample, dim
			laplacian_matrix: bs, num_sample, num_sample
		'''
		# Node encoding
		X0 = self.receiver_proj(X)
		# Neighborhood
		X1 = self.sender_proj(X)
		# Neighborhood conv
		X1 = torch.bmm(laplacian_matrix, X1)

		X = X0 + X1
		X = self.activate(X)
		return X