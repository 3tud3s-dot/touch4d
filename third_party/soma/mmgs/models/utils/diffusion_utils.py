import math
import torch
import torch.nn as nn
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      xavier_init, Swish)

from mmcv.runner import BaseModule


class TimeEmbedding(BaseModule):
    """
    ### Embeddings for $t$
    """

    def __init__(self,
                 in_channels=32, # original 64
                 out_channels=128, # 4*in_channels
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 final_act=False,
                 add_residual=False,
                 bias=True,
                 init_cfg=None,):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super(TimeEmbedding, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # First linear layer
        self.lin1 = Linear(self.in_channels, self.out_channels)
        # Activation; Originally it's swish
        # self.act = Swish()
        self.act = build_activation_layer(act_cfg) 
        # Second linear layer
        self.lin2 = Linear(self.out_channels, self.out_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.in_channels // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb