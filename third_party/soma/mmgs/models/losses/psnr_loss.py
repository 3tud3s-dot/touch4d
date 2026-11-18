import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from mmgs.core import multi_apply


def psnr_error(pred, label, max_val=1.0, eps=1e-8, weight=None, reduction='sum', avg_factor=None, **kwargs):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    mse = torch.mean((pred - label) ** 2)
    mse = torch.clamp(mse, min=eps)
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    loss = -psnr

    if weight is not None:
        loss *= weight

    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction)
    return loss,



@LOSSES.register_module()
class PSNRLoss(nn.Module):
    """Cross entropy loss

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_lovasz'.
    """

    def __init__(self,
                 max_val=1.0,
                 eps=1e-8,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_psnr',):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = psnr_error
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                ssim_max_val=1.0,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = multi_apply(
            self.criterion,
            cls_score, label,
            max_val=self.max_val, eps=self.eps,
            weight=weight, reduction=reduction, avg_factor=avg_factor)[0]
        # loss = []
        # for i in range(len(cls_score)):
        #     loss.append(self.criterion(
        #         cls_score[i], label[i],
        #         kernal_size=self.kernel_size, max_val=ssim_max_val,
        #         weight=weight, reduction=reduction, avg_factor=avg_factor)[0])
        loss = torch.stack(loss).mean()
        loss *= self.loss_weight
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
