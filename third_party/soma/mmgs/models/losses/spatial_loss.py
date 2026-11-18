import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from mmgs.core import multi_apply


def spatial_error(pred, label, weight=None, reduction='sum', avg_factor=None, **kwargs):
    device = pred.device
    H, W, C = pred.shape
    dtype = pred.dtype


    spatial_distribution = torch.ones_like(label)
    bg_pixel = label[0,0] #TODO, given a parameter
    object_mask = (label != bg_pixel)

    pred_object_mask = ~torch.all(torch.isclose(pred, bg_pixel, atol=1e-2), dim=-1)
    label_bg_mask = torch.all(torch.isclose(label, bg_pixel, atol=1e-2), dim=-1)
    label_object_mask = ~label_bg_mask

    object_positions = (~label_object_mask).nonzero(as_tuple=False).float()
    assert object_positions.shape[0] > 0, "No object in the label"
    y_coords = torch.arange(H, device=device, dtype=dtype).view(H, 1, 1)
    x_coords = torch.arange(W, device=device, dtype=dtype).view(1, W, 1)

    y_diffs = y_coords - object_positions[:, 0]
    x_diffs = x_coords - object_positions[:, 1]
    distance_to_all = torch.sqrt(y_diffs**2 + x_diffs**2) #开销太大
    distance_to_object_margin, _ = torch.min(distance_to_all, dim=-1)
    
    spatial_distribution = torch.ones(H, W, device=device)
    spatial_distribution[label_bg_mask] += distance_to_object_margin
    spatial_distribution = torch.log(spatial_distribution)

    spatial_loss = spatial_distribution * pred_object_mask
    spatial_loss = weight_reduce_loss(
        spatial_loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    assert not torch.isnan(spatial_loss)

    return spatial_loss

@LOSSES.register_module()
class SpatialLoss(nn.Module):
    def __init__(self, reduction='sum', loss_weight=1.0, loss_name='loss_spatial'):
        super(SpatialLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        self.criterion = spatial_error
        self._loss_name = loss_name

    def forward(self, 
                pred, 
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = multi_apply(
            self.criterion,
            pred, label,
            weight=weight, reduction=reduction, avg_factor=avg_factor)[0]
        
        loss = torch.stack(loss).mean()
        loss *= self.loss_weight
        return loss
    
    @property
    def loss_name(self):
        return self._loss_name
