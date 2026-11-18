# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import ssim_loss

from ..builder import ACCURACY
# from mmcv.ops import QueryAndGroup
# from mmgs.core import multi_apply


def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.
    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
                 if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res

def accuracy_mse(pred, label, indices, indices_type, prefix='', reduction='sum', default_key='total', overwrite_align=False, vert_mask=None, ratio=False, **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    # element-wise losses
    mse_error = []
    for i in range(bs):
        m_e = F.mse_loss(pred[i], label[i], reduction='none')
        if ratio:
            m_e = (pred[i] / (label[i]+1e-7))**2
        if vert_mask is not None:
            m_e *= vert_mask[i]
        mse_error.append(m_e)

    # Calculate per outfit error
    for b_error, b_ind, b_type in zip(mse_error, indices, indices_type):
        if reduction == 'sum':
            acc_dict[default_key].append(b_error[0])
            for i in range(1, b_ind.shape[0]):
                assert b_type.shape[-1] == 1, "Only support idx input instead of one hot"
                g_type_idx = b_type[i-1, 0].int()
                g_type = GARMENT_TYPE[g_type_idx]
                g_error = b_error[0]
                acc_dict[g_type].append(g_error)
        else:
            for i in range(1, b_ind.shape[0]):
                g_type_idx = torch.argmax(b_type[i-1], dim=0)
                g_type = GARMENT_TYPE[g_type_idx]
                start, end = b_ind[i-1], b_ind[i]
                g_error = torch.mean(b_error[start:end])
                acc_dict[g_type].append(g_error)
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    if reduction != 'sum' and not overwrite_align:
        # Align the length of keys
        for g_type in GARMENT_TYPE:
            key = f"{prefix}.{g_type}"
            if key not in acc_dict.keys():
                acc_dict[key] = torch.zeros(1).cuda()

    return acc_dict

def accuracy_l2(pred, label, mask_weights=None, prefix='', merge=True, **kwargs):
    '''
        pred: list[n_cam, 1*image]
        label: list[n_cam, 1*image]
    '''
    bs = len(pred)
    # mask_weights = kwargs.pop('mask_weights', None)
    # if mask_weights is not None:
    #     mask_weights = [mask.permute(1, 2, 0) for mask in mask_weights]

    acc_dict = defaultdict(list)
    # element-wise losses
    if mask_weights is not None:
        l2_error = [
            torch.sqrt(torch.sum(
                F.mse_loss(pred[i], label[i], reduction='none') * mask_weights[i], 
                dim=-1)) 
            for i in range(bs)]
    else:
        l2_error = [
            torch.sqrt(torch.sum(
                F.mse_loss(pred[i], label[i], reduction='none'), 
                dim=-1)) 
            for i in range(bs)]

    # Calculate per outfit error
    for cam_id in range(bs):
        acc_dict[cam_id].append(l2_error[cam_id])
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }

    if not merge:
        return acc_dict

    acc_dict = {
        f"{prefix}.acc": torch.mean(torch.stack(list(acc_dict.values())))
    }
    return acc_dict

def accuracy_l1(pred, label, mask_weights=None, prefix='', merge=True, **kwargs):
    '''
        pred: list[n_cam, 1*image]
        label: list[n_cam, 1*image]
    '''
    bs = len(pred)
    # mask_weights = kwargs.pop('mask_weights', None)
    # if mask_weights is not None:
    #     mask_weights = [mask.permute(1, 2, 0) for mask in mask_weights]
    acc_dict = defaultdict(list)
    # element-wise losses
    if mask_weights is not None:
        l2_error = [
            torch.sum(
                F.l1_loss(pred[i], label[i], reduction='none') * mask_weights[i], 
                dim=-1)
            for i in range(bs)]
    else:
        l2_error = [
        torch.sum(
            F.l1_loss(pred[i], label[i], reduction='none'), 
            dim=-1)
        for i in range(bs)]

    # Calculate per outfit error
    for cam_id in range(bs):
        acc_dict[cam_id].append(l2_error[cam_id])
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }

    if not merge:
        return acc_dict

    acc_dict = {
        f"{prefix}.acc": torch.mean(torch.stack(list(acc_dict.values())))
    }
    return acc_dict


@ACCURACY.register_module()
class MSEAccuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2',
                 ratio=False):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(MSEAccuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name
        self.ratio = ratio

    def forward(self, pred, target, indices, indices_type=None, overwrite_align=False, vert_mask=None, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_mse(pred, target, indices, indices_type=indices_type, prefix=self.acc_name, reduction=self.reduction, overwrite_align=overwrite_align, vert_mask=vert_mask, ratio=self.ratio, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name

@ACCURACY.register_module()
class L2Accuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(L2Accuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name

    def forward(self, pred, target, mask_weights=None, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_l2(pred, target, mask_weights=mask_weights, prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name

@ACCURACY.register_module()
class L1Accuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l1'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(L1Accuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name

    def forward(self, pred, target, mask_weights=None, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_l1(pred, target, mask_weights=mask_weights, prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name

def accuracy_ssim(pred, label, kernel_size=5, max_val=1.0, mask_weights=None, prefix='', merge=True, **kwargs):
    '''
        Calculate SSIM accuracy (higher is better)
        pred: list[n_cam, H*W*C]
        label: list[n_cam, H*W*C]
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    
    # element-wise SSIM calculation
    ssim_values = []
    for i in range(bs):
        # Calculate SSIM loss (returns 1 - SSIM)
        ssim_val = ssim_loss(
            pred[i].permute(2, 0, 1).unsqueeze(0), 
            label[i].permute(2, 0, 1).unsqueeze(0), 
            window_size=kernel_size, 
            max_val=max_val, 
            reduction='mean')
        # Convert to actual SSIM value (1 - loss)
        actual_ssim = 1.0 - ssim_val
        ssim_values.append(actual_ssim)
    
    # Calculate per camera
    for cam_id in range(bs):
        acc_dict[cam_id].append(ssim_values[cam_id])
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    
    if not merge:
        return acc_dict
    
    acc_dict = {
        f"{prefix}.acc": torch.mean(torch.stack(list(acc_dict.values())))
    }
    return acc_dict


def accuracy_psnr(pred, label, max_val=1.0, mask_weights=None, prefix='', merge=True, eps=1e-8, **kwargs):
    '''
        Calculate PSNR accuracy (higher is better)
        pred: list[n_cam, H*W*C]
        label: list[n_cam, H*W*C]
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    
    # element-wise PSNR calculation
    psnr_values = []
    for i in range(bs):
        # Calculate MSE
        mse = torch.mean((pred[i] - label[i]) ** 2)
        # Add small epsilon to avoid log(0)
        mse = torch.clamp(mse, min=eps)
        # Calculate PSNR
        psnr = 10 * torch.log10(max_val ** 2 / mse)
        psnr_values.append(psnr)
    
    # Calculate per camera
    for cam_id in range(bs):
        acc_dict[cam_id].append(psnr_values[cam_id])
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    
    if not merge:
        return acc_dict
    
    acc_dict = {
        f"{prefix}.acc": torch.mean(torch.stack(list(acc_dict.values())))
    }
    return acc_dict


@ACCURACY.register_module()
class SSIMAccuracy(nn.Module):

    def __init__(self,
                 kernel_size=5,
                 max_val=1.0,
                 reduction='mean',
                 acc_name='accuracy_ssim'):
        """Module to calculate SSIM accuracy.
        Args:
            kernel_size (int): Size of the kernel for SSIM calculation. Defaults to 5.
            max_val (float): Maximum value of the image. Defaults to 1.0.
            reduction (str): The method used to reduce the accuracy. Defaults to 'mean'.
            acc_name (str): Name of the accuracy metric. Defaults to 'accuracy_ssim'.
        """
        super(SSIMAccuracy, self).__init__()
        self.kernel_size = kernel_size
        self.max_val = max_val
        self.reduction = reduction
        self._acc_name = acc_name

    def forward(self, pred, target, mask_weights=None, **kwargs):
        """Forward function to calculate SSIM accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
            mask_weights (torch.Tensor, optional): Mask weights.
        Returns:
            dict: The SSIM accuracy values.
        """
        return accuracy_ssim(pred, target, kernel_size=self.kernel_size, 
                            max_val=self.max_val, mask_weights=mask_weights, 
                            prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        """Accuracy Name.
        This function must be implemented and will return the name of this
        accuracy metric. This name will be used to combine different accuracy items.
        Returns:
            str: The name of this accuracy item.
        """
        return self._acc_name


@ACCURACY.register_module()
class PSNRAccuracy(nn.Module):

    def __init__(self,
                 max_val=1.0,
                 reduction='mean',
                 acc_name='accuracy_psnr',
                 eps=1e-8):
        """Module to calculate PSNR accuracy.
        Args:
            max_val (float): Maximum value of the image. Defaults to 1.0.
            reduction (str): The method used to reduce the accuracy. Defaults to 'mean'.
            acc_name (str): Name of the accuracy metric. Defaults to 'accuracy_psnr'.
            eps (float): Small value to avoid division by zero. Defaults to 1e-8.
        """
        super(PSNRAccuracy, self).__init__()
        self.max_val = max_val
        self.reduction = reduction
        self.eps = eps
        self._acc_name = acc_name

    def forward(self, pred, target, mask_weights=None, **kwargs):
        """Forward function to calculate PSNR accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
            mask_weights (torch.Tensor, optional): Mask weights.
        Returns:
            dict: The PSNR accuracy values.
        """
        return accuracy_psnr(pred, target, max_val=self.max_val, 
                            mask_weights=mask_weights, prefix=self.acc_name, 
                            eps=self.eps, **kwargs)
    
    @property
    def acc_name(self):
        """Accuracy Name.
        This function must be implemented and will return the name of this
        accuracy metric. This name will be used to combine different accuracy items.
        Returns:
            str: The name of this accuracy item.
        """
        return self._acc_name