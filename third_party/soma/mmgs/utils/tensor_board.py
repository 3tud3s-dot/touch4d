import os
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import numpy as np

class TensorBoardLogger:
    def __init__(self, log_dir, enabled=True):
        self.enabled = enabled
        if self.enabled:
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            self.step_counters = defaultdict(int)
        
    def log_scalar(self, tag, value, step=None, category='train'):
        """记录标量值"""
        if not self.enabled:
            return
            
        if step is None:
            step = self.step_counters[f'{category}_{tag}']
            self.step_counters[f'{category}_{tag}'] += 1
        
        full_tag = f'{category}/{tag}'
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        self.writer.add_scalar(full_tag, value, step)
    
    def log_losses(self, losses_dict, step=None, category='train'):
        """批量记录损失"""
        if not self.enabled:
            return
            
        for key, value in losses_dict.items():
            if isinstance(value, (int, float, torch.Tensor)):
                self.log_scalar(key, value, step, category)
    
    def log_histogram(self, tag, values, step=None, category='train'):
        """记录直方图"""
        if not self.enabled:
            return
            
        if step is None:
            step = self.step_counters[f'{category}_{tag}']
            self.step_counters[f'{category}_{tag}'] += 1
        
        full_tag = f'{category}/{tag}'
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(full_tag, values, step)
    
    def log_gradients(self, model, step=None):
        """记录梯度信息"""
        if not self.enabled:
            return
            
        if step is None:
            step = self.step_counters['gradients']
            self.step_counters['gradients'] += 1
        
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 记录每层的梯度范数
                self.writer.add_scalar(f'gradients/{name}', param_norm, step)
        
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('gradients/total_norm', total_norm, step)
        self.writer.add_scalar('gradients/param_count', param_count, step)
    
    def log_learning_rate(self, optimizer, step=None):
        """记录学习率"""
        if not self.enabled:
            return
            
        if step is None:
            step = self.step_counters['lr']
            self.step_counters['lr'] += 1
        
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)
    
    def close(self):
        """关闭writer"""
        if self.enabled and hasattr(self, 'writer'):
            self.writer.close()
    
    def __del__(self):
        self.close()