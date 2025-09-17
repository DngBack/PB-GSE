"""
Loss functions for long-tail classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss"""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Loss for long-tail classification"""
    
    def __init__(self, class_frequencies: List[int]):
        super().__init__()
        # Convert to relative frequencies
        total_samples = sum(class_frequencies)
        self.class_priors = torch.tensor([freq / total_samples for freq in class_frequencies])
        
    def forward(self, logits, targets):
        # Add log prior to logits: logits + log(π_y)
        log_priors = torch.log(self.class_priors + 1e-8).to(logits.device)
        adjusted_logits = logits + log_priors
        return F.cross_entropy(adjusted_logits, targets)


class LogitAdjustLoss(nn.Module):
    """Logit Adjustment Loss"""
    
    def __init__(self, class_frequencies: List[int], tau: float = 1.0):
        super().__init__()
        # Calculate logit adjustments
        total_samples = sum(class_frequencies)
        class_priors = torch.tensor([freq / total_samples for freq in class_frequencies])
        self.logit_adjustments = tau * torch.log(class_priors + 1e-8)
        
    def forward(self, logits, targets):
        adjusted_logits = logits + self.logit_adjustments.to(logits.device)
        return F.cross_entropy(adjusted_logits, targets)


class LDAMLoss(nn.Module):
    """Large Margin Loss (LDAM)"""
    
    def __init__(self, class_frequencies: List[int], max_m: float = 0.5, s: float = 30.0):
        super().__init__()
        # Calculate margins: m_y = C / n_y^{1/4}
        class_counts = torch.tensor(class_frequencies, dtype=torch.float32)
        margins = max_m / torch.pow(class_counts, 0.25)
        self.margins = margins
        self.s = s  # scaling factor
        
    def forward(self, logits, targets):
        batch_size = logits.size(0)
        margins = self.margins.to(logits.device)
        
        # Create margin matrix
        margin_matrix = torch.zeros_like(logits)
        for i in range(batch_size):
            margin_matrix[i, targets[i]] = margins[targets[i]]
            
        # Apply margins
        adjusted_logits = logits - margin_matrix
        scaled_logits = self.s * adjusted_logits
        
        return F.cross_entropy(scaled_logits, targets)


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss using effective number of samples"""
    
    def __init__(self, class_frequencies: List[int], beta: float = 0.999, 
                 loss_type: str = 'focal', gamma: float = 2.0):
        super().__init__()
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, class_frequencies)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(class_frequencies)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        self.loss_type = loss_type
        self.gamma = gamma
        
    def forward(self, logits, targets):
        weights = self.weights.to(logits.device)
        
        if self.loss_type == 'focal':
            return self._focal_loss(logits, targets, weights)
        elif self.loss_type == 'ce':
            return F.cross_entropy(logits, targets, weight=weights)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _focal_loss(self, logits, targets, weights):
        ce_loss = F.cross_entropy(logits, targets, weight=weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets).to(logits.device)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()


class DeferredReWeighting:
    """Deferred Re-weighting strategy for LDAM"""
    
    def __init__(self, class_frequencies: List[int], start_epoch: int, 
                 total_epochs: int, drw_epochs: int = None):
        self.class_frequencies = class_frequencies
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs
        self.drw_epochs = drw_epochs or (total_epochs - start_epoch)
        
        # Calculate per-class weights (inverse frequency)
        total_samples = sum(class_frequencies)
        weights = []
        for freq in class_frequencies:
            weights.append(total_samples / (len(class_frequencies) * freq))
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
    def get_weight(self, epoch: int) -> torch.Tensor:
        """Get current weight based on epoch"""
        if epoch < self.start_epoch:
            # No re-weighting before start_epoch
            return torch.ones_like(self.weights)
        else:
            # Linear interpolation from 1.0 to target weight
            progress = (epoch - self.start_epoch) / self.drw_epochs
            progress = min(progress, 1.0)
            current_weights = 1.0 + progress * (self.weights - 1.0)
            return current_weights


def get_loss_function(config: Dict, class_frequencies: List[int], epoch: int = 0) -> nn.Module:
    """Get loss function based on config"""
    model_config = config['base_model']
    loss_name = model_config.get('loss', 'cross_entropy')
    
    if loss_name == 'cross_entropy':
        return CrossEntropyLoss()
    
    elif loss_name == 'balanced_softmax':
        return BalancedSoftmaxLoss(class_frequencies)
    
    elif loss_name == 'logit_adjust':
        return LogitAdjustLoss(class_frequencies)
    
    elif loss_name == 'ldam':
        ldam_config = model_config.get('ldam', {})
        loss_fn = LDAMLoss(
            class_frequencies,
            max_m=ldam_config.get('margin_scale', 0.5)
        )
        
        # Apply DRW if configured
        if 'drw' in model_config:
            drw_config = model_config['drw']
            drw = DeferredReWeighting(
                class_frequencies,
                start_epoch=drw_config['start_epoch'],
                total_epochs=model_config['epochs']
            )
            # Modify loss function to include DRW weights
            original_forward = loss_fn.forward
            
            def forward_with_drw(logits, targets):
                weights = drw.get_weight(epoch).to(logits.device)
                return F.cross_entropy(logits, targets, weight=weights)
            
            loss_fn.forward = forward_with_drw
        
        return loss_fn
    
    elif loss_name == 'cb_focal':
        cb_config = model_config.get('cb', {})
        focal_config = model_config.get('focal', {})
        return ClassBalancedLoss(
            class_frequencies,
            beta=cb_config.get('beta', 0.999),
            loss_type='focal',
            gamma=focal_config.get('gamma', 2.0)
        )
    
    elif loss_name == 'focal':
        focal_config = model_config.get('focal', {})
        alpha = None
        if 'alpha' in focal_config:
            alpha = torch.tensor(focal_config['alpha'])
        return FocalLoss(alpha=alpha, gamma=focal_config.get('gamma', 2.0))
    
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


class MixUpLoss:
    """Loss function for MixUp augmentation"""
    
    def __init__(self, base_loss_fn):
        self.base_loss_fn = base_loss_fn
        
    def __call__(self, logits, y_a, y_b, lam):
        loss_a = self.base_loss_fn(logits, y_a)
        loss_b = self.base_loss_fn(logits, y_b)
        return lam * loss_a + (1 - lam) * loss_b


class CutMixLoss:
    """Loss function for CutMix augmentation"""
    
    def __init__(self, base_loss_fn):
        self.base_loss_fn = base_loss_fn
        
    def __call__(self, logits, y_a, y_b, lam):
        loss_a = self.base_loss_fn(logits, y_a)
        loss_b = self.base_loss_fn(logits, y_b)
        return lam * loss_a + (1 - lam) * loss_b
