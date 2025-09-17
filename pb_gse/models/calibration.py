"""
Calibration methods for improving probability estimates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import log_loss


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration"""
    
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        
    def forward(self, logits):
        return logits / self.temperature
    
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, 
                 lr: float = 0.01, max_iter: int = 50):
        """Calibrate temperature using validation set"""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        
        # Ensure temperature is positive
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)


class GroupTemperatureScaling(nn.Module):
    """Temperature scaling per group (head/tail)"""
    
    def __init__(self, num_groups: int = 2, initial_temperature: float = 1.0):
        super().__init__()
        self.num_groups = num_groups
        self.temperatures = nn.Parameter(torch.ones(num_groups) * initial_temperature)
        
    def forward(self, logits, group_ids):
        """Apply group-specific temperature scaling"""
        batch_size = logits.size(0)
        calibrated_logits = torch.zeros_like(logits)
        
        for group_id in range(self.num_groups):
            mask = (group_ids == group_id)
            if mask.any():
                calibrated_logits[mask] = logits[mask] / self.temperatures[group_id]
                
        return calibrated_logits
    
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, 
                 group_ids: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """Calibrate temperatures for each group"""
        for group_id in range(self.num_groups):
            mask = (group_ids == group_id)
            if not mask.any():
                continue
                
            group_logits = logits[mask]
            group_labels = labels[mask]
            
            # Create temporary temperature parameter for this group
            temp_param = nn.Parameter(torch.ones(1) * self.temperatures[group_id].item())
            optimizer = torch.optim.LBFGS([temp_param], lr=lr, max_iter=max_iter)
            
            def eval_loss():
                optimizer.zero_grad()
                scaled_logits = group_logits / temp_param
                loss = F.cross_entropy(scaled_logits, group_labels)
                loss.backward()
                return loss
                
            optimizer.step(eval_loss)
            
            # Update group temperature
            with torch.no_grad():
                temp_param.clamp_(min=1e-3)
                self.temperatures[group_id] = temp_param.item()


class VectorScaling(nn.Module):
    """Vector scaling (Platt scaling) for calibration"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))
        
    def forward(self, logits):
        return logits * self.W + self.b
    
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor,
                 lr: float = 0.01, max_iter: int = 50):
        """Calibrate using validation set"""
        optimizer = torch.optim.LBFGS([self.W, self.b], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)


class GroupVectorScaling(nn.Module):
    """Vector scaling per group"""
    
    def __init__(self, num_classes: int, num_groups: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        
        # Parameters for each group
        self.W = nn.Parameter(torch.ones(num_groups, num_classes))
        self.b = nn.Parameter(torch.zeros(num_groups, num_classes))
        
    def forward(self, logits, group_ids):
        """Apply group-specific vector scaling"""
        batch_size = logits.size(0)
        calibrated_logits = torch.zeros_like(logits)
        
        for group_id in range(self.num_groups):
            mask = (group_ids == group_id)
            if mask.any():
                W_g = self.W[group_id]
                b_g = self.b[group_id]
                calibrated_logits[mask] = logits[mask] * W_g + b_g
                
        return calibrated_logits
    
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor,
                 group_ids: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """Calibrate parameters for each group"""
        for group_id in range(self.num_groups):
            mask = (group_ids == group_id)
            if not mask.any():
                continue
                
            group_logits = logits[mask]
            group_labels = labels[mask]
            
            # Create temporary parameters for this group
            W_param = nn.Parameter(self.W[group_id].clone())
            b_param = nn.Parameter(self.b[group_id].clone())
            
            optimizer = torch.optim.LBFGS([W_param, b_param], lr=lr, max_iter=max_iter)
            
            def eval_loss():
                optimizer.zero_grad()
                scaled_logits = group_logits * W_param + b_param
                loss = F.cross_entropy(scaled_logits, group_labels)
                loss.backward()
                return loss
                
            optimizer.step(eval_loss)
            
            # Update group parameters
            with torch.no_grad():
                self.W[group_id] = W_param.data
                self.b[group_id] = b_param.data


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, 
                             n_bins: int = 15) -> float:
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def group_expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                   group_ids: np.ndarray, n_bins: int = 15) -> Dict[int, float]:
    """Calculate ECE for each group"""
    group_eces = {}
    unique_groups = np.unique(group_ids)
    
    for group_id in unique_groups:
        mask = (group_ids == group_id)
        if mask.sum() > 0:
            group_probs = probs[mask]
            group_labels = labels[mask]
            group_eces[group_id] = expected_calibration_error(group_probs, group_labels, n_bins)
        else:
            group_eces[group_id] = 0.0
            
    return group_eces


class ModelCalibrator:
    """Wrapper for model calibration"""
    
    def __init__(self, model: nn.Module, num_classes: int, num_groups: int = 2,
                 method: str = 'temperature', group_aware: bool = True):
        self.model = model
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.method = method
        self.group_aware = group_aware
        
        # Initialize calibration method
        if method == 'temperature':
            if group_aware:
                self.calibrator = GroupTemperatureScaling(num_groups)
            else:
                self.calibrator = TemperatureScaling()
        elif method == 'vector':
            if group_aware:
                self.calibrator = GroupVectorScaling(num_classes, num_groups)
            else:
                self.calibrator = VectorScaling(num_classes)
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
    
    def calibrate(self, dataloader, group_info: Dict, device: str = 'cuda'):
        """Calibrate model using validation data"""
        self.model.eval()
        self.calibrator.train()
        
        all_logits = []
        all_labels = []
        all_group_ids = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = self.model(inputs)
                
                # Get group IDs for targets
                group_ids = torch.tensor([
                    group_info['class_to_group'][target.item()] 
                    for target in targets
                ]).to(device)
                
                all_logits.append(logits)
                all_labels.append(targets)
                all_group_ids.append(group_ids)
        
        # Concatenate all data
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_group_ids = torch.cat(all_group_ids, dim=0)
        
        # Calibrate
        if self.group_aware:
            self.calibrator.calibrate(all_logits, all_labels, all_group_ids)
        else:
            self.calibrator.calibrate(all_logits, all_labels)
    
    def get_calibrated_probs(self, inputs: torch.Tensor, 
                           group_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get calibrated probabilities"""
        self.model.eval()
        self.calibrator.eval()
        
        with torch.no_grad():
            logits = self.model(inputs)
            
            if self.group_aware and group_ids is not None:
                calibrated_logits = self.calibrator(logits, group_ids)
            else:
                calibrated_logits = self.calibrator(logits)
                
            return F.softmax(calibrated_logits, dim=1)
    
    def save_calibrator(self, path: str):
        """Save calibration parameters"""
        torch.save(self.calibrator.state_dict(), path)
    
    def load_calibrator(self, path: str):
        """Load calibration parameters"""
        self.calibrator.load_state_dict(torch.load(path))
