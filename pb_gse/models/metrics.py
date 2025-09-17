"""
Evaluation metrics for PB-GSE
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class SelectiveMetrics:
    """Metrics for selective prediction/classification with abstention"""
    
    def __init__(self, num_groups: int = 2):
        self.num_groups = num_groups
    
    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                          rejections: torch.Tensor, group_ids: torch.Tensor,
                          probs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute all selective prediction metrics"""
        
        results = {}
        
        # Basic metrics
        results.update(self.compute_basic_metrics(predictions, targets, rejections))
        
        # Group-wise metrics
        results.update(self.compute_group_metrics(predictions, targets, rejections, group_ids))
        
        # Coverage-based metrics
        if probs is not None:
            results.update(self.compute_coverage_metrics(predictions, targets, rejections, probs))
            results.update(self.compute_calibration_metrics(predictions, targets, rejections, 
                                                          group_ids, probs))
        
        return results
    
    def compute_basic_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                            rejections: torch.Tensor) -> Dict[str, float]:
        """Compute basic selective prediction metrics"""
        
        accept_mask = (rejections == 0)
        
        # Coverage (acceptance rate)
        coverage = accept_mask.float().mean().item()
        
        # Selective accuracy/error
        if accept_mask.sum() > 0:
            accepted_correct = (predictions[accept_mask] == targets[accept_mask]).float()
            selective_accuracy = accepted_correct.mean().item()
            selective_error = 1.0 - selective_accuracy
        else:
            selective_accuracy = 0.0
            selective_error = 1.0
        
        # Overall accuracy (treating rejections as incorrect)
        overall_correct = accept_mask.float() * (predictions == targets).float()
        overall_accuracy = overall_correct.mean().item()
        
        return {
            'coverage': coverage,
            'rejection_rate': 1.0 - coverage,
            'selective_accuracy': selective_accuracy,
            'selective_error': selective_error,
            'overall_accuracy': overall_accuracy,
            'overall_error': 1.0 - overall_accuracy
        }
    
    def compute_group_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                            rejections: torch.Tensor, group_ids: torch.Tensor) -> Dict[str, float]:
        """Compute group-wise metrics"""
        
        accept_mask = (rejections == 0)
        
        # Group-wise coverage and errors
        group_coverage = []
        group_errors = []
        group_acceptance_rates = []
        
        for group_id in range(self.num_groups):
            group_mask = (group_ids == group_id)
            
            if group_mask.sum() == 0:
                group_coverage.append(0.0)
                group_errors.append(1.0)
                group_acceptance_rates.append(0.0)
                continue
            
            # Group coverage (acceptance rate within group)
            group_accept_mask = group_mask & accept_mask
            group_cov = group_accept_mask.float().sum() / group_mask.float().sum()
            group_coverage.append(group_cov.item())
            
            # Group error (error rate among accepted samples in group)
            if group_accept_mask.sum() > 0:
                group_correct = (predictions[group_accept_mask] == targets[group_accept_mask]).float()
                group_error = 1.0 - group_correct.mean()
                group_errors.append(group_error.item())
            else:
                group_errors.append(1.0)  # No accepted samples
            
            # Acceptance rate (what fraction of accepted samples are from this group)
            if accept_mask.sum() > 0:
                acceptance_rate = group_accept_mask.float().sum() / accept_mask.float().sum()
                group_acceptance_rates.append(acceptance_rate.item())
            else:
                group_acceptance_rates.append(0.0)
        
        # Balanced selective error (BSE)
        balanced_selective_error = np.mean(group_errors)
        
        # Worst-group selective error (WGSE)
        worst_group_selective_error = np.max(group_errors)
        
        # Coverage fairness (standard deviation of group coverage)
        coverage_fairness = np.std(group_coverage)
        
        return {
            'balanced_selective_error': balanced_selective_error,
            'worst_group_selective_error': worst_group_selective_error,
            'group_coverage': group_coverage,
            'group_errors': group_errors,
            'group_acceptance_rates': group_acceptance_rates,
            'coverage_fairness': coverage_fairness
        }
    
    def compute_coverage_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                               rejections: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
        """Compute coverage-based metrics like AURC"""
        
        # Get confidence scores
        confidences = torch.max(probs, dim=1)[0]
        
        # Sort by confidence (descending)
        sorted_indices = torch.argsort(confidences, descending=True)
        sorted_predictions = predictions[sorted_indices]
        sorted_targets = targets[sorted_indices]
        sorted_confidences = confidences[sorted_indices]
        
        # Compute risk-coverage curve
        coverages = []
        risks = []
        
        n_samples = len(sorted_predictions)
        
        for i in range(1, n_samples + 1):
            # Coverage: fraction of samples included
            coverage = i / n_samples
            
            # Risk: error rate among included samples
            included_correct = (sorted_predictions[:i] == sorted_targets[:i]).float()
            risk = 1.0 - included_correct.mean()
            
            coverages.append(coverage)
            risks.append(risk.item())
        
        # Compute AURC (Area Under Risk-Coverage curve)
        aurc = np.trapz(risks, coverages)
        
        return {
            'aurc': aurc,
            'risk_coverage_curve': (coverages, risks)
        }
    
    def compute_calibration_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                                  rejections: torch.Tensor, group_ids: torch.Tensor,
                                  probs: torch.Tensor, n_bins: int = 15) -> Dict[str, float]:
        """Compute calibration metrics (ECE) per group"""
        
        accept_mask = (rejections == 0)
        
        if accept_mask.sum() == 0:
            return {
                'overall_ece': 1.0,
                'group_ece': [1.0] * self.num_groups
            }
        
        # Overall ECE
        accepted_probs = probs[accept_mask]
        accepted_predictions = predictions[accept_mask]
        accepted_targets = targets[accept_mask]
        
        overall_ece = self._expected_calibration_error(
            accepted_probs, accepted_targets, n_bins
        )
        
        # Group-wise ECE
        group_ece = []
        for group_id in range(self.num_groups):
            group_mask = (group_ids == group_id) & accept_mask
            
            if group_mask.sum() > 0:
                group_probs = probs[group_mask]
                group_targets = targets[group_mask]
                ece = self._expected_calibration_error(group_probs, group_targets, n_bins)
                group_ece.append(ece)
            else:
                group_ece.append(0.0)
        
        return {
            'overall_ece': overall_ece,
            'group_ece': group_ece
        }
    
    def _expected_calibration_error(self, probs: torch.Tensor, targets: torch.Tensor,
                                  n_bins: int = 15) -> float:
        """Compute Expected Calibration Error (ECE)"""
        
        # Get confidence scores and predictions
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == targets).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()


def compute_metrics_at_coverage(predictions: torch.Tensor, targets: torch.Tensor,
                              rejections: torch.Tensor, group_ids: torch.Tensor,
                              probs: torch.Tensor, target_coverage: float,
                              num_groups: int = 2) -> Dict[str, float]:
    """Compute metrics at a specific coverage level"""
    
    # Get confidence scores
    confidences = torch.max(probs, dim=1)[0]
    
    # Find threshold for target coverage
    sorted_confidences = torch.sort(confidences, descending=True)[0]
    threshold_idx = int(target_coverage * len(sorted_confidences))
    threshold = sorted_confidences[threshold_idx]
    
    # Create new rejections based on threshold
    new_rejections = (confidences < threshold).long()
    
    # Compute metrics with new rejections
    metrics = SelectiveMetrics(num_groups).compute_all_metrics(
        predictions, targets, new_rejections, group_ids, probs
    )
    
    return metrics


class MetricsLogger:
    """Logger for tracking metrics across experiments"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}
    
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None,
                   split: str = 'train'):
        """Log metrics for a specific epoch and split"""
        
        log_entry = {
            'epoch': epoch,
            'split': split,
            'metrics': metrics.copy()
        }
        
        self.metrics_history.append(log_entry)
        
        if split not in self.current_metrics:
            self.current_metrics[split] = {}
        self.current_metrics[split].update(metrics)
    
    def get_best_metrics(self, metric_name: str, split: str = 'val',
                        maximize: bool = False) -> Tuple[Dict, int]:
        """Get best metrics based on a specific metric"""
        
        relevant_entries = [entry for entry in self.metrics_history 
                          if entry['split'] == split and metric_name in entry['metrics']]
        
        if not relevant_entries:
            return {}, -1
        
        if maximize:
            best_entry = max(relevant_entries, key=lambda x: x['metrics'][metric_name])
        else:
            best_entry = min(relevant_entries, key=lambda x: x['metrics'][metric_name])
        
        return best_entry['metrics'], best_entry['epoch']
    
    def save_metrics(self, save_path: str):
        """Save metrics history"""
        torch.save({
            'metrics_history': self.metrics_history,
            'current_metrics': self.current_metrics
        }, save_path)
    
    def load_metrics(self, load_path: str):
        """Load metrics history"""
        data = torch.load(load_path)
        self.metrics_history = data['metrics_history']
        self.current_metrics = data['current_metrics']


def plot_risk_coverage_curves(risk_coverage_data: List[Tuple[List[float], List[float]]],
                             labels: List[str], save_path: Optional[str] = None):
    """Plot risk-coverage curves for multiple methods"""
    
    plt.figure(figsize=(10, 6))
    
    for (coverages, risks), label in zip(risk_coverage_data, labels):
        plt.plot(coverages, risks, label=label, linewidth=2)
    
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    plt.title('Risk-Coverage Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_metrics_table(results_dict: Dict[str, Dict[str, float]], 
                        coverage_levels: List[float] = [0.7, 0.8, 0.9]) -> str:
    """Create a formatted table of metrics for different methods"""
    
    # Define key metrics to include
    key_metrics = [
        'balanced_selective_error',
        'worst_group_selective_error', 
        'aurc',
        'overall_ece'
    ]
    
    # Create header
    header = "Method".ljust(20)
    for coverage in coverage_levels:
        header += f"BSE@{coverage}".ljust(12) + f"WGSE@{coverage}".ljust(12)
    header += "AURC".ljust(12) + "ECE".ljust(12)
    
    table = header + "\n" + "-" * len(header) + "\n"
    
    # Add rows for each method
    for method_name, method_results in results_dict.items():
        row = method_name.ljust(20)
        
        # Add coverage-specific metrics
        for coverage in coverage_levels:
            if f'metrics_at_{coverage}' in method_results:
                metrics = method_results[f'metrics_at_{coverage}']
                bse = f"{metrics.get('balanced_selective_error', 0.0):.3f}"
                wgse = f"{metrics.get('worst_group_selective_error', 0.0):.3f}"
                row += bse.ljust(12) + wgse.ljust(12)
            else:
                row += "N/A".ljust(12) + "N/A".ljust(12)
        
        # Add overall metrics
        aurc = f"{method_results.get('aurc', 0.0):.3f}"
        ece = f"{method_results.get('overall_ece', 0.0):.3f}"
        row += aurc.ljust(12) + ece.ljust(12)
        
        table += row + "\n"
    
    return table
