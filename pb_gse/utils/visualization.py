"""
Visualization utilities for PB-GSE
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


def plot_class_distribution(class_counts: List[int], save_path: Optional[str] = None):
    """Plot class distribution for long-tail dataset"""
    
    plt.figure(figsize=(12, 6))
    
    classes = range(len(class_counts))
    
    plt.subplot(1, 2, 1)
    plt.bar(classes, class_counts)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution (Linear Scale)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(classes, class_counts)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples (Log Scale)')
    plt.title('Class Distribution (Log Scale)')
    plt.yscale('log')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_risk_coverage_curves(methods_data: Dict[str, Tuple[List[float], List[float]]],
                             save_path: Optional[str] = None):
    """Plot risk-coverage curves for multiple methods"""
    
    plt.figure(figsize=(10, 6))
    
    for method_name, (coverages, risks) in methods_data.items():
        plt.plot(coverages, risks, label=method_name, linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Coverage')
    plt.ylabel('Risk (Error Rate)')
    plt.title('Risk-Coverage Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_group_metrics(group_metrics: Dict[str, List[float]], group_names: List[str],
                      save_path: Optional[str] = None):
    """Plot group-wise metrics comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['group_errors', 'group_coverage', 'group_ece', 'group_acceptance_rates']
    titles = ['Group Error Rates', 'Group Coverage', 'Group ECE', 'Group Acceptance Rates']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        if metric in group_metrics and idx < len(axes):
            values = group_metrics[metric]
            
            bars = axes[idx].bar(group_names, values, 
                               color=['skyblue', 'lightcoral'][:len(values)])
            axes[idx].set_title(title)
            axes[idx].set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_calibration_reliability(probs: torch.Tensor, targets: torch.Tensor, 
                                group_ids: torch.Tensor, group_names: List[str],
                                n_bins: int = 15, save_path: Optional[str] = None):
    """Plot calibration reliability diagrams"""
    
    fig, axes = plt.subplots(1, len(group_names), figsize=(5*len(group_names), 5))
    if len(group_names) == 1:
        axes = [axes]
    
    for group_id, (group_name, ax) in enumerate(zip(group_names, axes)):
        # Filter data for this group
        mask = (group_ids == group_id)
        if not mask.any():
            continue
            
        group_probs = probs[mask]
        group_targets = targets[mask]
        
        # Get confidence and predictions
        confidences, predictions = torch.max(group_probs, dim=1)
        accuracies = (predictions == group_targets).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                bin_accuracies.append(accuracy_in_bin.item())
                bin_confidences.append(avg_confidence_in_bin.item())
                bin_counts.append(in_bin.sum().item())
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # Plot reliability diagram
        ax.bar(bin_confidences, bin_accuracies, width=1.0/n_bins, alpha=0.7, 
               edgecolor='black', label='Accuracy')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Calibration - {group_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(metrics_history: List[Dict], save_path: Optional[str] = None):
    """Plot training curves"""
    
    # Extract metrics by split
    train_metrics = [m for m in metrics_history if m['split'] == 'train']
    val_metrics = [m for m in metrics_history if m['split'] == 'val']
    
    if not train_metrics or not val_metrics:
        print("No training history found")
        return
    
    # Get epochs and losses
    train_epochs = [m['epoch'] for m in train_metrics]
    train_losses = [m['metrics'].get('loss', 0) for m in train_metrics]
    val_epochs = [m['epoch'] for m in val_metrics]
    val_losses = [m['metrics'].get('loss', 0) for m in val_metrics]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(val_epochs, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy if available
    train_accs = [m['metrics'].get('accuracy', 0) for m in train_metrics]
    val_accs = [m['metrics'].get('accuracy', 0) for m in val_metrics]
    
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs, train_accs, label='Train Acc', marker='o')
    plt.plot(val_epochs, val_accs, label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor,
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
    """Plot confusion matrix"""
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(target_np, pred_np)
    
    plt.figure(figsize=(10, 8))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_results_dashboard(results: Dict, save_path: Optional[str] = None):
    """Create a comprehensive results dashboard"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main metrics
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['coverage', 'balanced_selective_error', 'worst_group_selective_error', 'aurc']
    values = [results.get(m, 0) for m in metrics]
    
    bars = ax1.bar(range(len(metrics)), values, color=['green', 'red', 'orange', 'blue'])
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
    ax1.set_title('Key Metrics')
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Group metrics
    if 'group_errors' in results:
        ax2 = plt.subplot(2, 3, 2)
        group_errors = results['group_errors']
        group_names = ['Head', 'Tail'] if len(group_errors) == 2 else [f'Group {i}' for i in range(len(group_errors))]
        
        bars = ax2.bar(group_names, group_errors, color=['skyblue', 'lightcoral'])
        ax2.set_title('Group Error Rates')
        ax2.set_ylabel('Error Rate')
        
        for bar, value in zip(bars, group_errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Coverage at different levels
    ax3 = plt.subplot(2, 3, 3)
    coverage_levels = [0.7, 0.8, 0.9]
    bse_values = []
    
    for coverage in coverage_levels:
        key = f'metrics_at_{coverage}'
        if key in results:
            bse_values.append(results[key].get('balanced_selective_error', 0))
        else:
            bse_values.append(0)
    
    ax3.plot(coverage_levels, bse_values, marker='o', linewidth=2)
    ax3.set_xlabel('Coverage Level')
    ax3.set_ylabel('Balanced Selective Error')
    ax3.set_title('BSE vs Coverage')
    ax3.grid(True, alpha=0.3)
    
    # Risk-coverage curve (if available)
    if 'risk_coverage_curve' in results:
        ax4 = plt.subplot(2, 3, 4)
        coverages, risks = results['risk_coverage_curve']
        ax4.plot(coverages, risks, linewidth=2)
        ax4.set_xlabel('Coverage')
        ax4.set_ylabel('Risk')
        ax4.set_title('Risk-Coverage Curve')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
