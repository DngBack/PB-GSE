"""
Evaluate baseline methods for comparison with PB-GSE
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.metrics import SelectiveMetrics, compute_metrics_at_coverage


def chow_rule_baseline(probs: torch.Tensor, targets: torch.Tensor, 
                      group_ids: torch.Tensor, rejection_cost: float = 0.1):
    """Standard Chow's rule baseline"""
    
    # Get confidence scores
    confidences, predictions = torch.max(probs, dim=1)
    
    # Chow's threshold: 1 - c
    threshold = 1.0 - rejection_cost
    rejections = (confidences < threshold).long()
    
    return predictions, rejections


def balanced_chow_baseline(probs: torch.Tensor, targets: torch.Tensor,
                          group_ids: torch.Tensor, rejection_cost: float = 0.1,
                          num_groups: int = 2):
    """Balanced Chow's rule (group-specific thresholds)"""
    
    confidences, predictions = torch.max(probs, dim=1)
    rejections = torch.zeros_like(predictions)
    
    # Compute group-specific thresholds
    for group_id in range(num_groups):
        group_mask = (group_ids == group_id)
        if not group_mask.any():
            continue
        
        group_confidences = confidences[group_mask]
        
        # Use quantile-based threshold to achieve target coverage
        target_coverage = 1.0 - rejection_cost
        threshold = torch.quantile(group_confidences, 1.0 - target_coverage)
        
        group_rejections = (group_confidences < threshold).long()
        rejections[group_mask] = group_rejections
    
    return predictions, rejections


def deep_ensemble_baseline(model_probs_list: list, targets: torch.Tensor,
                          group_ids: torch.Tensor, rejection_cost: float = 0.1):
    """Deep ensemble with uniform averaging + Chow's rule"""
    
    # Uniform ensemble
    ensemble_probs = torch.stack(model_probs_list, dim=0).mean(dim=0)
    
    # Apply Chow's rule
    return chow_rule_baseline(ensemble_probs, targets, group_ids, rejection_cost)


def conformal_prediction_baseline(cal_probs: torch.Tensor, cal_targets: torch.Tensor,
                                 test_probs: torch.Tensor, test_targets: torch.Tensor,
                                 test_group_ids: torch.Tensor, alpha: float = 0.1):
    """Conformal prediction baseline (simplified)"""
    
    # Compute conformity scores on calibration set
    cal_confidences = torch.max(cal_probs, dim=1)[0]
    cal_predictions = torch.argmax(cal_probs, dim=1)
    cal_correct = (cal_predictions == cal_targets).float()
    
    # Conformity score: 1 - confidence for incorrect predictions, confidence for correct
    conformity_scores = torch.where(cal_correct == 1, cal_confidences, 1 - cal_confidences)
    
    # Compute threshold
    n_cal = len(conformity_scores)
    threshold_idx = int(np.ceil((n_cal + 1) * (1 - alpha)))
    threshold = torch.sort(conformity_scores)[0][min(threshold_idx, n_cal - 1)]
    
    # Apply to test set
    test_confidences = torch.max(test_probs, dim=1)[0]
    test_predictions = torch.argmax(test_probs, dim=1)
    
    # Reject if conformity score is too low
    test_conformity = test_confidences  # Simplified
    rejections = (test_conformity < threshold).long()
    
    return test_predictions, rejections


def evaluate_baselines(config: dict, model_probs_dir: str, output_dir: str):
    """Evaluate all baseline methods"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model probabilities
    model_names = ['cRT', 'LDAM_DRW', 'CB_Focal']
    
    # Load calibration and test data
    cal_probs_list = []
    test_probs_list = []
    
    for model_name in model_names:
        # Calibration data
        cal_path = os.path.join(model_probs_dir, model_name, 'cal.pth')
        cal_data = torch.load(cal_path)
        cal_probs_list.append(cal_data['probs'])
        
        # Test data
        test_path = os.path.join(model_probs_dir, model_name, 'test.pth')
        test_data = torch.load(test_path)
        test_probs_list.append(test_data['probs'])
    
    # Get targets and group info
    cal_targets = cal_data['targets']
    test_targets = test_data['targets']
    
    # Load group info
    from data.datasets import load_group_info
    group_info = load_group_info(os.path.join(config['data']['root'], 'group_info.json'))
    
    # Compute group IDs
    cal_group_ids = torch.tensor([
        group_info['class_to_group'][target.item()] for target in cal_targets
    ])
    test_group_ids = torch.tensor([
        group_info['class_to_group'][target.item()] for target in test_targets
    ])
    
    # Move to device
    cal_probs_list = [p.to(device) for p in cal_probs_list]
    test_probs_list = [p.to(device) for p in test_probs_list]
    cal_targets = cal_targets.to(device)
    test_targets = test_targets.to(device)
    cal_group_ids = cal_group_ids.to(device)
    test_group_ids = test_group_ids.to(device)
    
    rejection_cost = config['plugin']['rejection_cost']
    num_groups = config['plugin']['groups']['num_groups']
    
    # Initialize metrics computer
    metrics_computer = SelectiveMetrics(num_groups)
    
    results = {}
    
    # Baseline 1: Single model + Chow's rule
    print("Evaluating Single Model + Chow's Rule...")
    best_single_probs = test_probs_list[0]  # Use first model
    predictions, rejections = chow_rule_baseline(
        best_single_probs, test_targets, test_group_ids, rejection_cost
    )
    
    metrics = metrics_computer.compute_all_metrics(
        predictions, test_targets, rejections, test_group_ids, best_single_probs
    )
    results['single_chow'] = metrics
    
    # Baseline 2: Deep Ensemble + Chow's rule
    print("Evaluating Deep Ensemble + Chow's Rule...")
    predictions, rejections = deep_ensemble_baseline(
        test_probs_list, test_targets, test_group_ids, rejection_cost
    )
    
    ensemble_probs = torch.stack(test_probs_list, dim=0).mean(dim=0)
    metrics = metrics_computer.compute_all_metrics(
        predictions, test_targets, rejections, test_group_ids, ensemble_probs
    )
    results['ensemble_chow'] = metrics
    
    # Baseline 3: Balanced Chow's rule
    print("Evaluating Balanced Chow's Rule...")
    predictions, rejections = balanced_chow_baseline(
        ensemble_probs, test_targets, test_group_ids, rejection_cost, num_groups
    )
    
    metrics = metrics_computer.compute_all_metrics(
        predictions, test_targets, rejections, test_group_ids, ensemble_probs
    )
    results['balanced_chow'] = metrics
    
    # Baseline 4: Conformal Prediction (simplified)
    print("Evaluating Conformal Prediction...")
    cal_ensemble_probs = torch.stack(cal_probs_list, dim=0).mean(dim=0)
    predictions, rejections = conformal_prediction_baseline(
        cal_ensemble_probs, cal_targets, ensemble_probs, test_targets, test_group_ids
    )
    
    metrics = metrics_computer.compute_all_metrics(
        predictions, test_targets, rejections, test_group_ids, ensemble_probs
    )
    results['conformal'] = metrics
    
    # Compute coverage-specific metrics for all methods
    coverage_levels = config['plugin']['coverage_levels']
    
    for method_name, method_results in results.items():
        for coverage in coverage_levels:
            coverage_metrics = compute_metrics_at_coverage(
                predictions, test_targets, rejections, test_group_ids,
                ensemble_probs, coverage, num_groups
            )
            method_results[f'metrics_at_{coverage}'] = coverage_metrics
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    json_results = {}
    for method_name, method_metrics in results.items():
        json_results[method_name] = {}
        for key, value in method_metrics.items():
            if isinstance(value, torch.Tensor):
                json_results[method_name][key] = value.cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                json_results[method_name][key] = [v.cpu().numpy().tolist() for v in value]
            else:
                json_results[method_name][key] = value
    
    # Save to file
    results_path = os.path.join(output_dir, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print comparison
    print("\n=== Baseline Results Comparison ===")
    print(f"{'Method':<20} {'Coverage':<10} {'BSE':<10} {'WGSE':<10} {'AURC':<10}")
    print("-" * 60)
    
    for method_name, method_metrics in results.items():
        coverage = method_metrics.get('coverage', 0.0)
        bse = method_metrics.get('balanced_selective_error', 1.0)
        wgse = method_metrics.get('worst_group_selective_error', 1.0)
        aurc = method_metrics.get('aurc', 1.0)
        
        print(f"{method_name:<20} {coverage:<10.3f} {bse:<10.3f} {wgse:<10.3f} {aurc:<10.3f}")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline methods')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--probs_dir', type=str, required=True, help='Model probabilities directory')
    parser.add_argument('--output_dir', type=str, default='./baseline_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate baselines
    results = evaluate_baselines(config, args.probs_dir, args.output_dir)
    
    print("Baseline evaluation completed!")
    return 0


if __name__ == '__main__':
    exit(main())
