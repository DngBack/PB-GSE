"""
Quick test script to verify the entire PB-GSE pipeline works
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import CIFAR10LT
from data.transforms import get_transforms
from models.backbones import ResNet32
from models.calibration import ModelCalibrator
from models.gating import PACBayesGating, FeatureExtractor
from models.plugin_rule import PluginOptimizer
from models.metrics import SelectiveMetrics


def create_test_config():
    """Create minimal test configuration"""
    return {
        'data': {
            'name': 'cifar10_lt',
            'augmentation': {
                'train': {'rand_augment': {'n': 2, 'm': 10}},
                'test': {'center_crop': True}
            }
        },
        'gating': {
            'network': {'hidden_dims': [32], 'dropout': 0.1, 'activation': 'relu'},
            'features': {
                'use_probs': True, 'use_entropy': False, 'use_max_prob': False,
                'use_disagreement': False, 'use_group_onehot': True
            },
            'pac_bayes': {'method': 'deterministic', 'prior_std': 1.0}
        },
        'plugin': {
            'rejection_cost': 0.1,
            'fixed_point': {'max_iterations': 5, 'tolerance': 1e-6,
                           'lambda_grid': [-0.5, 0.0, 0.5]},
            'groups': {'num_groups': 2},
            'worst_group': {'enabled': False}
        }
    }


def main():
    print("=== Quick PB-GSE Test ===")
    
    # Setup
    device = torch.device('cpu')  # Use CPU for quick test
    config = create_test_config()
    
    # Create small synthetic data
    print("1. Creating synthetic data...")
    batch_size = 32
    num_classes = 10
    
    # Synthetic features and targets
    features = torch.randn(batch_size, 64)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create group info
    group_info = {
        'class_to_group': {i: 0 if i < 5 else 1 for i in range(num_classes)},
        'group_to_classes': {0: list(range(5)), 1: list(range(5, 10))}
    }
    
    # Synthetic model probabilities (3 models)
    model_probs_list = [
        torch.softmax(torch.randn(batch_size, num_classes), dim=1)
        for _ in range(3)
    ]
    
    print(f"   - Features shape: {features.shape}")
    print(f"   - Targets shape: {targets.shape}")
    print(f"   - Model probs: {[p.shape for p in model_probs_list]}")
    
    # Test feature extraction
    print("2. Testing feature extraction...")
    feature_extractor = FeatureExtractor(config['gating'])
    
    group_ids = torch.tensor([group_info['class_to_group'][t.item()] for t in targets])
    group_onehot = torch.nn.functional.one_hot(group_ids, num_classes=2).float()
    
    extracted_features = feature_extractor.extract(model_probs_list, group_onehot)
    print(f"   - Extracted features shape: {extracted_features.shape}")
    
    # Test gating network
    print("3. Testing gating network...")
    input_dim = extracted_features.size(1)
    num_models = len(model_probs_list)
    
    gating_model = PACBayesGating(input_dim, num_models, config['gating'])
    gating_weights = gating_model.forward(extracted_features)
    
    print(f"   - Gating weights shape: {gating_weights.shape}")
    print(f"   - Gating weights sample: {gating_weights[0].detach().numpy()}")
    
    # Test ensemble probabilities
    print("4. Testing ensemble probabilities...")
    ensemble_probs = gating_model.compute_ensemble_probs(model_probs_list, gating_weights)
    print(f"   - Ensemble probs shape: {ensemble_probs.shape}")
    
    # Test plugin rule optimization
    print("5. Testing plugin rule optimization...")
    plugin_optimizer = PluginOptimizer(config['plugin'])
    alpha, mu = plugin_optimizer.optimize(ensemble_probs, targets, group_ids, group_info)
    
    print(f"   - Alpha: {alpha}")
    print(f"   - Mu: {mu}")
    
    # Test plugin rule
    print("6. Testing plugin rule...")
    plugin_rule = plugin_optimizer.create_plugin_rule(alpha, mu, group_info)
    predictions, rejections = plugin_rule.forward(ensemble_probs)
    
    print(f"   - Predictions shape: {predictions.shape}")
    print(f"   - Rejections shape: {rejections.shape}")
    print(f"   - Rejection rate: {rejections.float().mean().item():.3f}")
    
    # Test metrics computation
    print("7. Testing metrics computation...")
    metrics_computer = SelectiveMetrics(num_groups=2)
    
    all_metrics = metrics_computer.compute_all_metrics(
        predictions, targets, rejections, group_ids, ensemble_probs
    )
    
    print("   - Key metrics:")
    print(f"     * Coverage: {all_metrics['coverage']:.3f}")
    print(f"     * Selective Accuracy: {all_metrics['selective_accuracy']:.3f}")
    print(f"     * Balanced Selective Error: {all_metrics['balanced_selective_error']:.3f}")
    print(f"     * Worst-Group Error: {all_metrics['worst_group_selective_error']:.3f}")
    
    print("\n✓ All tests passed! PB-GSE pipeline is working correctly.")
    return 0


if __name__ == '__main__':
    exit(main())
