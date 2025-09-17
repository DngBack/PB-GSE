"""
Demo script for PB-GSE on CIFAR-10-LT
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import CIFAR10LT, DataSplitter
from data.transforms import get_transforms
from models.backbones import ResNet32
from models.losses_lt import CrossEntropyLoss
from models.calibration import ModelCalibrator
from models.gating import PACBayesGating, FeatureExtractor
from models.plugin_rule import PluginOptimizer
from models.inference import PBGSEInference
from models.metrics import SelectiveMetrics


def create_demo_config():
    """Create a minimal demo configuration"""
    config = {
        'data': {
            'name': 'cifar10_lt',
            'root': './data',
            'imbalance_factor': 100,
            'seed': 42,
            'splits': {'train': 0.8, 'cal': 0.1, 'val': 0.05, 'test': 0.05},
            'groups': {'num_groups': 2, 'tail_threshold': 50},
            'augmentation': {
                'train': {'rand_augment': {'n': 2, 'm': 10}, 'mixup': 0.2, 'cutmix': 0.2},
                'test': {'center_crop': True}
            },
            'batch_size': 64,
            'num_workers': 2,
            'pin_memory': True
        },
        'gating': {
            'network': {'hidden_dims': [64, 32], 'dropout': 0.1, 'activation': 'relu'},
            'features': {
                'use_probs': True, 'use_entropy': True, 'use_max_prob': True,
                'use_disagreement': True, 'use_group_onehot': True
            },
            'pac_bayes': {
                'method': 'deterministic', 'prior_std': 1.0, 'posterior_std_init': 0.1
            },
            'epochs': 10, 'lr': 1e-3, 'batch_size': 256
        },
        'plugin': {
            'rejection_cost': 0.1,
            'fixed_point': {'max_iterations': 10, 'tolerance': 1e-6,
                           'lambda_grid': [-1.0, -0.5, 0.0, 0.5, 1.0]},
            'groups': {'num_groups': 2, 'group_names': ['head', 'tail']},
            'worst_group': {'enabled': False},
            'coverage_levels': [0.7, 0.8, 0.9]
        }
    }
    return config


def train_simple_model(dataset, num_classes, device, epochs=5):
    """Train a simple model for demo"""
    model = ResNet32(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model


def get_model_probs(model, dataset, device):
    """Get model probabilities"""
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu())
            all_targets.append(targets)
    
    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def main():
    print("=== PB-GSE Demo on CIFAR-10-LT ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create config
    config = create_demo_config()
    
    # Create dataset
    print("\n1. Creating long-tail dataset...")
    lt_dataset = CIFAR10LT(root='./data', imbalance_factor=100, seed=42)
    
    # Get transforms
    from data.transforms import get_transforms
    transform_train, transform_test = get_transforms(config)
    
    # Get datasets
    train_dataset, test_dataset, group_info = lt_dataset.get_datasets(transform_train, transform_test)
    
    print(f"Group info: {group_info}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Split training set
    train_targets = [train_dataset.dataset[train_dataset.indices[i]][1] 
                    for i in range(len(train_dataset))]
    
    splitter = DataSplitter(train_dataset, train_targets, config['data']['splits'], 42)
    split_datasets = splitter.split()
    
    train_split = split_datasets['train']
    cal_split = split_datasets['cal']
    val_split = split_datasets['val']
    
    print(f"Train split: {len(train_split)}")
    print(f"Cal split: {len(cal_split)}")
    print(f"Val split: {len(val_split)}")
    
    # Train multiple base models (simplified)
    print("\n2. Training base models...")
    num_classes = 10
    
    models = []
    model_names = ['Model1', 'Model2', 'Model3']
    
    for i, name in enumerate(model_names):
        print(f"Training {name}...")
        model = train_simple_model(train_split, num_classes, device, epochs=3)
        models.append(model)
    
    # Get model probabilities
    print("\n3. Getting model probabilities...")
    model_probs_list = []
    
    for i, model in enumerate(models):
        probs, targets = get_model_probs(model, cal_split, device)
        model_probs_list.append(probs)
    
    print(f"Probability shapes: {[p.shape for p in model_probs_list]}")
    
    # Create simple gating features
    print("\n4. Creating gating features...")
    
    # Simple feature extraction (concatenate probabilities)
    features = torch.cat(model_probs_list, dim=1)  # [N, 3*10]
    
    # Add group information
    class_to_group = group_info['class_to_group']
    group_ids = torch.tensor([class_to_group[t.item()] for t in targets])
    group_onehot = torch.nn.functional.one_hot(group_ids, num_classes=2).float()
    
    features = torch.cat([features, group_onehot], dim=1)
    print(f"Feature shape: {features.shape}")
    
    # Create simple gating network
    print("\n5. Training gating network...")
    input_dim = features.size(1)
    num_models = len(models)
    
    gating_model = PACBayesGating(input_dim, num_models, config).to(device)
    
    # Simple gating training (just a few steps)
    optimizer = torch.optim.Adam(gating_model.parameters(), lr=1e-3)
    
    features = features.to(device)
    targets = targets.to(device)
    group_ids = group_ids.to(device)
    
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        gating_weights = gating_model.forward(features)
        
        # Simple loss (encourage uniform weighting for demo)
        uniform_target = torch.ones_like(gating_weights) / num_models
        loss = torch.nn.functional.mse_loss(gating_weights, uniform_target)
        
        loss.backward()
        optimizer.step()
        
        print(f"Gating epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Plugin rule optimization
    print("\n6. Optimizing plugin rule...")
    
    # Get ensemble probabilities
    with torch.no_grad():
        gating_weights = gating_model.forward(features)
        ensemble_probs = gating_model.compute_ensemble_probs(
            [p.to(device) for p in model_probs_list], gating_weights
        )
    
    # Optimize α, μ parameters
    plugin_optimizer = PluginOptimizer(config['plugin'])
    alpha, mu = plugin_optimizer.optimize(ensemble_probs, targets, group_ids, group_info)
    
    print(f"Optimized α: {alpha}")
    print(f"Optimized μ: {mu}")
    
    # Create plugin rule
    plugin_rule = plugin_optimizer.create_plugin_rule(alpha, mu, group_info)
    
    # Make predictions
    print("\n7. Making predictions...")
    predictions, rejections = plugin_rule.forward(ensemble_probs)
    
    # Compute metrics
    print("\n8. Computing metrics...")
    metrics_computer = SelectiveMetrics(num_groups=2)
    
    all_metrics = metrics_computer.compute_all_metrics(
        predictions, targets, rejections, group_ids, ensemble_probs
    )
    
    # Print results
    print("\n=== Results ===")
    print(f"Coverage: {all_metrics['coverage']:.3f}")
    print(f"Selective Accuracy: {all_metrics['selective_accuracy']:.3f}")
    print(f"Balanced Selective Error: {all_metrics['balanced_selective_error']:.3f}")
    print(f"Worst-Group Selective Error: {all_metrics['worst_group_selective_error']:.3f}")
    print(f"Group Coverage: {all_metrics['group_coverage']}")
    print(f"Group Errors: {all_metrics['group_errors']}")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == '__main__':
    main()
