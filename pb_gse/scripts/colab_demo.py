"""
Simple demo for Google Colab
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def create_colab_config():
    """Create configuration for Colab demo"""
    return {
        "data": {
            "name": "cifar10_lt",
            "root": "./data",
            "imbalance_factor": 100,
            "seed": 42,
            "splits": {"train": 0.8, "cal": 0.1, "val": 0.05, "test": 0.05},
            "groups": {"num_groups": 2, "tail_threshold": 50},
            "augmentation": {
                "train": {"rand_augment": {"n": 2, "m": 10}},
                "test": {"center_crop": True},
            },
            "batch_size": 64,
            "num_workers": 2,
            "pin_memory": False,  # Set to False for Colab
        },
        "base_model": {
            "name": "simple_model",
            "backbone": "resnet32",
            "epochs": 3,  # Short training for demo
            "lr": 0.01,
            "optimizer": "sgd",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "loss": "cross_entropy",
        },
        "gating": {
            "network": {"hidden_dims": [32], "dropout": 0.1, "activation": "relu"},
            "features": {
                "use_probs": True,
                "use_entropy": False,
                "use_max_prob": False,
                "use_disagreement": False,
                "use_group_onehot": True,
            },
            "pac_bayes": {"method": "deterministic", "prior_std": 1.0},
            "epochs": 3,
            "lr": 1e-3,
            "batch_size": 64,
        },
        "plugin": {
            "rejection_cost": 0.1,
            "fixed_point": {
                "max_iterations": 5,
                "tolerance": 1e-6,
                "lambda_grid": [-0.5, 0.0, 0.5],
            },
            "groups": {"num_groups": 2},
            "worst_group": {"enabled": False},
        },
    }


def main():
    print("=== PB-GSE Colab Demo ===")

    # Check if running in Colab
    try:
        import google.colab

        print("✓ Running in Google Colab")
        in_colab = True
    except ImportError:
        print("✓ Running locally")
        in_colab = False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # Create configuration
    config = create_colab_config()
    print("✓ Configuration created")

    # Test basic imports
    print("\n=== Testing Imports ===")
    try:
        from data.datasets import CIFAR10LT
        from data.transforms import get_transforms
        from models.backbones import ResNet32
        from models.gating import PACBayesGating, FeatureExtractor
        from models.plugin_rule import PluginOptimizer
        from models.metrics import SelectiveMetrics

        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return 1

    # Test dataset creation
    print("\n=== Testing Dataset ===")
    try:
        transform_train, transform_test = get_transforms(config)
        lt_dataset = CIFAR10LT(root="./data", imbalance_factor=100, seed=42)
        train_dataset, test_dataset, group_info = lt_dataset.get_datasets(
            transform_train, transform_test
        )
        print(
            f"✓ Dataset created: {len(train_dataset)} train, {len(test_dataset)} test"
        )
        print(f"✓ Group info: {group_info}")
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        return 1

    # Test model creation
    print("\n=== Testing Model ===")
    try:
        model = ResNet32(num_classes=10).to(device)
        print(
            f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters"
        )
    except Exception as e:
        print(f"✗ Model error: {e}")
        return 1

    # Test synthetic data pipeline
    print("\n=== Testing Pipeline with Synthetic Data ===")
    try:
        # Create synthetic data
        batch_size = 32
        num_classes = 10

        # Synthetic model probabilities (3 models)
        model_probs_list = [
            torch.softmax(torch.randn(batch_size, num_classes), dim=1) for _ in range(3)
        ]
        targets = torch.randint(0, num_classes, (batch_size,))

        # Test feature extraction
        feature_extractor = FeatureExtractor(config["gating"])
        group_ids = torch.tensor(
            [group_info["class_to_group"][t.item()] for t in targets]
        )
        group_onehot = torch.nn.functional.one_hot(group_ids, num_classes=2).float()

        features = feature_extractor.extract(model_probs_list, group_onehot)
        print(f"✓ Feature extraction: {features.shape}")

        # Test gating network
        input_dim = features.size(1)
        num_models = len(model_probs_list)
        gating_model = PACBayesGating(input_dim, num_models, config["gating"])
        gating_weights = gating_model.forward(features)
        print(f"✓ Gating network: {gating_weights.shape}")

        # Test ensemble probabilities
        ensemble_probs = gating_model.compute_ensemble_probs(
            model_probs_list, gating_weights
        )
        print(f"✓ Ensemble probabilities: {ensemble_probs.shape}")

        # Test plugin rule
        plugin_optimizer = PluginOptimizer(config["plugin"])
        alpha, mu = plugin_optimizer.optimize(
            ensemble_probs, targets, group_ids, group_info
        )
        print(f"✓ Plugin optimization: α={alpha}, μ={mu}")

        # Test predictions
        plugin_rule = plugin_optimizer.create_plugin_rule(alpha, mu, group_info)
        predictions, rejections = plugin_rule.forward(ensemble_probs)
        print(f"✓ Predictions: {predictions.shape}, rejections: {rejections.shape}")

        # Test metrics
        metrics_computer = SelectiveMetrics(num_groups=2)
        metrics = metrics_computer.compute_all_metrics(
            predictions, targets, rejections, group_ids, ensemble_probs
        )
        print(f"✓ Metrics computed:")
        print(f"  - Coverage: {metrics['coverage']:.3f}")
        print(f"  - BSE: {metrics['balanced_selective_error']:.3f}")
        print(f"  - WGSE: {metrics['worst_group_selective_error']:.3f}")

    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n🎉 All tests passed! PB-GSE is ready for use in Colab.")

    if in_colab:
        print("\n=== Next Steps for Colab ===")
        print("1. Download CIFAR-10 dataset:")
        print("   !mkdir -p data")
        print("   # Dataset will be downloaded automatically")
        print("\n2. Run quick experiment:")
        print("   python pb_gse/scripts/colab_demo.py")
        print("\n3. For full experiment, use:")
        print(
            "   python pb_gse/scripts/run_experiment.py --config pb_gse/configs/experiment.yaml"
        )

    return 0


if __name__ == "__main__":
    exit(main())
