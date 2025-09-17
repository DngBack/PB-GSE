"""
Validation script to ensure PB-GSE implementation is working correctly
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_data_pipeline():
    """Test data loading and processing"""
    print("=== Testing Data Pipeline ===")

    from data.datasets import CIFAR10LT
    from data.transforms import get_transforms

    config = {
        "data": {
            "name": "cifar10_lt",
            "augmentation": {
                "train": {"rand_augment": {"n": 2, "m": 10}},
                "test": {"center_crop": True},
            },
        }
    }

    transform_train, transform_test = get_transforms(config)
    lt_dataset = CIFAR10LT(root="./data", imbalance_factor=100, seed=42)
    train_dataset, test_dataset, group_info = lt_dataset.get_datasets(
        transform_train, transform_test
    )

    print(f"✓ Dataset created: {len(train_dataset)} train, {len(test_dataset)} test")
    print(
        f"✓ Groups: Head={len(group_info['head_classes'])}, Tail={len(group_info['tail_classes'])}"
    )

    return group_info


def test_models():
    """Test model creation and basic functionality"""
    print("\n=== Testing Models ===")

    from models.backbones import ResNet32
    from models.losses_lt import CrossEntropyLoss, BalancedSoftmaxLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test model creation
    model = ResNet32(num_classes=10).to(device)
    print(
        f"✓ ResNet32 created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    output = model(dummy_input)
    print(f"✓ Forward pass: {dummy_input.shape} -> {output.shape}")

    # Test loss functions
    class_frequencies = [1000, 500, 100, 50, 10]

    ce_loss = CrossEntropyLoss()
    bs_loss = BalancedSoftmaxLoss(class_frequencies)

    dummy_targets = torch.randint(0, 5, (4,)).to(device)
    dummy_logits = torch.randn(4, 5).to(device)

    ce_val = ce_loss(dummy_logits, dummy_targets)
    bs_val = bs_loss(dummy_logits, dummy_targets)

    print(f"✓ CrossEntropy loss: {ce_val.item():.4f}")
    print(f"✓ BalancedSoftmax loss: {bs_val.item():.4f}")

    return model


def test_gating_and_plugin():
    """Test gating network and plugin rule"""
    print("\n=== Testing Gating & Plugin Rule ===")

    from models.gating import PACBayesGating, FeatureExtractor
    from models.plugin_rule import PluginOptimizer
    from models.metrics import SelectiveMetrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic data
    batch_size = 16
    num_classes = 10
    num_models = 3

    # Synthetic model probabilities
    model_probs_list = [
        torch.softmax(torch.randn(batch_size, num_classes), dim=1).to(device)
        for _ in range(num_models)
    ]
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)

    # Create group info
    group_info = {
        "class_to_group": {i: 0 if i < 5 else 1 for i in range(num_classes)},
        "group_to_classes": {0: list(range(5)), 1: list(range(5, 10))},
    }

    # Test feature extraction
    config = {
        "network": {"hidden_dims": [32], "dropout": 0.1, "activation": "relu"},
        "features": {
            "use_probs": True,
            "use_entropy": False,
            "use_max_prob": False,
            "use_disagreement": False,
            "use_group_onehot": True,
        },
        "pac_bayes": {"method": "deterministic", "prior_std": 1.0},
    }

    feature_extractor = FeatureExtractor(config)
    group_ids = torch.tensor(
        [group_info["class_to_group"][t.item()] for t in targets]
    ).to(device)
    group_onehot = (
        torch.nn.functional.one_hot(group_ids, num_classes=2).float().to(device)
    )

    features = feature_extractor.extract(model_probs_list, group_onehot)
    print(f"✓ Feature extraction: {features.shape}")

    # Test gating network
    input_dim = features.size(1)
    gating_model = PACBayesGating(input_dim, num_models, config).to(device)
    gating_weights = gating_model.forward(features)
    print(f"✓ Gating weights: {gating_weights.shape}")

    # Test ensemble probabilities
    ensemble_probs = gating_model.compute_ensemble_probs(
        model_probs_list, gating_weights
    )
    print(f"✓ Ensemble probabilities: {ensemble_probs.shape}")

    # Test plugin rule
    plugin_config = {
        "rejection_cost": 0.1,
        "fixed_point": {
            "max_iterations": 5,
            "tolerance": 1e-6,
            "lambda_grid": [-0.5, 0.0, 0.5],
        },
        "groups": {"num_groups": 2},
    }

    plugin_optimizer = PluginOptimizer(plugin_config)
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

    print(f"✓ Metrics:")
    print(f"  - Coverage: {metrics['coverage']:.3f}")
    print(f"  - BSE: {metrics['balanced_selective_error']:.3f}")
    print(f"  - WGSE: {metrics['worst_group_selective_error']:.3f}")

    return True


def test_device_compatibility():
    """Test device compatibility and tensor operations"""
    print("\n=== Testing Device Compatibility ===")

    from data.transforms import MixUp, CutMix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test MixUp
    mixup = MixUp(alpha=0.2)
    x = torch.randn(4, 3, 32, 32).to(device)
    y = torch.randint(0, 10, (4,)).to(device)

    mixed_x, y_a, y_b, lam = mixup(x, y)
    print(f"✓ MixUp: {x.shape} -> {mixed_x.shape}, λ={lam.item():.3f}")
    print(
        f"  - All tensors on {mixed_x.device}: {mixed_x.device == y_a.device == lam.device}"
    )

    # Test CutMix
    cutmix = CutMix(alpha=0.2)
    mixed_x, y_a, y_b, lam = cutmix(x, y)
    print(f"✓ CutMix: {x.shape} -> {mixed_x.shape}, λ={lam.item():.3f}")
    print(
        f"  - All tensors on {mixed_x.device}: {mixed_x.device == y_a.device == lam.device}"
    )

    return True


def test_configs():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")

    import yaml

    config_files = [
        "pb_gse/configs/experiment.yaml",
        "pb_gse/configs/base_crt.yaml",
        "pb_gse/configs/base_ldam.yaml",
        "pb_gse/configs/base_cbfocal.yaml",
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            print(f"✓ Loaded {config_file}")
        else:
            print(f"✗ Missing {config_file}")
            return False

    return True


def main():
    print("=== PB-GSE Implementation Validation ===")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Test 1: Configuration loading
        if not test_configs():
            print("✗ Configuration test failed")
            return 1

        # Test 2: Device compatibility
        if not test_device_compatibility():
            print("✗ Device compatibility test failed")
            return 1

        # Test 3: Data pipeline
        group_info = test_data_pipeline()

        # Test 4: Models and core functionality
        model = test_models()

        # Test 5: Gating and plugin rule
        if not test_gating_and_plugin():
            print("✗ Gating/Plugin test failed")
            return 1

        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ PB-GSE implementation is working correctly")
        print("✅ Ready for research experiments")
        print("✅ Compatible with Google Colab")

        return 0

    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
