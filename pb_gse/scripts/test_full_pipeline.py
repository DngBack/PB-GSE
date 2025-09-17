"""
Test complete PB-GSE pipeline step by step
"""

import os
import sys
import torch
import yaml
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def test_data_loading():
    """Test 1: Data loading"""
    print("🔍 Test 1: Data Loading")

    try:
        from data.datasets import get_dataset_and_splits

        config = {
            "data": {
                "name": "cifar10_lt",
                "root": "./data",
                "imbalance_factor": 100,
                "seed": 42,
                "splits": {"train": 0.8, "cal": 0.1, "val": 0.05, "test": 0.05},
                "groups": {"num_groups": 2, "tail_threshold": 50},
                "augmentation": {
                    "train": {
                        "rand_augment": {"n": 2, "m": 10},
                        "mixup": 0.2,
                        "cutmix": 0.2,
                        "random_resized_crop": True,
                        "color_jitter": 0.1,
                    },
                    "test": {"center_crop": True},
                },
                "sampling": {"method": "class_aware", "balanced_batch_ratio": 0.5},
                "batch_size": 256,
                "num_workers": 2,
                "pin_memory": False,
            }
        }

        datasets, group_info = get_dataset_and_splits(config)

        print(f"✅ Data loaded successfully")
        print(f"   Train: {len(datasets['train'])} samples")
        print(f"   Test: {len(datasets['test'])} samples")
        print(
            f"   Groups: {group_info['num_groups']} (head: {len(group_info['head_classes'])}, tail: {len(group_info['tail_classes'])})"
        )

        return True, config, datasets, group_info

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False, None, None, None


def test_base_model():
    """Test 2: Single base model training (quick)"""
    print("\n🔍 Test 2: Base Model Training (Quick)")

    try:
        from models.backbones import ResNet32
        from models.losses_lt import CrossEntropyLoss
        import torch.nn as nn

        # Create simple model
        model = ResNet32(num_classes=10)
        criterion = CrossEntropyLoss()

        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))

        logits = model(x)
        loss = criterion(logits, y)

        print(f"✅ Model forward pass successful")
        print(f"   Input: {x.shape}")
        print(f"   Output: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")

        return True, model

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False, None


def test_calibration():
    """Test 3: Calibration"""
    print("\n🔍 Test 3: Calibration")

    try:
        from models.calibration import GroupTemperatureScaling

        # Create dummy data
        logits = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        group_ids = torch.randint(0, 2, (100,))

        # Test calibration
        calibrator = GroupTemperatureScaling(num_groups=2)
        calibrator.calibrate(logits, targets, group_ids)

        calibrated_logits = calibrator(logits, group_ids)

        print(f"✅ Calibration successful")
        print(f"   Input shape: {logits.shape}")
        print(f"   Output shape: {calibrated_logits.shape}")
        print(f"   Temperatures: {calibrator.temperatures}")

        return True

    except Exception as e:
        print(f"❌ Calibration test failed: {e}")
        return False


def test_gating_network():
    """Test 4: Gating Network"""
    print("\n🔍 Test 4: Gating Network")

    try:
        from models.gating import PACBayesGating, FeatureExtractor

        # Config for gating
        gating_config = {
            "network": {"hidden_dims": [64, 32], "dropout": 0.1, "activation": "relu"},
            "features": {
                "use_probs": True,
                "use_entropy": True,
                "use_max_prob": True,
                "use_disagreement": True,
                "use_group_onehot": True,
            },
            "pac_bayes": {
                "method": "deterministic",
                "prior_std": 1.0,
                "posterior_std_init": 0.1,
            },
        }

        # Create dummy ensemble probabilities
        num_models = 3
        num_samples = 100
        num_classes = 10
        num_groups = 2

        model_probs = torch.randn(num_models, num_samples, num_classes).softmax(dim=-1)
        targets = torch.randint(0, num_classes, (num_samples,))
        group_ids = torch.randint(0, num_groups, (num_samples,))

        # Create feature extractor and gating network
        feature_extractor = FeatureExtractor(gating_config)

        # Create group one-hot encoding
        group_onehot = torch.zeros(num_samples, num_groups)
        group_onehot[torch.arange(num_samples), group_ids] = 1.0

        # Extract features
        model_probs_list = [model_probs[i] for i in range(num_models)]
        features = feature_extractor.extract(model_probs_list, group_onehot)

        gating = PACBayesGating(
            input_dim=features.shape[1], num_models=num_models, config=gating_config
        )

        # Test forward pass
        weights = gating(features)

        print(f"✅ Gating network successful")
        print(f"   Feature dim: {features.shape[1]}")
        print(f"   Weights shape: {weights.shape}")
        print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

        return True

    except Exception as e:
        print(f"❌ Gating network test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_plugin_rule():
    """Test 5: Plugin Rule"""
    print("\n🔍 Test 5: Plugin Rule")

    try:
        from models.plugin_rule import PluginOptimizer

        # Config for plugin
        plugin_config = {
            "rejection_cost": 0.3,
            "fixed_point": {
                "max_iterations": 10,
                "tolerance": 1e-6,
                "lambda_grid": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
            "groups": {"num_groups": 2, "group_names": ["head", "tail"]},
            "worst_group": {"enabled": False},
        }

        # Create dummy data
        ensemble_probs = torch.randn(100, 10).softmax(dim=-1)
        targets = torch.randint(0, 10, (100,))
        group_ids = torch.randint(0, 2, (100,))

        group_info = {
            "num_groups": 2,
            "class_to_group": {i: 0 if i < 9 else 1 for i in range(10)},
            "head_classes": list(range(9)),
            "tail_classes": [9],
        }

        # Test plugin optimization
        plugin_optimizer = PluginOptimizer(plugin_config)
        alpha, mu = plugin_optimizer.optimize(
            ensemble_probs, targets, group_ids, group_info
        )

        # Create plugin rule
        plugin_rule = plugin_optimizer.create_plugin_rule(alpha, mu, group_info)
        predictions, rejections = plugin_rule.forward(ensemble_probs)

        print(f"✅ Plugin rule successful")
        print(f"   Alpha: {alpha}")
        print(f"   Mu: {mu}")
        print(f"   Coverage: {(~rejections).float().mean():.4f}")

        return True

    except Exception as e:
        print(f"❌ Plugin rule test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metrics():
    """Test 6: Metrics computation"""
    print("\n🔍 Test 6: Metrics")

    try:
        from models.metrics import SelectiveMetrics

        # Create dummy results
        predictions = torch.randint(0, 10, (100,))
        targets = torch.randint(0, 10, (100,))
        rejections = torch.rand(100) < 0.2  # 20% rejection
        group_ids = torch.randint(0, 2, (100,))
        probs = torch.randn(100, 10).softmax(dim=-1)

        # Compute metrics
        metrics_computer = SelectiveMetrics(num_groups=2)
        all_metrics = metrics_computer.compute_all_metrics(
            predictions, targets, rejections, group_ids, probs
        )

        print(f"✅ Metrics computation successful")
        print(f"   Coverage: {all_metrics['coverage']:.4f}")
        print(f"   BSE: {all_metrics['balanced_selective_error']:.4f}")
        print(f"   WGSE: {all_metrics['worst_group_selective_error']:.4f}")

        return True

    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def main():
    """Run all tests"""
    setup_logging()

    print("🚀 PB-GSE Pipeline Testing")
    print("=" * 50)

    results = []

    # Test 1: Data loading
    success, config, datasets, group_info = test_data_loading()
    results.append(("Data Loading", success))

    # Test 2: Base model
    success, model = test_base_model()
    results.append(("Base Model", success))

    # Test 3: Calibration
    success = test_calibration()
    results.append(("Calibration", success))

    # Test 4: Gating network
    success = test_gating_network()
    results.append(("Gating Network", success))

    # Test 5: Plugin rule
    success = test_plugin_rule()
    results.append(("Plugin Rule", success))

    # Test 6: Metrics
    success = test_metrics()
    results.append(("Metrics", success))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Pipeline is working correctly.")
    else:
        print("❌ Some tests failed. Check errors above.")

    return all_passed


if __name__ == "__main__":
    main()
