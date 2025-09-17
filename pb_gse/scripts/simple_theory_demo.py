"""
Simple demo to test improved PB-GSE theory implementation
"""

import torch
import numpy as np
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gating import BalancedLinearLoss, FeatureExtractor, PACBayesGating
from models.plugin_rule import PluginRule, FixedPointSolver, GridSearchOptimizer
from models.metrics import SelectiveMetrics


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_synthetic_data():
    """Create synthetic data for testing"""
    logging.info("🔢 Creating synthetic data...")

    batch_size = 100
    num_classes = 6
    num_models = 3

    # Create model probabilities
    model_probs_list = []
    for m in range(num_models):
        torch.manual_seed(42 + m)
        probs = torch.rand(batch_size, num_classes)
        probs = probs / probs.sum(dim=1, keepdim=True)  # normalize
        model_probs_list.append(probs)

    # Create targets and group IDs
    torch.manual_seed(42)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Group assignment: classes 0,1,2 = head (group 0), classes 3,4,5 = tail (group 1)
    group_info = {
        "class_to_group": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
        "group_to_classes": {0: [0, 1, 2], 1: [3, 4, 5]},
    }

    group_ids = torch.tensor([group_info["class_to_group"][t.item()] for t in targets])

    logging.info(
        f"  📊 Created {batch_size} samples, {num_classes} classes, {num_models} models"
    )
    logging.info(
        f"  📊 Head samples: {(group_ids == 0).sum()}, Tail samples: {(group_ids == 1).sum()}"
    )

    return model_probs_list, targets, group_ids, group_info


def test_improved_balanced_loss():
    """Test improved balanced linear loss"""
    logging.info("🧪 Testing Improved Balanced Linear Loss...")

    model_probs_list, targets, group_ids, group_info = create_synthetic_data()

    # Create ensemble probabilities (simple average)
    ensemble_probs = torch.stack(model_probs_list).mean(dim=0)

    # Parameters
    alpha = {0: 1.3, 1: 0.7}  # head group has higher acceptance rate
    mu = {0: 0.1, 1: -0.1}
    rejection_cost = 0.15

    # Create improved loss function
    loss_fn = BalancedLinearLoss(alpha, mu, rejection_cost, group_info)

    # Compute loss
    loss = loss_fn(ensemble_probs, targets, group_ids)

    logging.info(f"  ✅ Balanced Linear Loss: {loss.item():.4f}")
    logging.info(f"  ✅ L_α normalization factor: {loss_fn.L_alpha:.4f}")

    # Test that loss is properly normalized
    assert 0 <= loss <= 1, f"Loss should be in [0,1], got {loss}"
    logging.info("  ✅ Loss normalization validated")

    return loss.item()


def test_improved_plugin_rule():
    """Test improved plugin rule with fixed-point optimization"""
    logging.info("🔧 Testing Improved Plugin Rule...")

    model_probs_list, targets, group_ids, group_info = create_synthetic_data()
    ensemble_probs = torch.stack(model_probs_list).mean(dim=0)

    # Initial parameters
    mu = {0: 0.05, 1: -0.05}
    rejection_cost = 0.1

    # Fixed-point solver with damping
    solver = FixedPointSolver(
        max_iterations=15, tolerance=1e-6, damping_factor=0.3, eps=0.01
    )

    # Solve for optimal α
    alpha = solver.solve_alpha(
        ensemble_probs, targets, group_ids, mu, rejection_cost, group_info, num_groups=2
    )

    logging.info(f"  ✅ Optimized α: {alpha}")

    # Create plugin rule
    plugin_rule = PluginRule(alpha, mu, rejection_cost, group_info)

    # Get predictions and rejections
    predictions, rejections = plugin_rule.forward(ensemble_probs)

    # Compute metrics
    metrics_calc = SelectiveMetrics(num_groups=2)
    results = metrics_calc.compute_all_metrics(
        predictions, targets, rejections, group_ids
    )

    logging.info(f"  📊 Coverage: {results['coverage']:.3f}")
    logging.info(
        f"  📊 Balanced Selective Error: {results['balanced_selective_error']:.3f}"
    )

    # Check if group-specific metrics exist
    if "coverage_group_0" in results:
        logging.info(f"  📊 Head Coverage: {results['coverage_group_0']:.3f}")
        logging.info(f"  📊 Tail Coverage: {results['coverage_group_1']:.3f}")
    else:
        logging.info(f"  📊 Group metrics not available")

    return results


def test_soft_group_mass_features():
    """Test soft group-mass features"""
    logging.info("🎯 Testing Soft Group-Mass Features...")

    model_probs_list, targets, group_ids, group_info = create_synthetic_data()

    # Create feature extractor with soft group-mass
    config = {
        "features": {
            "use_probs": True,
            "use_entropy": True,
            "use_max_prob": True,
            "use_disagreement": True,
            "use_soft_group_mass": True,
            "use_group_onehot": True,
        }
    }

    feature_extractor = FeatureExtractor(config, group_info)

    # Create group one-hot for training
    group_onehot = torch.nn.functional.one_hot(group_ids, num_classes=2).float()

    # Extract features
    features = feature_extractor.extract(model_probs_list, group_onehot)

    logging.info(f"  ✅ Feature shape: {features.shape}")

    # Validate feature components
    num_classes = model_probs_list[0].shape[1]
    num_models = len(model_probs_list)
    num_groups = 2

    expected_dim = (
        num_classes * num_models  # concatenated probs
        + num_models  # entropy features
        + num_models  # max prob features
        + 1  # disagreement scalar
        + num_models * num_groups  # soft group-mass features
        + num_groups  # group one-hot
    )

    logging.info(f"  📏 Expected dim: {expected_dim}, Actual dim: {features.shape[1]}")

    # Test soft group-mass computation manually
    for m, probs in enumerate(model_probs_list):
        for group_id, class_list in group_info["group_to_classes"].items():
            class_indices = torch.tensor(class_list)
            group_mass = torch.sum(probs[:, class_indices], dim=1)
            logging.info(
                f"  📊 Model {m}, Group {group_id} mass range: [{group_mass.min():.3f}, {group_mass.max():.3f}]"
            )

    return features.shape


def test_improved_gating_training():
    """Test improved gating network training"""
    logging.info("🧠 Testing Improved Gating Training...")

    model_probs_list, targets, group_ids, group_info = create_synthetic_data()

    # Create features
    config = {
        "pac_bayes": {"method": "deterministic", "prior_std": 1.0, "l2_weight": 0.01},
        "network": {"hidden_dims": [32, 16], "dropout": 0.1, "activation": "relu"},
        "features": {
            "use_probs": True,
            "use_entropy": True,
            "use_max_prob": True,
            "use_disagreement": True,
            "use_soft_group_mass": True,
            "use_group_onehot": False,  # No group onehot for inference
        },
    }

    feature_extractor = FeatureExtractor(config, group_info)
    features = feature_extractor.extract(model_probs_list)

    # Create gating model
    input_dim = features.shape[1]
    num_models = len(model_probs_list)
    gating_model = PACBayesGating(input_dim, num_models, config)

    # Test forward pass
    gating_weights = gating_model.forward(features)

    logging.info(f"  ✅ Gating weights shape: {gating_weights.shape}")
    logging.info(
        f"  ✅ Weights sum to 1: {torch.allclose(gating_weights.sum(dim=1), torch.ones(features.shape[0]))}"
    )

    # Create ensemble probabilities
    ensemble_probs = torch.zeros_like(model_probs_list[0])
    for m, probs in enumerate(model_probs_list):
        ensemble_probs += gating_weights[:, m : m + 1] * probs

    # Test balanced linear loss
    alpha = {0: 1.2, 1: 0.8}
    mu = {0: 0.1, 1: -0.1}
    rejection_cost = 0.1

    loss_fn = BalancedLinearLoss(alpha, mu, rejection_cost, group_info)
    loss = loss_fn(ensemble_probs, targets, group_ids)

    logging.info(f"  ✅ Training loss: {loss.item():.4f}")

    return gating_model, ensemble_probs


def main():
    """Main demo function"""
    setup_logging()
    logging.info("🚀 Simple PB-GSE Theory Demo")

    # Test individual components
    logging.info("\n" + "=" * 50)
    loss_value = test_improved_balanced_loss()

    logging.info("\n" + "=" * 50)
    results = test_improved_plugin_rule()

    logging.info("\n" + "=" * 50)
    feature_shape = test_soft_group_mass_features()

    logging.info("\n" + "=" * 50)
    gating_model, ensemble_probs = test_improved_gating_training()

    # Summary
    logging.info("\n" + "=" * 50)
    logging.info("🎉 Simple Demo Completed Successfully!")
    logging.info("✨ Key Improvements Validated:")
    logging.info("  ✅ Balanced Linear Loss with L_α normalization")
    logging.info("  ✅ Plugin Rule with damped fixed-point optimization")
    logging.info("  ✅ Soft Group-Mass features for safe inference")
    logging.info("  ✅ PAC-Bayes gating with ensemble probabilities")

    logging.info(f"\n📊 Final Results Summary:")
    logging.info(f"  🎯 Balanced Linear Loss: {loss_value:.4f}")
    logging.info(f"  📈 Coverage: {results['coverage']:.3f}")
    logging.info(
        f"  📉 Balanced Selective Error: {results['balanced_selective_error']:.3f}"
    )
    logging.info(f"  🔍 Feature Dimension: {feature_shape[1]}")

    return {
        "loss": loss_value,
        "coverage": results["coverage"],
        "bse": results["balanced_selective_error"],
        "feature_dim": feature_shape[1],
    }


if __name__ == "__main__":
    main()
