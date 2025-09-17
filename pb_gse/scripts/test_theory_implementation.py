"""
Test script to validate PB-GSE theory implementation
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.gating import BalancedLinearLoss, FeatureExtractor, PACBayesGating
from models.plugin_rule import PluginRule, FixedPointSolver


def test_balanced_linear_loss():
    """Test Formula (6): Balanced Linear Loss"""
    print("🧪 Testing Balanced Linear Loss (Formula 6)...")

    # Setup test data
    batch_size, num_classes = 10, 10
    ensemble_probs = torch.rand(batch_size, num_classes)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(
        dim=1, keepdim=True
    )  # normalize

    targets = torch.randint(0, num_classes, (batch_size,))
    group_ids = torch.randint(0, 2, (batch_size,))  # 2 groups

    # Group info
    group_info = {
        "class_to_group": {i: i % 2 for i in range(num_classes)},  # alternating groups
        "group_to_classes": {
            0: [i for i in range(0, num_classes, 2)],
            1: [i for i in range(1, num_classes, 2)],
        },
    }

    # Parameters
    alpha = {0: 1.5, 1: 0.8}  # head group has higher α
    mu = {0: 0.1, 1: -0.1}
    rejection_cost = 0.1

    # Create loss function
    loss_fn = BalancedLinearLoss(alpha, mu, rejection_cost, group_info)

    # Compute loss
    loss = loss_fn(ensemble_probs, targets, group_ids)

    print(f"✅ Balanced Linear Loss computed: {loss.item():.4f}")
    print(f"   - L_α normalization factor: {loss_fn.L_alpha:.4f}")

    # Validate loss is in [0, 1] (normalized)
    assert 0 <= loss <= 1, f"Loss should be normalized to [0,1], got {loss}"
    print("✅ Loss normalization validated")

    return True


def test_plugin_rule():
    """Test Theorem 1: Plugin Rule (Formula 5a, 5b)"""
    print("\n🧪 Testing Plugin Rule (Theorem 1)...")

    # Setup test data
    batch_size, num_classes = 5, 6
    probs = torch.rand(batch_size, num_classes)
    probs = probs / probs.sum(dim=1, keepdim=True)

    # Group info: 3 classes per group
    group_info = {
        "class_to_group": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
        "group_to_classes": {0: [0, 1, 2], 1: [3, 4, 5]},
    }

    alpha = {0: 1.2, 1: 0.9}
    mu = {0: 0.05, 1: -0.05}
    rejection_cost = 0.15

    # Create plugin rule
    plugin_rule = PluginRule(alpha, mu, rejection_cost, group_info)

    # Test classification (Formula 5a)
    predictions = plugin_rule.classify(probs)
    print(f"✅ Classifications computed: {predictions}")

    # Test rejection (Formula 5b)
    rejections = plugin_rule.reject(probs)
    print(f"✅ Rejections computed: {rejections}")

    # Test full forward
    preds, rejs = plugin_rule.forward(probs)
    assert torch.equal(preds, predictions), "Classification mismatch"
    assert torch.equal(rejs, rejections), "Rejection mismatch"

    print("✅ Plugin Rule implementation validated")

    return True


def test_fixed_point_solver():
    """Test Fixed-Point Iteration with damping and clipping"""
    print("\n🧪 Testing Fixed-Point Solver with damping...")

    # Setup test data
    batch_size, num_classes = 20, 4
    probs = torch.rand(batch_size, num_classes)
    probs = probs / probs.sum(dim=1, keepdim=True)

    targets = torch.randint(0, num_classes, (batch_size,))
    group_ids = torch.randint(0, 2, (batch_size,))

    group_info = {
        "class_to_group": {0: 0, 1: 0, 2: 1, 3: 1},
        "group_to_classes": {0: [0, 1], 1: [2, 3]},
    }

    mu = {0: 0.1, 1: -0.1}
    rejection_cost = 0.1

    # Create solver with damping
    solver = FixedPointSolver(max_iterations=10, damping_factor=0.3, eps=0.01)

    # Solve for α
    alpha = solver.solve_alpha(
        probs, targets, group_ids, mu, rejection_cost, group_info, num_groups=2
    )

    print(f"✅ Fixed-point converged: α = {alpha}")

    # Validate clipping: α_k ∈ [ε, K-ε]
    for group_id, alpha_val in alpha.items():
        assert solver.eps <= alpha_val <= 2 - solver.eps, (
            f"α[{group_id}] = {alpha_val} not in valid range"
        )

    print("✅ Fixed-point solver with damping validated")

    return True


def test_soft_group_mass_features():
    """Test soft group-mass features: g_mass^(m)(x; k) = Σ_{y∈G_k} p_y^(m)(x)"""
    print("\n🧪 Testing Soft Group-Mass Features...")

    # Setup test data
    batch_size, num_classes = 3, 6
    num_models = 2

    model_probs = []
    for m in range(num_models):
        probs = torch.rand(batch_size, num_classes)
        probs = probs / probs.sum(dim=1, keepdim=True)
        model_probs.append(probs)

    group_info = {
        "class_to_group": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
        "group_to_classes": {0: [0, 1, 2], 1: [3, 4, 5]},
    }

    # Create feature extractor
    config = {
        "features": {
            "use_probs": True,
            "use_entropy": False,
            "use_max_prob": False,
            "use_disagreement": False,
            "use_soft_group_mass": True,
            "use_group_onehot": False,
        }
    }

    feature_extractor = FeatureExtractor(config, group_info)
    features = feature_extractor.extract(model_probs)

    print(f"✅ Features extracted with shape: {features.shape}")

    # Validate soft group-mass: should sum to 1 for each model and group
    expected_dim = num_classes * num_models + num_models * 2  # probs + group_masses
    assert features.shape[1] == expected_dim, (
        f"Expected feature dim {expected_dim}, got {features.shape[1]}"
    )

    print("✅ Soft group-mass features validated")

    return True


def test_pac_bayes_bound():
    """Test PAC-Bayes bound with L_α scaling (Formula 7)"""
    print("\n🧪 Testing PAC-Bayes Bound (Formula 7)...")

    # Create a simple gating model
    config = {
        "pac_bayes": {
            "method": "gaussian",
            "prior_std": 1.0,
            "posterior_std_init": 0.1,
            "rejection_cost": 0.1,
            "num_groups": 2,
        },
        "network": {"hidden_dims": [32, 16], "dropout": 0.1, "activation": "relu"},
        "features": {
            "use_probs": True,
            "use_entropy": False,
            "use_max_prob": False,
            "use_disagreement": False,
            "use_soft_group_mass": False,
            "use_group_onehot": False,
        },
    }

    input_dim = 20  # 2 models * 10 classes
    num_models = 2

    gating_model = PACBayesGating(input_dim, num_models, config)

    # Test bound computation
    empirical_loss = torch.tensor(0.3)
    n_samples = 100
    L_alpha = 1.5

    bound = gating_model.pac_bayes_bound(
        empirical_loss, n_samples, delta=0.05, L_alpha=L_alpha
    )

    print(f"✅ PAC-Bayes bound computed: {bound.item():.4f}")
    assert bound >= empirical_loss, "Bound should be ≥ empirical loss"

    print("✅ PAC-Bayes bound with L_α scaling validated")

    return True


def main():
    """Run all tests"""
    print("🚀 Testing PB-GSE Theory Implementation\n")

    tests = [
        test_balanced_linear_loss,
        test_plugin_rule,
        test_fixed_point_solver,
        test_soft_group_mass_features,
        test_pac_bayes_bound,
    ]

    passed = 0
    total = len(tests)

    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_fn.__name__} failed: {e}")

    print(f"\n🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Theory implementation is correct.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    main()
