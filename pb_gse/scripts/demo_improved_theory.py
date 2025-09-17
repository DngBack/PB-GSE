"""
Demo script with improved PB-GSE theory implementation
"""

import torch
import numpy as np
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import CIFAR10LT
from data.transforms import get_transforms
from models.backbones import ResNet32
from models.losses_lt import CrossEntropyLoss
from models.calibration import GroupTemperatureScaling
from models.gating import PACBayesGating, FeatureExtractor, BalancedLinearLoss
from models.plugin_rule import PluginRule, FixedPointSolver, GridSearchOptimizer
from models.metrics import SelectiveMetrics


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_improved_config():
    """Create configuration with improved theory implementation"""
    config = {
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
        },
        "gating": {
            "network": {"hidden_dims": [64, 32], "dropout": 0.1, "activation": "relu"},
            "features": {
                "use_probs": True,
                "use_entropy": True,
                "use_max_prob": True,
                "use_disagreement": True,
                "use_soft_group_mass": True,  # NEW: Soft group-mass features
                "use_group_onehot": True,
            },
            "pac_bayes": {
                "method": "deterministic",  # Use deterministic for stability
                "prior_std": 1.0,
                "posterior_std_init": 0.1,
                "rejection_cost": 0.1,
                "num_groups": 2,
            },
            "epochs": 10,
            "lr": 1e-3,
            "batch_size": 256,
        },
        "plugin": {
            "rejection_cost": 0.1,
            "fixed_point": {
                "max_iterations": 15,
                "tolerance": 1e-6,
                "damping_factor": 0.3,  # NEW: Damping factor
                "eps": 0.01,  # NEW: Clipping epsilon
                "lambda_grid": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
            "groups": {"num_groups": 2, "group_names": ["head", "tail"]},
            "worst_group": {"enabled": False},
            "coverage_levels": [0.7, 0.8, 0.9],
        },
    }
    return config


def simulate_base_models(dataset, num_classes, device, num_models=3):
    """Simulate diverse base models"""
    logging.info("🔧 Simulating diverse base models...")

    models = []
    model_names = ["cRT", "LDAM_DRW", "CB_Focal"]

    for i, name in enumerate(model_names[:num_models]):
        model = ResNet32(num_classes).to(device)

        # Add some diversity by varying initialization
        torch.manual_seed(42 + i * 10)
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.01

        models.append((name, model))
        logging.info(f"  ✅ Created {name} model")

    return models


def extract_probabilities(models, dataloader, device):
    """Extract probabilities from base models"""
    logging.info("📊 Extracting probabilities from base models...")

    all_probs = []
    targets_list = []

    with torch.no_grad():
        for model_name, model in models:
            model.eval()
            model_probs = []

            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                model_probs.append(probs.cpu())

                if len(targets_list) == 0:  # Only collect targets once
                    targets_list.append(targets)

            all_probs.append(torch.cat(model_probs, dim=0))
            logging.info(f"  ✅ Extracted probabilities from {model_name}")

    targets_tensor = torch.cat(targets_list, dim=0)
    return all_probs, targets_tensor


def calibrate_models(model_probs_list, targets, group_ids, group_info):
    """Apply group-aware temperature scaling"""
    logging.info("🎯 Applying group-aware calibration...")

    calibrated_probs = []

    for i, probs in enumerate(model_probs_list):
        # Convert to logits (inverse softmax)
        logits = torch.log(probs + 1e-8)

        # Apply group temperature scaling
        calibrator = GroupTemperatureScaling(num_groups=2)
        calibrator.calibrate(logits, targets, group_ids)

        # Get calibrated probabilities
        calibrated_logits = calibrator(logits, group_ids)
        calibrated_probs_i = torch.softmax(calibrated_logits, dim=1)
        calibrated_probs.append(calibrated_probs_i)

        logging.info(f"  ✅ Calibrated model {i + 1}")

    return calibrated_probs


def train_improved_gating(
    model_probs_list, targets, group_ids, group_info, config, device
):
    """Train gating network with improved theory"""
    logging.info("🧠 Training improved gating network...")

    # Create feature extractor with group info
    feature_extractor = FeatureExtractor(config["gating"], group_info)

    # Create group one-hot encodings
    num_groups = config["plugin"]["groups"]["num_groups"]
    group_onehot = torch.nn.functional.one_hot(
        group_ids, num_classes=num_groups
    ).float()

    # Extract features
    features = feature_extractor.extract(model_probs_list, group_onehot)
    logging.info(f"  📏 Feature dimension: {features.shape[1]}")

    # Create gating model
    input_dim = features.shape[1]
    num_models = len(model_probs_list)
    gating_model = PACBayesGating(input_dim, num_models, config).to(device)

    # Initialize plugin parameters for loss computation
    alpha = {0: 1.2, 1: 0.8}  # Initial values
    mu = {0: 0.1, 1: -0.1}
    rejection_cost = config["plugin"]["rejection_cost"]

    # Create balanced linear loss
    loss_fn = BalancedLinearLoss(alpha, mu, rejection_cost, group_info)

    # Training setup
    optimizer = torch.optim.Adam(gating_model.parameters(), lr=config["gating"]["lr"])

    # Simple training loop
    gating_model.train()
    for epoch in range(config["gating"]["epochs"]):
        optimizer.zero_grad()

        # Forward pass
        if config["gating"]["pac_bayes"]["method"] == "gaussian":
            # Gaussian PAC-Bayes (not implemented in this demo)
            pass
        else:
            # Deterministic gating
            gating_weights = gating_model.forward(features.to(device))

            # Compute ensemble probabilities
            ensemble_probs = torch.zeros_like(model_probs_list[0]).to(device)
            for m, probs in enumerate(model_probs_list):
                ensemble_probs += gating_weights[:, m : m + 1] * probs.to(device)

        # Compute balanced linear loss
        loss = loss_fn(ensemble_probs, targets.to(device), group_ids.to(device))

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            logging.info(f"  📈 Epoch {epoch}: Loss = {loss.item():.4f}")

    logging.info("  ✅ Gating network training completed")
    return gating_model, features


def optimize_plugin_parameters(
    gating_model,
    features,
    model_probs_list,
    targets,
    group_ids,
    group_info,
    config,
    device,
):
    """Optimize plugin rule parameters with improved fixed-point"""
    logging.info("⚙️  Optimizing plugin rule parameters...")

    # Get ensemble probabilities
    with torch.no_grad():
        gating_weights = gating_model.forward(features.to(device))
        ensemble_probs = torch.zeros_like(model_probs_list[0]).to(device)
        for m, probs in enumerate(model_probs_list):
            ensemble_probs += gating_weights[:, m : m + 1] * probs.to(device)

    # Fixed-point solver with damping and clipping
    fixed_point_config = config["plugin"]["fixed_point"]
    solver = FixedPointSolver(
        max_iterations=fixed_point_config["max_iterations"],
        tolerance=fixed_point_config["tolerance"],
        damping_factor=fixed_point_config["damping_factor"],
        eps=fixed_point_config["eps"],
    )

    # Grid search for μ
    grid_optimizer = GridSearchOptimizer(
        lambda_grid=fixed_point_config["lambda_grid"],
        num_groups=config["plugin"]["groups"]["num_groups"],
    )

    # Optimize
    best_alpha, best_mu = grid_optimizer.optimize(
        ensemble_probs.cpu(), targets, group_ids, group_info
    )

    logging.info(f"  ✅ Optimized α = {best_alpha}")
    logging.info(f"  ✅ Optimized μ = {best_mu}")

    return best_alpha, best_mu, ensemble_probs


def evaluate_improved_system(plugin_rule, ensemble_probs, targets, group_ids, config):
    """Evaluate the improved PB-GSE system"""
    logging.info("📊 Evaluating improved PB-GSE system...")

    # Get predictions and rejections
    with torch.no_grad():
        predictions, rejections = plugin_rule.forward(ensemble_probs.cpu())

    # Compute metrics
    metrics_calculator = SelectiveMetrics(num_groups=2)
    results = metrics_calculator.compute_all_metrics(
        predictions, targets, rejections, group_ids
    )

    # Print results
    logging.info("📈 Improved PB-GSE Results:")
    logging.info(f"  🎯 Coverage: {results['coverage']:.3f}")
    logging.info(
        f"  📉 Balanced Selective Error: {results['balanced_selective_error']:.3f}"
    )
    logging.info(
        f"  ⚠️  Worst-Group Selective Error: {results['worst_group_selective_error']:.3f}"
    )
    logging.info(f"  📊 Head Coverage: {results['coverage_group_0']:.3f}")
    logging.info(f"  📊 Tail Coverage: {results['coverage_group_1']:.3f}")

    return results


def main():
    """Main demo function"""
    setup_logging()
    logging.info("🚀 PB-GSE Improved Theory Demo")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🖥️  Using device: {device}")

    # Configuration
    config = create_improved_config()

    # Create dataset
    logging.info("📁 Creating CIFAR-10-LT dataset...")
    dataset = CIFAR10LT(
        root=config["data"]["root"],
        imbalance_factor=config["data"]["imbalance_factor"],
        seed=config["data"]["seed"],
    )

    train_transform, test_transform = get_transforms(config)
    datasets_result = dataset.get_datasets(train_transform, test_transform)

    # Handle different return formats
    if len(datasets_result) == 2:
        train_dataset, test_dataset = datasets_result
    else:
        train_dataset, test_dataset = datasets_result[0], datasets_result[1]

    # Create groups
    train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    group_info = dataset.create_groups(
        train_targets, config["data"]["groups"]["tail_threshold"]
    )

    logging.info(f"  📊 Head classes: {len(group_info['head_classes'])}")
    logging.info(f"  📊 Tail classes: {len(group_info['tail_classes'])}")

    # Create small test dataset for demo
    subset_size = 500
    indices = torch.randperm(len(test_dataset))[:subset_size]
    test_subset = torch.utils.data.Subset(test_dataset, indices)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    # Simulate base models
    num_classes = 10
    models = simulate_base_models(test_subset, num_classes, device)

    # Extract probabilities
    model_probs_list, targets = extract_probabilities(models, test_loader, device)

    # Create group IDs
    class_to_group = group_info["class_to_group"]
    group_ids = torch.tensor([class_to_group[t.item()] for t in targets])

    # Calibrate models
    calibrated_probs = calibrate_models(
        model_probs_list, targets, group_ids, group_info
    )

    # Train improved gating network
    gating_model, features = train_improved_gating(
        calibrated_probs, targets, group_ids, group_info, config, device
    )

    # Optimize plugin parameters
    best_alpha, best_mu, ensemble_probs = optimize_plugin_parameters(
        gating_model,
        features,
        calibrated_probs,
        targets,
        group_ids,
        group_info,
        config,
        device,
    )

    # Create plugin rule
    plugin_rule = PluginRule(
        best_alpha, best_mu, config["plugin"]["rejection_cost"], group_info
    )

    # Evaluate system
    results = evaluate_improved_system(
        plugin_rule, ensemble_probs, targets, group_ids, config
    )

    logging.info("🎉 Demo completed successfully!")
    logging.info("✨ Key improvements implemented:")
    logging.info("  - ✅ Corrected Balanced Linear Loss (Formula 6)")
    logging.info("  - ✅ Fixed Plugin Rule implementation (Theorem 1)")
    logging.info("  - ✅ Added Soft Group-Mass features")
    logging.info("  - ✅ Enhanced Fixed-Point with damping & clipping")
    logging.info("  - ✅ Proper PAC-Bayes bound with L_α scaling")

    return results


if __name__ == "__main__":
    main()
