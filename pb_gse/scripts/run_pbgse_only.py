"""
Run only the PB-GSE method (gating + plugin rule) without retraining base models
"""

import os
import sys
import torch
import yaml
import argparse
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import get_dataset_and_splits
from models.gating import PACBayesGating, FeatureExtractor
from models.plugin_rule import PluginOptimizer
from models.inference import PBGSEInference
from models.metrics import SelectiveMetrics, compute_metrics_at_coverage, MetricsLogger
import json


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pbgse_only.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_model_probs(probs_dir: str, model_names: list, split: str) -> tuple:
    """Load model probabilities"""
    model_probs_list = []
    targets = None

    for model_name in model_names:
        prob_path = os.path.join(probs_dir, model_name, f"{split}.pth")
        if not os.path.exists(prob_path):
            raise FileNotFoundError(f"Probabilities not found: {prob_path}")

        data = torch.load(prob_path)
        probs = data["probs"]
        model_probs_list.append(probs)

        if targets is None:
            targets = data["targets"]

    return model_probs_list, targets


def create_synthetic_base_models(config: dict, device: str):
    """Create synthetic base model probabilities for demo"""

    logging.info("Creating synthetic base model probabilities...")

    # Load dataset to get proper targets and group info
    split_datasets, group_info = get_dataset_and_splits(config)

    # Create synthetic probabilities for each split
    num_classes = len(set(group_info["class_to_group"].keys()))
    model_names = ["synthetic_model1", "synthetic_model2", "synthetic_model3"]

    probs_dir = "./synthetic_probs"
    os.makedirs(probs_dir, exist_ok=True)

    for split_name, dataset in split_datasets.items():
        if split_name == "test":
            continue  # Skip test for now

        # Get targets for this split
        if hasattr(dataset, "indices"):
            targets = torch.tensor([dataset.dataset[idx][1] for idx in dataset.indices])
        else:
            targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])

        dataset_size = len(targets)

        for model_name in model_names:
            # Create synthetic probabilities with some class bias
            probs = torch.softmax(
                torch.randn(dataset_size, num_classes) + torch.randn(num_classes) * 0.5,
                dim=1,
            )

            # Save probabilities
            model_dir = os.path.join(probs_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            torch.save(
                {
                    "probs": probs,
                    "targets": targets,
                    "indices": torch.arange(dataset_size),
                },
                os.path.join(model_dir, f"{split_name}.pth"),
            )

    logging.info(f"Created synthetic probabilities in {probs_dir}")
    return probs_dir, group_info


def run_pbgse_method(config: dict, probs_dir: str, group_info: dict, device: str):
    """Run only the PB-GSE method (gating + plugin rule)"""

    logging.info("=== Running PB-GSE Method ===")

    # Use the actual trained models instead of synthetic
    model_names = ["cRT", "LDAM_DRW", "CB_Focal"]
    logging.info(f"Using trained models: {model_names}")

    # Load model probabilities
    cal_probs, cal_targets = load_model_probs(probs_dir, model_names, "cal")
    val_probs, val_targets = load_model_probs(probs_dir, model_names, "val")

    # Move to device
    cal_probs = [p.to(device) for p in cal_probs]
    val_probs = [p.to(device) for p in val_probs]
    cal_targets = cal_targets.to(device)
    val_targets = val_targets.to(device)

    # Create group IDs
    class_to_group = group_info["class_to_group"]
    cal_group_ids = torch.tensor([class_to_group[t.item()] for t in cal_targets]).to(
        device
    )
    val_group_ids = torch.tensor([class_to_group[t.item()] for t in val_targets]).to(
        device
    )

    # Extract features for gating
    feature_extractor = FeatureExtractor(config["gating"])

    cal_group_onehot = (
        torch.nn.functional.one_hot(
            cal_group_ids, num_classes=config["plugin"]["groups"]["num_groups"]
        )
        .float()
        .to(device)
    )
    cal_features = feature_extractor.extract(cal_probs, cal_group_onehot).to(device)

    val_group_onehot = (
        torch.nn.functional.one_hot(
            val_group_ids, num_classes=config["plugin"]["groups"]["num_groups"]
        )
        .float()
        .to(device)
    )
    val_features = feature_extractor.extract(val_probs, val_group_onehot).to(device)

    logging.info(f"Feature shapes: cal={cal_features.shape}, val={val_features.shape}")

    # Create and train gating network (simplified)
    input_dim = cal_features.size(1)
    num_models = len(model_names)

    gating_model = PACBayesGating(input_dim, num_models, config["gating"]).to(device)

    if config["gating"]["pac_bayes"]["method"] == "gaussian":
        params = [gating_model.posterior.mu, gating_model.posterior.log_sigma]
    else:
        params = gating_model.gating_net.parameters()

    optimizer = torch.optim.Adam(params, lr=float(config["gating"]["lr"]))

    # Simple training loop
    epochs = config["gating"]["epochs"]

    for epoch in range(epochs):
        gating_model.train()
        optimizer.zero_grad()

        # Forward pass
        gating_weights = gating_model.forward(cal_features)
        ensemble_probs = gating_model.compute_ensemble_probs(cal_probs, gating_weights)

        # Simple loss (encourage diversity)
        uniform_weights = torch.ones_like(gating_weights) / num_models
        loss = torch.nn.functional.mse_loss(gating_weights, uniform_weights)

        loss.backward()
        optimizer.step()

        logging.info(f"Gating epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Optimize plugin rule parameters
    logging.info("=== Optimizing Plugin Rule ===")

    gating_model.eval()
    with torch.no_grad():
        gating_weights = gating_model.forward(cal_features)
        ensemble_probs = gating_model.compute_ensemble_probs(cal_probs, gating_weights)

    plugin_optimizer = PluginOptimizer(config["plugin"])
    alpha, mu = plugin_optimizer.optimize(
        ensemble_probs, cal_targets, cal_group_ids, group_info
    )

    logging.info(f"Optimized alpha: {alpha}")
    logging.info(f"Optimized mu: {mu}")

    # Create plugin rule
    plugin_rule = plugin_optimizer.create_plugin_rule(alpha, mu, group_info)

    # Evaluate on validation set
    logging.info("=== Evaluating on Validation Set ===")

    with torch.no_grad():
        val_gating_weights = gating_model.forward(val_features)
        val_ensemble_probs = gating_model.compute_ensemble_probs(
            val_probs, val_gating_weights
        )

        predictions, rejections = plugin_rule.forward(val_ensemble_probs)

    # Compute metrics
    metrics_computer = SelectiveMetrics(
        num_groups=config["plugin"]["groups"]["num_groups"]
    )

    all_metrics = metrics_computer.compute_all_metrics(
        predictions, val_targets, rejections, val_group_ids, val_ensemble_probs
    )

    # Compute coverage-specific metrics
    coverage_levels = config["plugin"]["coverage_levels"]
    coverage_metrics = {}

    for coverage in coverage_levels:
        coverage_metrics[f"metrics_at_{coverage}"] = compute_metrics_at_coverage(
            predictions,
            val_targets,
            rejections,
            val_group_ids,
            val_ensemble_probs,
            coverage,
            config["plugin"]["groups"]["num_groups"],
        )

    # Combine results
    final_results = {**all_metrics, **coverage_metrics}

    # Add alpha and mu to results
    final_results["alpha"] = alpha
    final_results["mu"] = mu

    return final_results, gating_model, plugin_rule


def main():
    parser = argparse.ArgumentParser(description="Run PB-GSE method only")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--probs_dir", type=str, help="Directory with model probabilities"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./pbgse_outputs", help="Save directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--use_synthetic", action="store_true", help="Use synthetic data for demo"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Setup logging
    setup_logging(os.path.join(args.save_dir, "logs"))
    logging.info("Running PB-GSE method only")
    logging.info(f"Config: {config}")

    # Set seeds
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        # Load group info
        _, group_info = get_dataset_and_splits(config)

        if args.probs_dir and os.path.exists(args.probs_dir):
            # Use provided probabilities (trained models)
            probs_dir = args.probs_dir
            logging.info(f"Using existing model probabilities from: {probs_dir}")

            # Check if all required model directories exist
            required_models = ["cRT", "LDAM_DRW", "CB_Focal"]
            missing_models = []
            for model_name in required_models:
                model_dir = os.path.join(probs_dir, model_name)
                if not os.path.exists(model_dir):
                    missing_models.append(model_name)

            if missing_models:
                logging.warning(f"Missing model probabilities for: {missing_models}")
                logging.info("Falling back to synthetic data...")
                probs_dir, group_info = create_synthetic_base_models(config, device)
            else:
                logging.info("All trained model probabilities found!")

        elif args.use_synthetic:
            # Explicitly use synthetic data
            logging.info("Using synthetic data as requested...")
            probs_dir, group_info = create_synthetic_base_models(config, device)
        else:
            # Default: try to find existing probabilities, fallback to synthetic
            default_probs_dir = "./outputs/probs_calibrated"
            if os.path.exists(default_probs_dir):
                probs_dir = default_probs_dir
                logging.info(f"Found existing probabilities at: {probs_dir}")
            else:
                logging.info(
                    "No existing probabilities found, creating synthetic data..."
                )
                probs_dir, group_info = create_synthetic_base_models(config, device)

        # Run PB-GSE method
        results, gating_model, plugin_rule = run_pbgse_method(
            config, probs_dir, group_info, device
        )

        # Save results
        os.makedirs(os.path.join(args.save_dir, "results"), exist_ok=True)

        # Convert tensors for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    json_results[key] = [v.cpu().numpy().tolist() for v in value]
                else:
                    json_results[key] = value
            else:
                json_results[key] = value

        # Save results
        results_path = os.path.join(args.save_dir, "results", "pbgse_results.json")
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)

        # Save models
        gating_path = os.path.join(args.save_dir, "models", "gating_model.pth")
        os.makedirs(os.path.dirname(gating_path), exist_ok=True)
        torch.save(gating_model.state_dict(), gating_path)

        plugin_path = os.path.join(args.save_dir, "models", "plugin_params.json")
        with open(plugin_path, "w") as f:
            json.dump({"alpha": results["alpha"], "mu": results["mu"]}, f, indent=2)

        # Print results
        logging.info("=== PB-GSE Results ===")
        logging.info(f"Coverage: {results['coverage']:.3f}")
        logging.info(
            f"Balanced Selective Error: {results['balanced_selective_error']:.3f}"
        )
        logging.info(
            f"Worst-Group Selective Error: {results['worst_group_selective_error']:.3f}"
        )
        logging.info(f"AURC: {results['aurc']:.3f}")

        # Coverage-specific results
        for coverage in config["plugin"]["coverage_levels"]:
            metrics_key = f"metrics_at_{coverage}"
            if metrics_key in results:
                m = results[metrics_key]
                logging.info(f"At {coverage * 100}% coverage:")
                logging.info(f"  BSE: {m['balanced_selective_error']:.3f}")
                logging.info(f"  WGSE: {m['worst_group_selective_error']:.3f}")

        logging.info(f"Results saved to: {results_path}")
        logging.info("=== PB-GSE method completed successfully! ===")

        return 0

    except Exception as e:
        logging.error(f"PB-GSE method failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
