"""
Training script for gating network using PAC-Bayes bound
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import get_dataset_and_splits
from models.gating import PACBayesGating, FeatureExtractor, BalancedLinearLoss
from models.plugin_rule import PluginOptimizer, compute_group_ids


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "gating_training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_model_probs(probs_dir: str, model_names: list, split: str = "train") -> tuple:
    """Load model probabilities for ensemble"""
    model_probs_list = []
    targets = None

    for model_name in model_names:
        prob_path = os.path.join(probs_dir, model_name, f"{split}.pth")
        if not os.path.exists(prob_path):
            raise FileNotFoundError(f"Probabilities not found: {prob_path}")

        data = torch.load(prob_path)
        probs = data["probs"]
        model_probs_list.append(probs)

        # Use targets from first model (should be same for all)
        if targets is None:
            targets = data["targets"]

    return model_probs_list, targets


def create_gating_dataset(
    model_probs_list: list, targets: torch.Tensor, group_info: dict, config: dict
) -> TensorDataset:
    """Create dataset for gating network training"""

    # Extract features
    feature_extractor = FeatureExtractor(config["gating"])

    # Create group one-hot encodings
    num_groups = config["plugin"]["groups"]["num_groups"]
    class_to_group = group_info["class_to_group"]

    group_ids = torch.tensor([class_to_group[target.item()] for target in targets])
    group_onehot = torch.nn.functional.one_hot(
        group_ids, num_classes=num_groups
    ).float()

    # Extract features
    features = feature_extractor.extract(model_probs_list, group_onehot)

    return TensorDataset(features, targets, group_ids), features.size(1)


def train_gating_epoch(
    gating_model,
    dataloader,
    optimizer,
    alpha,
    mu,
    rejection_cost,
    group_info,
    config,
    device,
    epoch=0,
):
    """Train gating network for one epoch"""
    gating_model.train()

    total_loss = 0.0
    total_bound = 0.0
    n_samples = len(dataloader.dataset)

    # Create balanced linear loss
    balanced_loss_fn = BalancedLinearLoss(alpha, mu, rejection_cost, group_info)

    for batch_idx, (features, targets, group_ids) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)
        group_ids = group_ids.to(device)

        optimizer.zero_grad()

        # Forward pass through gating
        if config["gating"]["pac_bayes"]["method"] == "gaussian":
            # Monte Carlo sampling for Gaussian posterior
            num_samples = 5  # Number of MC samples
            batch_losses = []

            for _ in range(num_samples):
                gating_weights = gating_model.forward(features)

                # Reconstruct model probabilities (dummy for this batch)
                # In practice, you'd need to store and pass the actual model probs
                # For now, we'll use a simplified version
                batch_size, num_classes = features.size(0), 10  # Assuming CIFAR-10
                dummy_probs = [
                    torch.softmax(torch.randn(batch_size, num_classes), dim=1).to(
                        device
                    )
                    for _ in range(gating_model.num_models)
                ]

                # Compute ensemble probabilities
                ensemble_probs = gating_model.compute_ensemble_probs(
                    dummy_probs, gating_weights
                )

                # Compute balanced linear loss
                loss = balanced_loss_fn(ensemble_probs, targets, group_ids)
                batch_losses.append(loss)

            # Average over MC samples
            empirical_loss = torch.stack(batch_losses).mean()

        else:
            # Deterministic gating
            gating_weights = gating_model.forward(features)

            # Dummy ensemble probabilities (replace with actual computation)
            batch_size, num_classes = features.size(0), 10
            dummy_probs = [
                torch.softmax(torch.randn(batch_size, num_classes), dim=1).to(device)
                for _ in range(gating_model.num_models)
            ]

            ensemble_probs = gating_model.compute_ensemble_probs(
                dummy_probs, gating_weights
            )
            empirical_loss = balanced_loss_fn(ensemble_probs, targets, group_ids)

        # Compute PAC-Bayes bound
        delta = config["gating"]["confidence_threshold"]
        pac_bayes_bound = gating_model.pac_bayes_bound(empirical_loss, n_samples, delta)

        # Backward pass
        pac_bayes_bound.backward()
        optimizer.step()

        total_loss += empirical_loss.item()
        total_bound += pac_bayes_bound.item()

    avg_loss = total_loss / len(dataloader)
    avg_bound = total_bound / len(dataloader)

    return avg_loss, avg_bound


def validate_gating(
    gating_model, dataloader, alpha, mu, rejection_cost, group_info, device
):
    """Validate gating network"""
    gating_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    balanced_loss_fn = BalancedLinearLoss(alpha, mu, rejection_cost, group_info)

    with torch.no_grad():
        for features, targets, group_ids in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            group_ids = group_ids.to(device)

            gating_weights = gating_model.forward(features)

            # Dummy computation (replace with actual)
            batch_size, num_classes = features.size(0), 10
            dummy_probs = [
                torch.softmax(torch.randn(batch_size, num_classes), dim=1).to(device)
                for _ in range(gating_model.num_models)
            ]

            ensemble_probs = gating_model.compute_ensemble_probs(
                dummy_probs, gating_weights
            )
            loss = balanced_loss_fn(ensemble_probs, targets, group_ids)

            total_loss += loss.item()

            # Simple accuracy computation
            _, predicted = ensemble_probs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train gating network with PAC-Bayes")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--probs_dir",
        type=str,
        required=True,
        help="Directory with model probabilities",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./outputs", help="Save directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Setup logging
    setup_logging(os.path.join(args.save_dir, "logs"))
    logging.info("Starting gating network training")

    # Load dataset for group info
    _, group_info = get_dataset_and_splits(config)

    # Model names (should match trained models)
    model_names = ["cRT", "LDAM_DRW", "CB_Focal"]

    # Load model probabilities
    logging.info("Loading model probabilities...")
    train_probs, train_targets = load_model_probs(args.probs_dir, model_names, "train")
    val_probs, val_targets = load_model_probs(args.probs_dir, model_names, "val")
    cal_probs, cal_targets = load_model_probs(args.probs_dir, model_names, "cal")

    # Create gating datasets
    train_dataset, input_dim = create_gating_dataset(
        train_probs, train_targets, group_info, config
    )
    val_dataset, _ = create_gating_dataset(val_probs, val_targets, group_info, config)
    cal_dataset, _ = create_gating_dataset(cal_probs, cal_targets, group_info, config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["gating"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["gating"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    cal_loader = DataLoader(
        cal_dataset,
        batch_size=config["gating"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Create gating model
    num_models = len(model_names)
    gating_model = PACBayesGating(input_dim, num_models, config).to(device)

    # Create optimizer
    if config["gating"]["pac_bayes"]["method"] == "gaussian":
        params = [gating_model.posterior.mu, gating_model.posterior.log_sigma]
    else:
        params = gating_model.gating_net.parameters()

    optimizer = optim.Adam(params, lr=config["gating"]["lr"])

    # Initialize plugin rule optimizer
    plugin_optimizer = PluginOptimizer(config["plugin"])

    # Get initial α, μ parameters using calibration set
    logging.info("Computing initial α, μ parameters...")
    cal_features = cal_dataset.tensors[0].to(device)
    cal_targets = cal_dataset.tensors[1].to(device)
    cal_group_ids = cal_dataset.tensors[2].to(device)

    # Dummy ensemble probabilities for initial optimization
    with torch.no_grad():
        gating_weights = gating_model.forward(cal_features)
        dummy_probs = [
            torch.softmax(torch.randn(cal_features.size(0), 10), dim=1).to(device)
            for _ in range(num_models)
        ]
        ensemble_probs = gating_model.compute_ensemble_probs(
            dummy_probs, gating_weights
        )

    alpha, mu = plugin_optimizer.optimize(
        ensemble_probs, cal_targets, cal_group_ids, group_info
    )
    rejection_cost = config["plugin"]["rejection_cost"]

    logging.info(f"Initial α: {alpha}")
    logging.info(f"Initial μ: {mu}")

    # Training loop
    epochs = config["gating"]["epochs"]
    best_val_loss = float("inf")
    patience = config["gating"].get("patience", 5)
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss, train_bound = train_gating_epoch(
            gating_model,
            train_loader,
            optimizer,
            alpha,
            mu,
            rejection_cost,
            group_info,
            config,
            device,
            epoch,
        )

        # Validate
        val_loss, val_acc = validate_gating(
            gating_model, val_loader, alpha, mu, rejection_cost, group_info, device
        )

        logging.info(f"Epoch {epoch + 1}/{epochs}:")
        logging.info(f"  Train Loss: {train_loss:.4f}, Train Bound: {train_bound:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_path = os.path.join(args.save_dir, "gating", "best_gating.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(
                {
                    "model_state_dict": gating_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "alpha": alpha,
                    "mu": mu,
                    "config": config,
                },
                save_path,
            )

            logging.info(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    logging.info("Gating network training completed")


if __name__ == "__main__":
    main()
