import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import get_dataset_and_splits  # type: ignore
from models.gating import BalancedLinearLoss, FeatureExtractor, PACBayesGating
from models.plugin_rule import PluginOptimizer


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "gating_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def load_model_probs(probs_dir: str, model_names: List[str], split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    probs: List[torch.Tensor] = []
    targets: Optional[torch.Tensor] = None
    for name in model_names:
        path = os.path.join(probs_dir, name, f"{split}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Probability file not found: {path}")
        data = torch.load(path)
        probs.append(data["probs"].float())
        if targets is None:
            targets = data["targets"].long()
    assert targets is not None
    return probs, targets


def create_gating_dataset(
    model_probs: List[torch.Tensor],
    targets: torch.Tensor,
    group_info: Dict,
    config: Dict,
    include_group_onehot: bool = True,
) -> Tuple[TensorDataset, int, int]:
    gating_cfg = config["gating"]
    extractor = FeatureExtractor(gating_cfg, group_info)
    stacked_probs = torch.stack(model_probs, dim=1)  # [N, M, C]

    num_groups = config["plugin"]["groups"]["num_groups"]
    class_to_group = group_info["class_to_group"]
    group_ids = torch.tensor([int(class_to_group[int(t.item())]) for t in targets], dtype=torch.long)
    group_onehot = None
    if include_group_onehot and getattr(extractor, "use_group_onehot", False):
        group_onehot = F.one_hot(group_ids, num_classes=num_groups).float()

    features = extractor.extract(stacked_probs, group_onehot)
    dataset = TensorDataset(features, stacked_probs, targets, group_ids)
    input_dim = features.size(1)
    num_classes = stacked_probs.size(-1)
    return dataset, input_dim, num_classes


def train_epoch(
    gating_model: PACBayesGating,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: BalancedLinearLoss,
    delta: float,
    device: torch.device,
    mc_samples: int,
) -> Tuple[float, float]:
    gating_model.train()
    total_raw = 0.0
    total_bound = 0.0
    n_batches = 0
    loss_fn = loss_fn.to(device)
    n_samples = len(dataloader.dataset)

    for features, model_probs, targets, group_ids in dataloader:
        features = features.to(device)
        model_probs = model_probs.to(device)
        targets = targets.to(device)
        group_ids = group_ids.to(device)

        optimizer.zero_grad()
        raw_losses: List[torch.Tensor] = []

        samples = mc_samples if gating_model.is_gaussian else 1
        for _ in range(samples):
            gating_weights = gating_model(features, sample=gating_model.is_gaussian)
            ensemble = gating_model.compute_ensemble_probs(model_probs, gating_weights)
            raw_loss, _ = loss_fn(ensemble, targets, group_ids)
            raw_losses.append(raw_loss)

        empirical_risk = torch.stack(raw_losses).mean()
        bound = gating_model.pac_bayes_bound(empirical_risk, n_samples, delta, loss_fn.L_alpha)
        bound.backward()
        optimizer.step()

        total_raw += empirical_risk.item()
        total_bound += bound.item()
        n_batches += 1

    return total_raw / max(n_batches, 1), total_bound / max(n_batches, 1)


def evaluate(
    gating_model: PACBayesGating,
    dataloader: DataLoader,
    loss_fn: BalancedLinearLoss,
    device: torch.device,
) -> Tuple[float, float]:
    gating_model.eval()
    total_raw = 0.0
    total_accuracy = 0.0
    total_samples = 0
    loss_fn = loss_fn.to(device)

    with torch.no_grad():
        for features, model_probs, targets, group_ids in dataloader:
            features = features.to(device)
            model_probs = model_probs.to(device)
            targets = targets.to(device)
            group_ids = group_ids.to(device)

            gating_weights = gating_model(features, sample=False)
            ensemble = gating_model.compute_ensemble_probs(model_probs, gating_weights)
            raw_loss, _, predictions, accept_mask = loss_fn(ensemble, targets, group_ids, return_details=True)
            total_raw += raw_loss.item() * targets.size(0)
            if accept_mask.sum() > 0:
                acc = (predictions[accept_mask] == targets[accept_mask]).float().mean().item()
            else:
                acc = 0.0
            total_accuracy += acc * targets.size(0)
            total_samples += targets.size(0)

    return total_raw / max(total_samples, 1), total_accuracy / max(total_samples, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PB-GSE gating network")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--probs_dir", type=str, required=True, help="Directory of calibrated probabilities")
    parser.add_argument("--save_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device identifier")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    setup_logging(os.path.join(args.save_dir, "logs"))

    logging.info("Loading dataset metadata")
    _, group_info = get_dataset_and_splits(config)

    model_names = ["cRT", "LDAM_DRW", "CB_Focal"]
    logging.info("Loading model probabilities")
    train_probs, train_targets = load_model_probs(args.probs_dir, model_names, "train")
    val_probs, val_targets = load_model_probs(args.probs_dir, model_names, "val")
    cal_probs, cal_targets = load_model_probs(args.probs_dir, model_names, "cal")

    logging.info("Creating gating datasets")
    train_dataset, input_dim, num_classes = create_gating_dataset(train_probs, train_targets, group_info, config)
    val_dataset, _, _ = create_gating_dataset(val_probs, val_targets, group_info, config)
    cal_dataset, _, _ = create_gating_dataset(cal_probs, cal_targets, group_info, config, include_group_onehot=False)

    batch_size = int(config["gating"].get("batch_size", 512))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_models = len(model_names)
    gating_model = PACBayesGating(input_dim, num_models, config["gating"], group_info).to(device)
    if gating_model.is_gaussian:
        params = list(gating_model.posterior.parameters())
    else:
        params = gating_model.parameters()
    lr = float(config["gating"].get("lr", 1e-3))
    optimizer = optim.Adam(params, lr=lr)

    plugin_optimizer = PluginOptimizer(config["plugin"])

    logging.info("Initialising plug-in parameters from calibration set")
    cal_features, cal_probs_tensor, cal_targets_tensor, cal_group_ids = cal_dataset.tensors
    cal_features = cal_features.to(device)
    cal_probs_tensor = cal_probs_tensor.to(device)
    cal_targets_tensor = cal_targets_tensor.to(device)
    cal_group_ids = cal_group_ids.to(device)

    with torch.no_grad():
        cal_weights = gating_model(cal_features, sample=False)
        cal_ensemble = gating_model.compute_ensemble_probs(cal_probs_tensor, cal_weights)
    alpha, mu = plugin_optimizer.optimize(cal_ensemble, cal_targets_tensor, cal_group_ids, group_info)
    loss_fn = BalancedLinearLoss(alpha, mu, float(config["plugin"]["rejection_cost"]), group_info)

    epochs = int(config["gating"].get("epochs", 30))
    delta = float(config["gating"].get("confidence_threshold", 0.05))
    mc_samples = int(config["gating"].get("pac_bayes", {}).get("mc_samples", 1))

    best_val = float("inf")
    patience = config["gating"].get("patience", 5)
    patience_counter = 0

    for epoch in range(epochs):
        train_raw, train_bound = train_epoch(gating_model, train_loader, optimizer, loss_fn, delta, device, mc_samples)
        val_loss, val_acc = evaluate(gating_model, val_loader, loss_fn, device)

        logging.info(
            "Epoch %d/%d - train risk: %.4f, bound: %.4f, val risk: %.4f, val selective acc: %.4f",
            epoch + 1,
            epochs,
            train_raw,
            train_bound,
            val_loss,
            val_acc,
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_counter = 0
            save_path = os.path.join(args.save_dir, "gating", "best_gating.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": gating_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "alpha": alpha,
                    "mu": mu,
                    "input_dim": input_dim,
                    "num_models": num_models,
                    "config": config,
                },
                save_path,
            )
            logging.info("Saved gating model to %s", save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

    logging.info("Recomputing plug-in parameters with final model")
    with torch.no_grad():
        cal_weights = gating_model(cal_features, sample=False)
        cal_ensemble = gating_model.compute_ensemble_probs(cal_probs_tensor, cal_weights)
    alpha, mu = plugin_optimizer.optimize(cal_ensemble, cal_targets_tensor, cal_group_ids, group_info)

    final_path = os.path.join(args.save_dir, "gating", "final_gating.pth")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": gating_model.state_dict(),
            "alpha": alpha,
            "mu": mu,
            "input_dim": input_dim,
            "num_models": num_models,
            "config": config,
        },
        final_path,
    )
    logging.info("Saved final gating checkpoint to %s", final_path)


if __name__ == "__main__":
    main()
