"""Evaluation metrics for PB-GSE."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


class SelectiveMetrics:
    """Compute selective prediction metrics with optional group information."""

    def __init__(self, num_groups: int = 2):
        self.num_groups = num_groups

    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rejections: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        metrics = self.compute_basic_metrics(predictions, targets, rejections)
        if group_ids is not None:
            metrics.update(self.compute_group_metrics(predictions, targets, rejections, group_ids))
        if probs is not None:
            metrics.update(self.compute_coverage_metrics(predictions, targets, probs))
            if group_ids is not None:
                metrics.update(
                    self.compute_calibration_metrics(predictions, targets, rejections, group_ids, probs)
                )
        return metrics

    def compute_basic_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rejections: torch.Tensor,
    ) -> Dict[str, float]:
        accept_mask = rejections == 0
        coverage = accept_mask.float().mean().item()
        if accept_mask.sum() > 0:
            acc = (predictions[accept_mask] == targets[accept_mask]).float().mean().item()
        else:
            acc = 0.0
        overall_acc = (accept_mask.float() * (predictions == targets).float()).mean().item()
        return {
            "coverage": coverage,
            "rejection_rate": 1.0 - coverage,
            "selective_accuracy": acc,
            "selective_error": 1.0 - acc,
            "overall_accuracy": overall_acc,
            "overall_error": 1.0 - overall_acc,
        }

    def compute_group_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rejections: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Dict[str, float]:
        accept_mask = rejections == 0
        group_coverage: List[float] = []
        group_errors: List[float] = []
        group_acceptance: List[float] = []

        for group in range(self.num_groups):
            group_mask = group_ids == group
            if group_mask.sum() == 0:
                group_coverage.append(0.0)
                group_errors.append(1.0)
                group_acceptance.append(0.0)
                continue

            group_accept = group_mask & accept_mask
            coverage = group_accept.float().sum().item() / group_mask.float().sum().item()
            group_coverage.append(coverage)

            if group_accept.sum() > 0:
                err = 1.0 - (predictions[group_accept] == targets[group_accept]).float().mean().item()
            else:
                err = 1.0
            group_errors.append(err)

            if accept_mask.sum() > 0:
                group_acceptance.append(group_accept.float().sum().item() / accept_mask.float().sum().item())
            else:
                group_acceptance.append(0.0)

        metrics = {
            "balanced_selective_error": float(np.mean(group_errors)),
            "worst_group_selective_error": float(np.max(group_errors)),
            "group_coverage": group_coverage,
            "group_errors": group_errors,
            "group_acceptance_rates": group_acceptance,
            "coverage_fairness": float(np.std(group_coverage)),
        }
        return metrics

    def compute_coverage_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor,
    ) -> Dict[str, float]:
        confidences = torch.max(probs, dim=1).values
        sorted_confidences, indices = torch.sort(confidences, descending=True)
        sorted_predictions = predictions[indices]
        sorted_targets = targets[indices]

        coverages: List[float] = []
        risks: List[float] = []
        n = len(sorted_predictions)
        for i in range(1, n + 1):
            coverage = i / n
            risk = 1.0 - (sorted_predictions[:i] == sorted_targets[:i]).float().mean().item()
            coverages.append(coverage)
            risks.append(risk)

        aurc = float(np.trapz(risks, coverages))
        return {"aurc": aurc, "risk_coverage_curve": (coverages, risks)}

    def compute_calibration_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rejections: torch.Tensor,
        group_ids: torch.Tensor,
        probs: torch.Tensor,
        n_bins: int = 15,
    ) -> Dict[str, float]:
        accept_mask = rejections == 0
        if accept_mask.sum() == 0:
            return {"overall_ece": 1.0, "group_ece": [1.0] * self.num_groups}

        accepted_probs = probs[accept_mask]
        accepted_targets = targets[accept_mask]
        overall_ece = self._expected_calibration_error(accepted_probs, accepted_targets, n_bins)

        group_ece: List[float] = []
        for group in range(self.num_groups):
            mask = (group_ids == group) & accept_mask
            if mask.sum() == 0:
                group_ece.append(0.0)
                continue
            group_probs = probs[mask]
            group_targets = targets[mask]
            group_ece.append(self._expected_calibration_error(group_probs, group_targets, n_bins))

        return {"overall_ece": overall_ece, "group_ece": group_ece}

    @staticmethod
    def _expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int) -> float:
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = (predictions == targets).float()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device)
        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (confidences > lower) & (confidences <= upper)
            proportion = in_bin.float().mean()
            if proportion > 0:
                accuracy = accuracies[in_bin].mean()
                avg_conf = confidences[in_bin].mean()
                ece += torch.abs(avg_conf - accuracy) * proportion
        return float(ece.item())


def compute_metrics_at_coverage(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    probs: torch.Tensor,
    target_coverage: float,
    group_ids: Optional[torch.Tensor] = None,
    num_groups: int = 2,
) -> Dict[str, float]:
    confidences = torch.max(probs, dim=1).values
    sorted_confidences, _ = torch.sort(confidences, descending=True)
    threshold_index = min(int(target_coverage * len(sorted_confidences)), len(sorted_confidences) - 1)
    threshold = sorted_confidences[threshold_index]
    rejections = (confidences < threshold).long()

    metrics = SelectiveMetrics(num_groups)
    return metrics.compute_all_metrics(predictions, targets, rejections, group_ids, probs)


class MetricsLogger:
    """Track metrics across training/evaluation epochs."""

    def __init__(self):
        self.metrics_history: List[Dict] = []

    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None, split: str = "train"):
        entry = {"epoch": epoch, "split": split, "metrics": metrics.copy()}
        self.metrics_history.append(entry)

    def get_best_metrics(self, metric_name: str, split: str = "val", maximize: bool = False):
        relevant = [m for m in self.metrics_history if m["split"] == split and metric_name in m["metrics"]]
        if not relevant:
            return {}, -1
        best = max(relevant, key=lambda m: m["metrics"][metric_name]) if maximize else min(
            relevant, key=lambda m: m["metrics"][metric_name]
        )
        return best["metrics"], best["epoch"]

    def save_metrics(self, path: str) -> None:
        torch.save(self.metrics_history, path)

    def load_metrics(self, path: str) -> None:
        self.metrics_history = torch.load(path)


def plot_risk_coverage_curves(
    curves: Sequence[Tuple[Sequence[float], Sequence[float]]],
    labels: Sequence[str],
    save_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=(10, 6))
    for (coverages, risks), label in zip(curves, labels):
        plt.plot(coverages, risks, label=label, linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title("Risk-Coverage Curves")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_metrics_table(results: Dict[str, Dict[str, float]], coverage_levels: Sequence[float]) -> str:
    header = "Method".ljust(20)
    for level in coverage_levels:
        header += f"BSE@{level}".ljust(14) + f"WGSE@{level}".ljust(14)
    header += "AURC".ljust(12) + "ECE".ljust(12)

    lines = [header, "-" * len(header)]
    for method, metrics in results.items():
        row = method.ljust(20)
        for level in coverage_levels:
            key = f"metrics_at_{level}"
            if key in metrics:
                m = metrics[key]
                row += f"{m.get('balanced_selective_error', 0.0):.3f}".ljust(14)
                row += f"{m.get('worst_group_selective_error', 0.0):.3f}".ljust(14)
            else:
                row += "N/A".ljust(14) + "N/A".ljust(14)
        row += f"{metrics.get('aurc', 0.0):.3f}".ljust(12)
        row += f"{metrics.get('overall_ece', 0.0):.3f}".ljust(12)
        lines.append(row)
    return "\n".join(lines)
