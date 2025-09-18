"""Plug-in rule (Theorem 1) utilities for PB-GSE."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _to_int_dict(mapping: Dict) -> Dict[int, int]:
    return {int(key): int(value) for key, value in mapping.items()}


def _sorted_class_to_group(class_to_group: Dict[int, int]) -> torch.Tensor:
    processed = _to_int_dict(class_to_group)
    items = sorted(processed.items(), key=lambda kv: kv[0])
    if not items:
        raise ValueError("class_to_group mapping must not be empty")
    num_classes = items[-1][0] + 1
    mapping = torch.full((num_classes,), -1, dtype=torch.long)
    for cls, group in items:
        mapping[cls] = group
    return mapping


def _group_tensor(values: Dict[int, float]) -> torch.Tensor:
    processed = {int(k): float(v) for k, v in values.items()}
    groups = sorted(processed.keys())
    tensor = torch.tensor([processed[g] for g in groups], dtype=torch.float32)
    return tensor


class PluginRule:
    """Plug-in optimal classifier and rejector as per Theorem 1."""

    def __init__(
        self,
        alpha: Dict[int, float],
        mu: Dict[int, float],
        rejection_cost: float,
        group_info: Dict,
    ):
        self.rejection_cost = float(rejection_cost)
        self.class_to_group = _sorted_class_to_group(group_info["class_to_group"])
        self.alpha = _group_tensor(alpha)
        self.mu = _group_tensor(mu)
        if torch.any(self.alpha <= 0):
            raise ValueError("alpha values must be positive")

    def _prepare_tensors(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = probs.device
        alpha = self.alpha.to(device)
        mu = self.mu.to(device)
        class_to_group = self.class_to_group.to(device)
        return alpha, mu, class_to_group

    def classify(self, probs: torch.Tensor) -> torch.Tensor:
        alpha, _, class_to_group = self._prepare_tensors(probs)
        alpha_per_class = alpha[class_to_group]
        weighted = probs / alpha_per_class.unsqueeze(0)
        return torch.argmax(weighted, dim=-1)

    def reject(self, probs: torch.Tensor) -> torch.Tensor:
        alpha, mu, class_to_group = self._prepare_tensors(probs)
        alpha_per_class = alpha[class_to_group]
        mu_per_class = mu[class_to_group]

        weighted = probs / alpha_per_class.unsqueeze(0)
        lhs = torch.max(weighted, dim=-1).values
        rhs_coeffs = (1.0 / alpha_per_class) - mu_per_class
        rhs = torch.sum(rhs_coeffs.unsqueeze(0) * probs, dim=-1) - self.rejection_cost
        return (lhs < rhs).long()

    def forward(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = self.classify(probs)
        rejections = self.reject(probs)
        return predictions, rejections


class FixedPointSolver:
    """Fixed-point iteration for the α parameters."""

    def __init__(
        self,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        damping_factor: float = 0.3,
        eps: float = 1e-3,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping_factor = damping_factor
        self.eps = eps

    def solve_alpha(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        mu: Dict[int, float],
        rejection_cost: float,
        group_info: Dict,
        num_groups: int,
    ) -> Dict[int, float]:
        total = float(probs.size(0))
        if total == 0:
            raise ValueError("No samples provided for fixed-point iteration")

        alpha = {group_id: 1.0 for group_id in range(num_groups)}
        for iteration in range(self.max_iterations):
            old_alpha = alpha.copy()
            plugin_rule = PluginRule(alpha, mu, rejection_cost, group_info)
            _, rejections = plugin_rule.forward(probs)
            accept_mask = rejections == 0

            for group_id in range(num_groups):
                group_mask = group_ids == group_id
                accept_count = (group_mask & accept_mask).sum().item()
                joint_prob = accept_count / total
                new_alpha = max(self.eps, min(num_groups - self.eps, num_groups * joint_prob))
                alpha[group_id] = (
                    (1 - self.damping_factor) * old_alpha[group_id]
                    + self.damping_factor * new_alpha
                )

            max_change = max(abs(alpha[g] - old_alpha[g]) for g in range(num_groups))
            if max_change < self.tolerance:
                break

        return alpha


class GridSearchOptimizer:
    """Grid search over λ to determine μ and α."""

    def __init__(self, lambda_grid: Sequence[float], num_groups: int = 2):
        self.lambda_grid = list(lambda_grid)
        self.num_groups = num_groups

    def _lambda_to_mu(self, lambda_value: float) -> Dict[int, float]:
        if self.num_groups == 2:
            return {0: lambda_value, 1: -lambda_value}
        mu = {}
        for group in range(self.num_groups):
            sign = 1.0 if group % 2 == 0 else -1.0
            mu[group] = lambda_value * sign
        return mu

    def _evaluate(self, predictions, targets, rejections, group_ids, metric: str) -> float:
        accept_mask = rejections == 0
        if accept_mask.sum() == 0:
            return 1.0

        group_errors: List[float] = []
        for group_id in torch.unique(group_ids):
            mask = (group_ids == group_id) & accept_mask
            if mask.sum() == 0:
                group_errors.append(1.0)
                continue
            error = 1.0 - (predictions[mask] == targets[mask]).float().mean().item()
            group_errors.append(error)

        if metric == "balanced_selective_error":
            return float(np.mean(group_errors))
        if metric == "worst_group_error":
            return float(np.max(group_errors))
        raise ValueError(f"Unsupported metric: {metric}")

    def optimize(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        rejection_cost: float,
        group_info: Dict,
        fixed_point_solver: FixedPointSolver,
        metric: str = "balanced_selective_error",
    ) -> Tuple[Dict[int, float], Dict[int, float], float]:
        best_score = float("inf")
        best_alpha: Optional[Dict[int, float]] = None
        best_mu: Optional[Dict[int, float]] = None
        best_lambda: float = 0.0

        for lambda_value in self.lambda_grid:
            mu = self._lambda_to_mu(lambda_value)
            alpha = fixed_point_solver.solve_alpha(
                probs, targets, group_ids, mu, rejection_cost, group_info, self.num_groups
            )
            plugin_rule = PluginRule(alpha, mu, rejection_cost, group_info)
            predictions, rejections = plugin_rule.forward(probs)
            score = self._evaluate(predictions, targets, rejections, group_ids, metric)
            if score < best_score:
                best_score = score
                best_alpha = alpha
                best_mu = mu
                best_lambda = lambda_value

        if best_alpha is None or best_mu is None:
            raise RuntimeError("Grid search failed to produce parameters")

        return best_alpha, best_mu, best_lambda


class PluginOptimizer:
    """Utility wrapper for optimising plug-in rule parameters."""

    def __init__(self, config: Dict):
        self.config = config
        fixed_cfg = config.get("fixed_point", {})
        self.fixed_point_solver = FixedPointSolver(
            max_iterations=fixed_cfg.get("max_iterations", 20),
            tolerance=fixed_cfg.get("tolerance", 1e-6),
            damping_factor=fixed_cfg.get("damping_factor", 0.3),
            eps=fixed_cfg.get("eps", 1e-3),
        )
        lambda_grid = fixed_cfg.get("lambda_grid", np.linspace(-2.0, 2.0, 9))
        num_groups = config["groups"]["num_groups"]
        self.grid_search = GridSearchOptimizer(lambda_grid, num_groups)

    def optimize(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        group_info: Dict,
        metric: str = "balanced_selective_error",
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        rejection_cost = float(self.config["rejection_cost"])
        alpha, mu, _ = self.grid_search.optimize(
            probs,
            targets,
            group_ids,
            rejection_cost,
            group_info,
            self.fixed_point_solver,
            metric,
        )
        return alpha, mu

    def create_plugin_rule(
        self, alpha: Dict[int, float], mu: Dict[int, float], group_info: Dict
    ) -> PluginRule:
        return PluginRule(alpha, mu, self.config["rejection_cost"], group_info)


def compute_group_ids(targets: torch.Tensor, group_info: Dict) -> torch.Tensor:
    class_to_group = _sorted_class_to_group(group_info["class_to_group"])
    return class_to_group.to(targets.device)[targets]


def evaluate_plugin_rule(
    plugin_rule: PluginRule,
    probs: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
) -> Dict[str, float]:
    predictions, rejections = plugin_rule.forward(probs)
    accept_mask = rejections == 0

    results: Dict[str, float] = {}
    if accept_mask.sum() > 0:
        accuracy = (predictions[accept_mask] == targets[accept_mask]).float().mean().item()
        results["selective_accuracy"] = accuracy
        results["selective_error"] = 1.0 - accuracy
    else:
        results["selective_accuracy"] = 0.0
        results["selective_error"] = 1.0

    coverage = accept_mask.float().mean().item()
    results["coverage"] = coverage
    results["rejection_rate"] = 1.0 - coverage

    group_errors: List[float] = []
    group_coverages: List[float] = []
    for group in torch.unique(group_ids):
        mask = group_ids == group
        accept_group = mask & accept_mask
        if mask.sum() > 0:
            group_coverages.append(accept_group.float().sum().item() / mask.float().sum().item())
        else:
            group_coverages.append(0.0)
        if accept_group.sum() > 0:
            error = 1.0 - (
                predictions[accept_group] == targets[accept_group]
            ).float().mean().item()
            group_errors.append(error)
        else:
            group_errors.append(1.0)

    results["balanced_selective_error"] = float(np.mean(group_errors))
    results["worst_group_error"] = float(np.max(group_errors))
    results["group_coverage"] = group_coverages
    results["group_errors"] = group_errors
    return results
