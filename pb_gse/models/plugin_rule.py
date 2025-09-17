"""
Plug-in rule (Theorem 1) and fixed-point optimization for α, μ
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging


class PluginRule:
    """Plug-in optimal rule from Theorem 1"""

    def __init__(
        self,
        alpha: Dict[int, float],
        mu: Dict[int, float],
        rejection_cost: float,
        group_info: Dict,
    ):
        self.alpha = alpha
        self.mu = mu
        self.rejection_cost = rejection_cost
        self.group_info = group_info
        self.class_to_group = group_info["class_to_group"]

    def classify(self, probs: torch.Tensor) -> torch.Tensor:
        """Classifier h_θ(x) = argmax_y p_Q_θ,y(x) / α[y]"""
        num_classes = probs.size(-1)

        # Create alpha tensor for all classes
        alpha_tensor = torch.zeros(num_classes).to(probs.device)
        for class_id in range(num_classes):
            group_id = self.class_to_group[class_id]
            alpha_tensor[class_id] = self.alpha[group_id]

        # Compute weighted probabilities
        weighted_probs = probs / alpha_tensor

        return torch.argmax(weighted_probs, dim=-1)

    def reject(self, probs: torch.Tensor) -> torch.Tensor:
        """Rejector r_θ(x) based on threshold comparison"""
        batch_size = probs.size(0)
        num_classes = probs.size(-1)

        # Create alpha and mu tensors
        alpha_tensor = torch.zeros(num_classes).to(probs.device)
        mu_tensor = torch.zeros(num_classes).to(probs.device)

        for class_id in range(num_classes):
            group_id = self.class_to_group[class_id]
            alpha_tensor[class_id] = self.alpha[group_id]
            mu_tensor[class_id] = self.mu[group_id]

        # Left-hand side: max_y p_Q_θ,y(x) / α[y]
        weighted_probs = probs / alpha_tensor
        lhs = torch.max(weighted_probs, dim=-1)[0]

        # Right-hand side: Σ_j (1/α[j] - μ[j]) p_Q_θ,j(x) - c
        rhs_coeffs = 1.0 / alpha_tensor - mu_tensor
        rhs = torch.sum(rhs_coeffs * probs, dim=-1) - self.rejection_cost

        # Reject if lhs < rhs
        return (lhs < rhs).long()

    def forward(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: classify and reject"""
        predictions = self.classify(probs)
        rejections = self.reject(probs)

        return predictions, rejections


class FixedPointSolver:
    """Fixed-point iteration for α parameters"""

    def __init__(self, max_iterations: int = 20, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve_alpha(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        mu: Dict[int, float],
        rejection_cost: float,
        group_info: Dict,
        num_groups: int = 2,
    ) -> Dict[int, float]:
        """Solve for α using fixed-point iteration"""

        # Initialize α
        alpha = {group_id: 1.0 for group_id in range(num_groups)}

        for iteration in range(self.max_iterations):
            alpha_old = alpha.copy()

            # Create plugin rule with current α
            plugin_rule = PluginRule(alpha, mu, rejection_cost, group_info)

            # Compute decisions
            predictions, rejections = plugin_rule.forward(probs)
            accept_mask = rejections == 0

            # Update α for each group
            for group_id in range(num_groups):
                group_mask = group_ids == group_id
                group_accept_mask = group_mask & accept_mask

                if group_mask.sum() > 0:
                    accept_rate = (
                        group_accept_mask.sum().float() / group_mask.sum().float()
                    )
                    alpha[group_id] = num_groups * accept_rate.item()
                else:
                    alpha[group_id] = 1.0

            # Check convergence
            converged = True
            for group_id in range(num_groups):
                if abs(alpha[group_id] - alpha_old[group_id]) > self.tolerance:
                    converged = False
                    break

            if converged:
                logging.info(f"Fixed-point converged after {iteration + 1} iterations")
                break
        else:
            logging.warning(
                f"Fixed-point did not converge after {self.max_iterations} iterations"
            )

        return alpha


class GridSearchOptimizer:
    """Grid search for optimal λ (and thus μ) parameters"""

    def __init__(self, lambda_grid: List[float], num_groups: int = 2):
        self.lambda_grid = lambda_grid
        self.num_groups = num_groups

    def optimize(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        rejection_cost: float,
        group_info: Dict,
        metric: str = "balanced_selective_error",
    ) -> Tuple[Dict[int, float], Dict[int, float], float]:
        """Optimize μ parameters using grid search"""

        best_score = float("inf")
        best_alpha = None
        best_mu = None
        best_lambda = None

        fixed_point_solver = FixedPointSolver()

        for lambda_val in self.lambda_grid:
            # Set μ = [λ, -λ] for 2 groups
            if self.num_groups == 2:
                mu = {0: lambda_val, 1: -lambda_val}  # head: λ, tail: -λ
            else:
                # For more groups, distribute λ values
                mu = {}
                for group_id in range(self.num_groups):
                    # Simple strategy: alternate signs
                    mu[group_id] = lambda_val * (1 if group_id % 2 == 0 else -1)

            # Solve for α with current μ
            alpha = fixed_point_solver.solve_alpha(
                probs,
                targets,
                group_ids,
                mu,
                rejection_cost,
                group_info,
                self.num_groups,
            )

            # Evaluate performance
            score = self._evaluate_performance(
                probs, targets, group_ids, alpha, mu, rejection_cost, group_info, metric
            )

            if score < best_score:
                best_score = score
                best_alpha = alpha.copy()
                best_mu = mu.copy()
                best_lambda = lambda_val

        logging.info(f"Best lambda: {best_lambda}, Best score: {best_score}")
        return best_alpha, best_mu, best_lambda

    def _evaluate_performance(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        alpha: Dict[int, float],
        mu: Dict[int, float],
        rejection_cost: float,
        group_info: Dict,
        metric: str,
    ) -> float:
        """Evaluate performance metric"""

        plugin_rule = PluginRule(alpha, mu, rejection_cost, group_info)
        predictions, rejections = plugin_rule.forward(probs)

        if metric == "balanced_selective_error":
            return self._compute_balanced_selective_error(
                predictions, targets, rejections, group_ids
            )
        elif metric == "worst_group_error":
            return self._compute_worst_group_error(
                predictions, targets, rejections, group_ids
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _compute_balanced_selective_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rejections: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> float:
        """Compute balanced selective error"""
        accept_mask = rejections == 0

        if accept_mask.sum() == 0:
            return 1.0  # All rejected

        group_errors = []
        unique_groups = torch.unique(group_ids)

        for group_id in unique_groups:
            group_mask = (group_ids == group_id) & accept_mask

            if group_mask.sum() > 0:
                group_correct = (predictions[group_mask] == targets[group_mask]).float()
                group_error = 1.0 - group_correct.mean()
                group_errors.append(group_error.item())
            else:
                group_errors.append(0.0)  # No samples in this group

        return np.mean(group_errors)

    def _compute_worst_group_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rejections: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> float:
        """Compute worst-group error"""
        accept_mask = rejections == 0

        if accept_mask.sum() == 0:
            return 1.0  # All rejected

        group_errors = []
        unique_groups = torch.unique(group_ids)

        for group_id in unique_groups:
            group_mask = (group_ids == group_id) & accept_mask

            if group_mask.sum() > 0:
                group_correct = (predictions[group_mask] == targets[group_mask]).float()
                group_error = 1.0 - group_correct.mean()
                group_errors.append(group_error.item())

        return max(group_errors) if group_errors else 1.0


class PluginOptimizer:
    """Main optimizer for plugin rule parameters"""

    def __init__(self, config: Dict):
        self.config = config
        self.fixed_point_solver = FixedPointSolver(
            max_iterations=config["fixed_point"]["max_iterations"],
            tolerance=config["fixed_point"]["tolerance"],
        )
        self.grid_search = GridSearchOptimizer(
            lambda_grid=config["fixed_point"]["lambda_grid"],
            num_groups=config["groups"]["num_groups"],
        )

    def optimize(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        group_info: Dict,
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Optimize α and μ parameters"""

        rejection_cost = self.config["rejection_cost"]

        # Grid search for optimal λ (and thus μ)
        best_alpha, best_mu, best_lambda = self.grid_search.optimize(
            probs, targets, group_ids, rejection_cost, group_info
        )

        logging.info(f"Optimized parameters:")
        logging.info(f"alpha: {best_alpha}")
        logging.info(f"mu: {best_mu}")
        logging.info(f"lambda: {best_lambda}")

        return best_alpha, best_mu

    def create_plugin_rule(
        self, alpha: Dict[int, float], mu: Dict[int, float], group_info: Dict
    ) -> PluginRule:
        """Create plugin rule with optimized parameters"""
        return PluginRule(alpha, mu, self.config["rejection_cost"], group_info)


def compute_group_ids(targets: torch.Tensor, group_info: Dict) -> torch.Tensor:
    """Compute group IDs for targets"""
    class_to_group = group_info["class_to_group"]
    group_ids = torch.tensor([class_to_group[target.item()] for target in targets]).to(
        targets.device
    )
    return group_ids


def evaluate_plugin_rule(
    plugin_rule: PluginRule,
    probs: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate plugin rule performance"""
    predictions, rejections = plugin_rule.forward(probs)
    accept_mask = rejections == 0

    results = {}

    # Overall metrics
    if accept_mask.sum() > 0:
        accepted_correct = (predictions[accept_mask] == targets[accept_mask]).float()
        results["selective_accuracy"] = accepted_correct.mean().item()
        results["selective_error"] = 1.0 - results["selective_accuracy"]
    else:
        results["selective_accuracy"] = 0.0
        results["selective_error"] = 1.0

    results["coverage"] = accept_mask.float().mean().item()
    results["rejection_rate"] = 1.0 - results["coverage"]

    # Group-wise metrics
    unique_groups = torch.unique(group_ids)
    group_errors = []
    group_coverage = []

    for group_id in unique_groups:
        group_mask = group_ids == group_id
        group_accept_mask = group_mask & accept_mask

        # Group coverage
        if group_mask.sum() > 0:
            group_cov = group_accept_mask.float().sum() / group_mask.float().sum()
            group_coverage.append(group_cov.item())
        else:
            group_coverage.append(0.0)

        # Group error
        if group_accept_mask.sum() > 0:
            group_correct = (
                predictions[group_accept_mask] == targets[group_accept_mask]
            ).float()
            group_error = 1.0 - group_correct.mean()
            group_errors.append(group_error.item())
        else:
            group_errors.append(1.0)  # No accepted samples

    results["balanced_selective_error"] = np.mean(group_errors)
    results["worst_group_error"] = max(group_errors) if group_errors else 1.0
    results["group_coverage"] = group_coverage
    results["group_errors"] = group_errors

    return results
