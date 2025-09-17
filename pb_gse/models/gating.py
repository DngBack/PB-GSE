"""
Gating network and PAC-Bayes bound optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class GatingNetwork(nn.Module):
    """MLP-based gating network for ensemble weighting"""

    def __init__(
        self,
        input_dim: int,
        num_models: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_models = num_models

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_models))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """Forward pass returning softmax weights"""
        logits = self.network(x)
        return F.softmax(logits, dim=1)


class FeatureExtractor:
    """Extract features for gating network input"""

    def __init__(self, config: Dict):
        self.config = config["features"]

    def extract(
        self,
        model_probs: List[torch.Tensor],
        group_onehot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features from model predictions"""
        features = []

        # Concatenate all model probabilities
        if self.config.get("use_probs", True):
            concat_probs = torch.cat(model_probs, dim=1)  # [batch, M*C]
            features.append(concat_probs)

        # Entropy of each model
        if self.config.get("use_entropy", True):
            entropies = []
            for probs in model_probs:
                entropy = -torch.sum(
                    probs * torch.log(probs + 1e-8), dim=1, keepdim=True
                )
                entropies.append(entropy)
            entropy_features = torch.cat(entropies, dim=1)  # [batch, M]
            features.append(entropy_features)

        # Max probability of each model
        if self.config.get("use_max_prob", True):
            max_probs = []
            for probs in model_probs:
                max_prob = torch.max(probs, dim=1, keepdim=True)[0]
                max_probs.append(max_prob)
            max_prob_features = torch.cat(max_probs, dim=1)  # [batch, M]
            features.append(max_prob_features)

        # Disagreement between models (variance)
        if self.config.get("use_disagreement", True):
            if len(model_probs) > 1:
                stacked_probs = torch.stack(model_probs, dim=1)  # [batch, M, C]
                disagreement = torch.var(stacked_probs, dim=1)  # [batch, C]
                disagreement_scalar = torch.mean(
                    disagreement, dim=1, keepdim=True
                )  # [batch, 1]
                features.append(disagreement_scalar)

        # Group one-hot encoding
        if self.config.get("use_group_onehot", True) and group_onehot is not None:
            features.append(group_onehot)

        return torch.cat(features, dim=1)


class GaussianPosterior(nn.Module):
    """Gaussian posterior Q = N(μ, σ²I) for PAC-Bayes"""

    def __init__(self, param_dim: int, initial_std: float = 0.1):
        super().__init__()
        self.param_dim = param_dim
        self.mu = nn.Parameter(torch.zeros(param_dim))
        self.log_sigma = nn.Parameter(torch.log(torch.ones(param_dim) * initial_std))

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def sample(self, num_samples: int = 1):
        """Sample parameters from posterior"""
        if num_samples == 1:
            epsilon = torch.randn_like(self.mu)
            return self.mu + self.sigma * epsilon
        else:
            epsilon = torch.randn(num_samples, self.param_dim).to(self.mu.device)
            return self.mu.unsqueeze(0) + self.sigma.unsqueeze(0) * epsilon

    def kl_divergence(self, prior_std: float = 1.0):
        """KL divergence KL(Q||Π) where Π = N(0, σ₀²I)"""
        prior_var = prior_std**2
        posterior_var = self.sigma**2

        kl = 0.5 * torch.sum(
            posterior_var / prior_var
            + self.mu**2 / prior_var
            - 1
            - torch.log(posterior_var / prior_var)
        )
        return kl


class PACBayesGating(nn.Module):
    """PAC-Bayes Gating with bound optimization"""

    def __init__(self, input_dim: int, num_models: int, config: Dict):
        super().__init__()

        self.input_dim = input_dim
        self.num_models = num_models
        self.config = config["pac_bayes"]

        # Feature extractor
        self.feature_extractor = FeatureExtractor(config)

        if self.config["method"] == "gaussian":
            # Gaussian posterior for parameters
            gating_config = config["network"]
            hidden_dims = gating_config["hidden_dims"]

            # Calculate total parameter dimension
            total_params = input_dim * hidden_dims[0]
            for i in range(len(hidden_dims) - 1):
                total_params += hidden_dims[i] * hidden_dims[i + 1]
            total_params += hidden_dims[-1] * num_models
            # Add bias terms
            total_params += sum(hidden_dims) + num_models

            self.posterior = GaussianPosterior(
                total_params, self.config["posterior_std_init"]
            )

            # Store network structure for parameter reconstruction
            self.layer_shapes = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layer_shapes.append((prev_dim, hidden_dim))
                prev_dim = hidden_dim
            self.layer_shapes.append((prev_dim, num_models))

        else:
            # Deterministic gating network
            self.gating_net = GatingNetwork(
                input_dim,
                num_models,
                config["network"]["hidden_dims"],
                config["network"]["dropout"],
                config["network"]["activation"],
            )

    def _reconstruct_network_params(
        self, flat_params: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Reconstruct network weights and biases from flattened parameters"""
        params = []
        start_idx = 0

        for in_dim, out_dim in self.layer_shapes:
            # Weight matrix
            weight_size = in_dim * out_dim
            weight = flat_params[start_idx : start_idx + weight_size].view(
                out_dim, in_dim
            )
            start_idx += weight_size

            # Bias vector
            bias = flat_params[start_idx : start_idx + out_dim]
            start_idx += out_dim

            params.append((weight, bias))

        return params

    def _forward_with_params(
        self, x: torch.Tensor, params: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass with given parameters"""
        h = x
        for i, (weight, bias) in enumerate(params):
            h = F.linear(h, weight, bias)
            if i < len(params) - 1:  # Not the last layer
                h = F.relu(h)
        return F.softmax(h, dim=1)

    def forward(self, features: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Forward pass through gating network"""
        if self.config["method"] == "gaussian":
            if num_samples == 1:
                # Single sample
                sampled_params = self.posterior.sample()
                network_params = self._reconstruct_network_params(sampled_params)
                return self._forward_with_params(features, network_params)
            else:
                # Multiple samples for Monte Carlo estimation
                sampled_params = self.posterior.sample(num_samples)
                outputs = []
                for i in range(num_samples):
                    network_params = self._reconstruct_network_params(sampled_params[i])
                    output = self._forward_with_params(features, network_params)
                    outputs.append(output)
                return torch.stack(outputs, dim=0).mean(dim=0)
        else:
            # Deterministic forward
            return self.gating_net(features)

    def compute_ensemble_probs(
        self, model_probs: List[torch.Tensor], gating_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute ensemble probabilities p_Q_θ(x) = Σ_m w_m(x) p^{(m)}(x)"""
        ensemble_probs = torch.zeros_like(model_probs[0])

        for m, probs in enumerate(model_probs):
            ensemble_probs += gating_weights[:, m : m + 1] * probs

        return ensemble_probs

    def pac_bayes_bound(
        self, empirical_loss: torch.Tensor, n_samples: int, delta: float = 0.05
    ) -> torch.Tensor:
        """Compute PAC-Bayes bound"""
        if self.config["method"] == "gaussian":
            # KL divergence term
            prior_std = self.config["prior_std"]
            kl_div = self.posterior.kl_divergence(prior_std)

            # PAC-Bayes bound
            bound = empirical_loss + torch.sqrt(
                (kl_div + math.log(2 * n_samples / delta)) / (2 * (n_samples - 1))
            )
        else:
            # Deterministic case: use L2 regularization as KL surrogate
            l2_reg = 0
            for param in self.gating_net.parameters():
                l2_reg += torch.sum(param**2)

            bound = empirical_loss + self.config.get("l2_weight", 0.01) * l2_reg

        return bound


class BalancedLinearLoss:
    """Balanced linear loss for PAC-Bayes optimization"""

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

    def __call__(
        self,
        ensemble_probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute balanced linear loss"""
        batch_size = ensemble_probs.size(0)
        losses = []

        for i in range(batch_size):
            probs = ensemble_probs[i]
            target = targets[i].item()
            group_id = group_ids[i].item()

            # Compute classifier decision h_θ(x)
            alpha_weighted_probs = probs / self.alpha[target]
            pred_class = torch.argmax(alpha_weighted_probs).item()

            # Compute rejection decision r_θ(x)
            lhs = torch.max(alpha_weighted_probs)

            rhs = 0
            for class_id in range(len(probs)):
                class_group = self.group_info["class_to_group"][class_id]
                rhs += (1 / self.alpha[class_group] - self.mu[class_group]) * probs[
                    class_id
                ]
            rhs -= self.rejection_cost

            accept = lhs >= rhs

            # Compute loss
            if accept:
                if pred_class != target:
                    loss = 1.0 / self.alpha[group_id]
                else:
                    loss = 0.0
            else:
                loss = self.rejection_cost

            losses.append(loss)

        return torch.tensor(losses).mean().to(ensemble_probs.device)


def create_gating_features(
    model_probs_list: List[List[torch.Tensor]],
    group_onehots: List[torch.Tensor],
    config: Dict,
) -> torch.Tensor:
    """Create gating features from model probabilities"""
    feature_extractor = FeatureExtractor(config)

    all_features = []
    for i, model_probs in enumerate(model_probs_list):
        group_onehot = group_onehots[i] if i < len(group_onehots) else None
        features = feature_extractor.extract(model_probs, group_onehot)
        all_features.append(features)

    return torch.cat(all_features, dim=0)
