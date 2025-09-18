"""Utilities for learning the PB-GSE gating network."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


def _activation_fn(name: str) -> nn.Module:
    """Return the activation module specified by ``name``."""

    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class FeatureExtractor:
    """Create input features for the gating network.

    The extractor is designed to work both during training (where group labels are
    available) and at test time (where they are typically unknown).  Features are
    computed from the calibrated model probabilities and optionally include
    aggregated statistics such as entropy, disagreement and soft group masses.
    """

    def __init__(self, config: Dict, group_info: Optional[Dict] = None):
        self.config = config.get("features", {})
        self.group_info = group_info or {}
        self.use_group_onehot = bool(self.config.get("use_group_onehot", False))

        group_to_classes = self.group_info.get("group_to_classes", {})
        processed: Dict[int, torch.Tensor] = {}
        for group_id, class_list in group_to_classes.items():
            if not class_list:
                continue
            indices = torch.tensor(sorted(int(cls) for cls in class_list), dtype=torch.long)
            processed[int(group_id)] = indices
        self.group_class_indices = processed
        self.num_groups = max(processed.keys(), default=-1) + 1

    def _stack_probs(self, model_probs: TensorOrList) -> torch.Tensor:
        if isinstance(model_probs, list):
            return torch.stack(model_probs, dim=1)
        if model_probs.dim() == 2:
            # [B, C] for a single model – treat as ensemble of size 1
            return model_probs.unsqueeze(1)
        if model_probs.dim() == 3:
            return model_probs
        raise ValueError("model_probs must have shape [B, C] or [B, M, C]")

    def extract(
        self,
        model_probs: TensorOrList,
        group_onehot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return gating features for the provided probabilities."""

        stacked = self._stack_probs(model_probs)  # [B, M, C]
        batch_size, num_models, num_classes = stacked.shape
        device = stacked.device
        features: List[torch.Tensor] = []

        if self.config.get("use_probs", True):
            features.append(stacked.reshape(batch_size, num_models * num_classes))

        if self.config.get("use_entropy", True):
            entropy = -torch.sum(stacked * (stacked.clamp_min(1e-12).log()), dim=-1)
            features.append(entropy)

        if self.config.get("use_max_prob", True):
            max_prob = torch.max(stacked, dim=-1).values
            features.append(max_prob)

        if self.config.get("use_disagreement", True) and num_models > 1:
            variance = torch.var(stacked, dim=1)
            features.append(variance.mean(dim=-1, keepdim=True))

        if self.config.get("use_soft_group_mass", True) and self.group_class_indices:
            group_masses: List[torch.Tensor] = []
            for group_id in range(self.num_groups):
                class_indices = self.group_class_indices.get(group_id)
                if class_indices is None:
                    continue
                indices = class_indices.to(device)
                mass = stacked.index_select(-1, indices).sum(dim=-1)
                group_masses.append(mass)
            if group_masses:
                features.append(torch.cat(group_masses, dim=1))

        if self.use_group_onehot:
            if group_onehot is None:
                if self.num_groups <= 0:
                    raise ValueError(
                        "group_onehot requested but group information is unavailable"
                    )
                group_onehot = torch.zeros(batch_size, self.num_groups, device=device)
            features.append(group_onehot.float())

        if not features:
            raise ValueError("At least one feature must be enabled for the gating network")

        return torch.cat(features, dim=1)


class FeedForwardArchitecture(nn.Module):
    """Simple feed-forward network used by the gating model."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output size")

        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])
        )
        self.activation = _activation_fn(activation)
        self.dropout_layers = nn.ModuleList(
            nn.Dropout(dropout) for _ in range(max(len(dims) - 2, 0))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx < len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout_layers:
                    h = self.dropout_layers[idx](h)
        return F.softmax(h, dim=-1)

    def forward_with_params(
        self, x: torch.Tensor, params: Sequence[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        if len(params) != len(self.layers):
            raise ValueError("Number of parameter tuples does not match architecture")

        h = x
        for idx, (weight, bias) in enumerate(params):
            h = F.linear(h, weight, bias)
            if idx < len(params) - 1:
                h = self.activation(h)
                if self.dropout_layers:
                    h = self.dropout_layers[idx](h)
        return F.softmax(h, dim=-1)


class GaussianPosterior(nn.Module):
    """Diagonal Gaussian posterior for the gating network parameters."""

    def __init__(self, layer_dims: Sequence[Tuple[int, int]], initial_std: float = 0.1):
        super().__init__()
        if initial_std <= 0:
            raise ValueError("initial_std must be positive")

        self.mu_weights = nn.ParameterList()
        self.mu_biases = nn.ParameterList()
        self.log_sigma_weights = nn.ParameterList()
        self.log_sigma_biases = nn.ParameterList()

        for in_dim, out_dim in layer_dims:
            weight_mu = nn.Parameter(torch.empty(out_dim, in_dim))
            nn.init.kaiming_uniform_(weight_mu, a=math.sqrt(5))
            bias_mu = nn.Parameter(torch.zeros(out_dim))

            weight_log_sigma = nn.Parameter(
                torch.full((out_dim, in_dim), math.log(initial_std))
            )
            bias_log_sigma = nn.Parameter(torch.full((out_dim,), math.log(initial_std)))

            self.mu_weights.append(weight_mu)
            self.mu_biases.append(bias_mu)
            self.log_sigma_weights.append(weight_log_sigma)
            self.log_sigma_biases.append(bias_log_sigma)

    def sample(
        self, sample: bool = True
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        params: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for mu_w, mu_b, log_sigma_w, log_sigma_b in zip(
            self.mu_weights,
            self.mu_biases,
            self.log_sigma_weights,
            self.log_sigma_biases,
        ):
            if sample:
                eps_w = torch.randn_like(mu_w)
                eps_b = torch.randn_like(mu_b)
                weight = mu_w + torch.exp(log_sigma_w) * eps_w
                bias = mu_b + torch.exp(log_sigma_b) * eps_b
            else:
                weight = mu_w
                bias = mu_b
            params.append((weight, bias))

        return params

    def kl_divergence(self, prior_std: float) -> torch.Tensor:
        if prior_std <= 0:
            raise ValueError("prior_std must be positive")

        prior_var = prior_std**2
        log_prior_std = math.log(prior_std)
        kl = torch.tensor(0.0, device=self.mu_weights[0].device)

        for mu_w, mu_b, log_sigma_w, log_sigma_b in zip(
            self.mu_weights,
            self.mu_biases,
            self.log_sigma_weights,
            self.log_sigma_biases,
        ):
            sigma_w = torch.exp(log_sigma_w)
            sigma_b = torch.exp(log_sigma_b)

            kl += torch.sum(
                log_prior_std - log_sigma_w
                + 0.5 * ((sigma_w.pow(2) + mu_w.pow(2)) / prior_var - 1.0)
            )
            kl += torch.sum(
                log_prior_std - log_sigma_b
                + 0.5 * ((sigma_b.pow(2) + mu_b.pow(2)) / prior_var - 1.0)
            )

        return kl


class PACBayesGating(nn.Module):
    """Gating network optimised via a PAC-Bayes objective."""

    def __init__(
        self,
        input_dim: int,
        num_models: int,
        config: Dict,
        group_info: Optional[Dict] = None,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_models <= 0:
            raise ValueError("num_models must be positive")

        self.num_models = num_models
        self.config = config
        self.feature_extractor = FeatureExtractor(config, group_info)

        network_cfg = config.get("network", {})
        hidden_dims = list(network_cfg.get("hidden_dims", []))
        dims = [input_dim] + hidden_dims + [num_models]
        dropout = float(network_cfg.get("dropout", 0.0))
        activation = network_cfg.get("activation", "relu")

        self.architecture = FeedForwardArchitecture(dims, activation, dropout)
        self.pac_bayes_cfg = config.get("pac_bayes", {})
        self.method = self.pac_bayes_cfg.get("method", "deterministic").lower()

        if self.method == "gaussian":
            layer_dims = list(zip(dims[:-1], dims[1:]))
            init_std = float(self.pac_bayes_cfg.get("posterior_std_init", 0.1))
            self.posterior = GaussianPosterior(layer_dims, init_std)
            for param in self.architecture.parameters():
                param.requires_grad_(False)
        elif self.method != "deterministic":
            raise ValueError("pac_bayes.method must be either 'deterministic' or 'gaussian'")

    @property
    def is_gaussian(self) -> bool:
        return self.method == "gaussian"

    def forward(self, features: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if self.is_gaussian:
            params = self.posterior.sample(sample=sample)
            return self.architecture.forward_with_params(features, params)
        return self.architecture(features)

    def compute_ensemble_probs(
        self, model_probs: TensorOrList, gating_weights: torch.Tensor
    ) -> torch.Tensor:
        stacked = model_probs
        if isinstance(model_probs, list):
            stacked = torch.stack(model_probs, dim=1)
        if stacked.dim() != 3:
            raise ValueError("model_probs must have shape [B, M, C]")
        if stacked.size(1) != gating_weights.size(1):
            raise ValueError("Number of models in probabilities and weights mismatch")

        return torch.sum(gating_weights.unsqueeze(-1) * stacked, dim=1)

    def pac_bayes_bound(
        self,
        empirical_risk: torch.Tensor,
        n_samples: int,
        delta: float,
        L_alpha: float,
    ) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if L_alpha <= 0:
            raise ValueError("L_alpha must be positive")

        if self.is_gaussian:
            prior_std = float(self.pac_bayes_cfg.get("prior_std", 1.0))
            kl = self.posterior.kl_divergence(prior_std)
            kl_weight = float(self.config.get("kl_weight", 1.0))
            kl = kl * kl_weight
            denominator = 2.0 * max(n_samples - 1, 1)
            complexity = (kl + math.log(2.0 * n_samples / delta)) / denominator
            return empirical_risk + L_alpha * torch.sqrt(complexity)

        l2_weight = float(self.config.get("l2_weight", 0.0))
        if l2_weight == 0.0:
            return empirical_risk
        reg = sum(param.pow(2).sum() for param in self.architecture.parameters())
        return empirical_risk + l2_weight * reg


class BalancedLinearLoss:
    """Balanced linear loss used in the PAC-Bayes objective."""

    def __init__(
        self,
        alpha: Dict[int, float],
        mu: Dict[int, float],
        rejection_cost: float,
        group_info: Dict,
    ):
        if rejection_cost < 0:
            raise ValueError("rejection_cost must be non-negative")

        class_to_group_map = group_info.get("class_to_group", {})
        if not class_to_group_map:
            raise ValueError("group_info must contain class_to_group mapping")

        items = sorted((int(cls), int(group)) for cls, group in class_to_group_map.items())
        num_classes = items[-1][0] + 1
        self.class_to_group = torch.full((num_classes,), -1, dtype=torch.long)
        for cls, group in items:
            self.class_to_group[cls] = group

        group_ids = sorted({int(g) for g in alpha.keys()})
        self.alpha = torch.tensor([float(alpha[g]) for g in group_ids], dtype=torch.float32)
        self.mu = torch.tensor([float(mu[g]) for g in group_ids], dtype=torch.float32)
        if torch.any(self.alpha <= 0):
            raise ValueError("alpha values must be positive")

        self.rejection_cost = float(rejection_cost)
        self.num_groups = len(group_ids)
        inv_alpha_sum = torch.sum(1.0 / self.alpha).item()
        self.L_alpha = max(self.rejection_cost, inv_alpha_sum) + self.rejection_cost

    def to(self, device: torch.device) -> "BalancedLinearLoss":
        self.class_to_group = self.class_to_group.to(device)
        self.alpha = self.alpha.to(device)
        self.mu = self.mu.to(device)
        return self

    def __call__(
        self,
        ensemble_probs: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
        return_details: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = ensemble_probs.device
        class_to_group = self.class_to_group.to(device)
        alpha_by_group = self.alpha.to(device)
        mu_by_group = self.mu.to(device)

        alpha_per_class = alpha_by_group[class_to_group]
        mu_per_class = mu_by_group[class_to_group]

        weighted_probs = ensemble_probs / alpha_per_class.unsqueeze(0)
        predictions = torch.argmax(weighted_probs, dim=-1)
        lhs = torch.max(weighted_probs, dim=-1).values
        rhs_coeffs = (1.0 / alpha_per_class) - mu_per_class
        rhs = torch.sum(rhs_coeffs.unsqueeze(0) * ensemble_probs, dim=-1) - self.rejection_cost
        accept_mask = lhs >= rhs

        error_mask = predictions != targets
        target_alpha = alpha_by_group[group_ids]
        raw_loss = error_mask.float() * accept_mask.float() / target_alpha + (
            self.rejection_cost * (~accept_mask).float()
        )

        raw_mean = raw_loss.mean()
        normalized_mean = (raw_loss / self.L_alpha).mean()

        if return_details:
            return raw_mean, normalized_mean, predictions, accept_mask
        return raw_mean, normalized_mean


def create_gating_features(
    model_probs_list: List[List[torch.Tensor]],
    group_onehots: List[torch.Tensor],
    config: Dict,
    group_info: Optional[Dict] = None,
) -> torch.Tensor:
    extractor = FeatureExtractor(config, group_info)
    features: List[torch.Tensor] = []
    for probs, group_onehot in zip(model_probs_list, group_onehots):
        features.append(extractor.extract(probs, group_onehot))
    return torch.cat(features, dim=0)
