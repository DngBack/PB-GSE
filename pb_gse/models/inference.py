"""Inference utilities for PB-GSE."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .gating import PACBayesGating
from .plugin_rule import PluginOptimizer, PluginRule, compute_group_ids


class PBGSEInference:
    """Run the PB-GSE inference pipeline."""

    def __init__(self, config: Dict):
        self.config = config
        self.plugin_optimizer = PluginOptimizer(config["plugin"])

    def _stack_model_probs(self, model_probs_list: List[torch.Tensor]) -> torch.Tensor:
        if not model_probs_list:
            raise ValueError("model_probs_list must contain at least one tensor")
        stacked = torch.stack(model_probs_list, dim=1)
        if stacked.dim() != 3:
            raise ValueError("Model probabilities must have shape [N, C]")
        return stacked

    def _extract_features(
        self,
        gating_model: PACBayesGating,
        stacked_probs: torch.Tensor,
        group_onehot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        extractor = gating_model.feature_extractor
        return extractor.extract(stacked_probs, group_onehot)

    def inference(
        self,
        gating_model: PACBayesGating,
        model_probs_list: List[torch.Tensor],
        targets: torch.Tensor,
        group_info: Dict,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        gating_model.eval()
        stacked_probs = self._stack_model_probs([p.to(device) for p in model_probs_list])
        targets = targets.to(device)

        group_onehot = None
        extractor = gating_model.feature_extractor
        if getattr(extractor, "use_group_onehot", False):
            num_groups = extractor.num_groups
            if num_groups <= 0:
                raise ValueError("Feature extractor expects group one-hot but num_groups is 0")
            group_onehot = torch.zeros(stacked_probs.size(0), num_groups, device=device)

        with torch.no_grad():
            features = self._extract_features(gating_model, stacked_probs, group_onehot)
            gating_weights = gating_model(features, sample=False)
            ensemble_probs = gating_model.compute_ensemble_probs(stacked_probs, gating_weights)

        group_ids = compute_group_ids(targets, group_info)
        alpha, mu = self.plugin_optimizer.optimize(ensemble_probs, targets, group_ids, group_info)
        plugin_rule = self.plugin_optimizer.create_plugin_rule(alpha, mu, group_info)
        predictions, rejections = plugin_rule.forward(ensemble_probs)

        results: Dict[str, torch.Tensor] = {
            "ensemble_probs": ensemble_probs,
            "predictions": predictions,
            "rejections": rejections,
        }
        results["alpha"] = torch.tensor([alpha[int(k)] for k in sorted(alpha.keys())], device=device)
        results["mu"] = torch.tensor([mu[int(k)] for k in sorted(mu.keys())], device=device)
        results["plugin_rule"] = plugin_rule

        return results

    def predict(
        self,
        gating_model: PACBayesGating,
        model_probs_list: List[torch.Tensor],
        plugin_rule: PluginRule,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gating_model.eval()
        stacked_probs = self._stack_model_probs([p.to(device) for p in model_probs_list])

        extractor = gating_model.feature_extractor
        group_onehot = None
        if getattr(extractor, "use_group_onehot", False) and extractor.num_groups > 0:
            group_onehot = torch.zeros(stacked_probs.size(0), extractor.num_groups, device=device)

        with torch.no_grad():
            features = self._extract_features(gating_model, stacked_probs, group_onehot)
            gating_weights = gating_model(features, sample=False)
            ensemble_probs = gating_model.compute_ensemble_probs(stacked_probs, gating_weights)
            predictions, rejections = plugin_rule.forward(ensemble_probs)
        return predictions, rejections


def save_inference_results(results: Dict, save_path: str) -> None:
    serialisable = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serialisable[key] = value.cpu()
        elif isinstance(value, PluginRule):
            serialisable[key] = {
                "alpha": value.alpha.cpu(),
                "mu": value.mu.cpu(),
            }
        else:
            serialisable[key] = value
    torch.save(serialisable, save_path)


def load_inference_results(load_path: str, device: torch.device) -> Dict:
    saved = torch.load(load_path, map_location=device)
    results: Dict = {}
    for key, value in saved.items():
        if isinstance(value, torch.Tensor):
            results[key] = value.to(device)
        else:
            results[key] = value
    return results
