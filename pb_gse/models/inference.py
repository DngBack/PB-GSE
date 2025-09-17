"""
Inference and worst-group extension with Exponentiated Gradient
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from .plugin_rule import PluginRule, PluginOptimizer
from .gating import PACBayesGating


class ExponentiatedGradient:
    """Exponentiated Gradient algorithm for worst-group optimization"""
    
    def __init__(self, num_groups: int, learning_rate: float = 0.1):
        self.num_groups = num_groups
        self.learning_rate = learning_rate
        self.reset()
        
    def reset(self):
        """Reset to uniform distribution"""
        self.weights = np.ones(self.num_groups) / self.num_groups
        
    def update(self, losses: np.ndarray):
        """Update weights using exponentiated gradient"""
        # Exponentiated gradient update
        self.weights = self.weights * np.exp(self.learning_rate * losses)
        
        # Normalize to maintain simplex constraint
        self.weights = self.weights / np.sum(self.weights)
        
    def get_weights(self) -> np.ndarray:
        """Get current weights"""
        return self.weights.copy()


class WorstGroupOptimizer:
    """Worst-group extension using mixture of abstainers"""
    
    def __init__(self, config: Dict):
        self.config = config['worst_group']
        self.num_groups = config['groups']['num_groups']
        self.max_iterations = self.config['max_iterations']
        self.learning_rate = self.config['learning_rate']
        self.inner_epochs = self.config['inner_epochs']
        
        # Initialize EG optimizer
        self.eg_optimizer = ExponentiatedGradient(self.num_groups, self.learning_rate)
        
    def optimize(self, gating_model: PACBayesGating, model_probs_list: List[torch.Tensor],
                targets: torch.Tensor, group_ids: torch.Tensor, group_info: Dict,
                plugin_optimizer: PluginOptimizer, device: str = 'cuda') -> Tuple[np.ndarray, List[Dict]]:
        """Run worst-group optimization"""
        
        # Store results for each iteration
        iteration_results = []
        best_worst_group_error = float('inf')
        best_weights = None
        best_plugin_params = None
        
        for iteration in range(self.max_iterations):
            current_weights = self.eg_optimizer.get_weights()
            logging.info(f"Iteration {iteration + 1}: β = {current_weights}")
            
            # Fine-tune gating with current group weights
            self._fine_tune_gating(gating_model, model_probs_list, targets, 
                                 group_ids, current_weights, device)
            
            # Get ensemble probabilities with updated gating
            ensemble_probs = self._get_ensemble_probs(
                gating_model, model_probs_list, group_ids, group_info, device
            )
            
            # Optimize plugin rule parameters
            alpha, mu = plugin_optimizer.optimize(ensemble_probs, targets, group_ids, group_info)
            plugin_rule = plugin_optimizer.create_plugin_rule(alpha, mu, group_info)
            
            # Evaluate group-wise performance
            group_losses = self._evaluate_group_performance(
                plugin_rule, ensemble_probs, targets, group_ids
            )
            
            # Update EG weights
            self.eg_optimizer.update(group_losses)
            
            # Track best performance
            worst_group_error = np.max(group_losses)
            if worst_group_error < best_worst_group_error:
                best_worst_group_error = worst_group_error
                best_weights = current_weights.copy()
                best_plugin_params = (alpha.copy(), mu.copy())
            
            # Store iteration results
            iteration_results.append({
                'iteration': iteration + 1,
                'group_weights': current_weights.copy(),
                'group_losses': group_losses.copy(),
                'worst_group_error': worst_group_error,
                'alpha': alpha.copy(),
                'mu': mu.copy()
            })
            
            logging.info(f"Group losses: {group_losses}")
            logging.info(f"Worst-group error: {worst_group_error}")
            
        logging.info(f"Best worst-group error: {best_worst_group_error}")
        logging.info(f"Best weights: {best_weights}")
        
        return best_weights, iteration_results
    
    def _fine_tune_gating(self, gating_model: PACBayesGating, model_probs_list: List[torch.Tensor],
                         targets: torch.Tensor, group_ids: torch.Tensor, 
                         group_weights: np.ndarray, device: str):
        """Fine-tune gating model with group-weighted loss"""
        gating_model.train()
        
        # Create optimizer for gating parameters
        if gating_model.config['method'] == 'gaussian':
            params = [gating_model.posterior.mu, gating_model.posterior.log_sigma]
        else:
            params = gating_model.gating_net.parameters()
            
        optimizer = torch.optim.Adam(params, lr=1e-4)
        
        # Convert to tensors
        group_weight_tensor = torch.tensor(group_weights, dtype=torch.float32).to(device)
        
        for epoch in range(self.inner_epochs):
            optimizer.zero_grad()
            
            # Forward pass through gating
            features = gating_model.feature_extractor.extract(model_probs_list)
            gating_weights = gating_model.forward(features)
            
            # Compute ensemble probabilities
            ensemble_probs = gating_model.compute_ensemble_probs(model_probs_list, gating_weights)
            
            # Compute group-weighted loss
            group_losses = []
            for group_id in range(self.num_groups):
                group_mask = (group_ids == group_id)
                if group_mask.sum() > 0:
                    group_loss = F.cross_entropy(
                        torch.log(ensemble_probs[group_mask] + 1e-8), 
                        targets[group_mask]
                    )
                    group_losses.append(group_loss)
                else:
                    group_losses.append(torch.tensor(0.0).to(device))
            
            # Weighted combination of group losses
            total_loss = sum(group_weight_tensor[i] * group_losses[i] 
                           for i in range(self.num_groups))
            
            total_loss.backward()
            optimizer.step()
    
    def _get_ensemble_probs(self, gating_model: PACBayesGating, 
                           model_probs_list: List[torch.Tensor], 
                           group_ids: torch.Tensor, group_info: Dict, 
                           device: str) -> torch.Tensor:
        """Get ensemble probabilities from gating model"""
        gating_model.eval()
        
        with torch.no_grad():
            # Extract features
            group_onehot = F.one_hot(group_ids, num_classes=self.num_groups).float()
            features = gating_model.feature_extractor.extract(model_probs_list, group_onehot)
            
            # Get gating weights
            gating_weights = gating_model.forward(features)
            
            # Compute ensemble probabilities
            ensemble_probs = gating_model.compute_ensemble_probs(model_probs_list, gating_weights)
            
        return ensemble_probs
    
    def _evaluate_group_performance(self, plugin_rule: PluginRule, 
                                  ensemble_probs: torch.Tensor, targets: torch.Tensor,
                                  group_ids: torch.Tensor) -> np.ndarray:
        """Evaluate performance for each group"""
        predictions, rejections = plugin_rule.forward(ensemble_probs)
        accept_mask = (rejections == 0)
        
        group_losses = np.zeros(self.num_groups)
        
        for group_id in range(self.num_groups):
            group_mask = (group_ids == group_id)
            group_accept_mask = group_mask & accept_mask
            
            if group_accept_mask.sum() > 0:
                group_correct = (predictions[group_accept_mask] == targets[group_accept_mask]).float()
                group_error = 1.0 - group_correct.mean()
                group_losses[group_id] = group_error.item()
            else:
                # No accepted samples in this group
                group_losses[group_id] = 1.0
                
        return group_losses


class PBGSEInference:
    """Main inference class for PB-GSE"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.plugin_optimizer = PluginOptimizer(config['plugin'])
        
        if config['plugin']['worst_group']['enabled']:
            self.worst_group_optimizer = WorstGroupOptimizer(config)
        else:
            self.worst_group_optimizer = None
    
    def inference(self, gating_model: PACBayesGating, model_probs_list: List[torch.Tensor],
                 targets: torch.Tensor, group_info: Dict, device: str = 'cuda') -> Dict:
        """Run full inference pipeline"""
        
        # Compute group IDs
        class_to_group = group_info['class_to_group']
        group_ids = torch.tensor([
            class_to_group[target.item()] for target in targets
        ]).to(device)
        
        results = {}
        
        if self.worst_group_optimizer is not None:
            # Run worst-group optimization
            logging.info("Running worst-group optimization...")
            best_weights, iteration_results = self.worst_group_optimizer.optimize(
                gating_model, model_probs_list, targets, group_ids, 
                group_info, self.plugin_optimizer, device
            )
            
            results['worst_group'] = {
                'best_weights': best_weights,
                'iteration_results': iteration_results
            }
            
            # Use best configuration for final inference
            # Fine-tune gating one more time with best weights
            self.worst_group_optimizer._fine_tune_gating(
                gating_model, model_probs_list, targets, group_ids, best_weights, device
            )
            
        # Get final ensemble probabilities
        ensemble_probs = self._get_ensemble_probs(
            gating_model, model_probs_list, group_ids, group_info, device
        )
        
        # Optimize plugin rule parameters
        alpha, mu = self.plugin_optimizer.optimize(ensemble_probs, targets, group_ids, group_info)
        plugin_rule = self.plugin_optimizer.create_plugin_rule(alpha, mu, group_info)
        
        # Final predictions
        predictions, rejections = plugin_rule.forward(ensemble_probs)
        
        results.update({
            'ensemble_probs': ensemble_probs,
            'predictions': predictions,
            'rejections': rejections,
            'alpha': alpha,
            'mu': mu,
            'plugin_rule': plugin_rule
        })
        
        return results
    
    def _get_ensemble_probs(self, gating_model: PACBayesGating, 
                           model_probs_list: List[torch.Tensor], 
                           group_ids: torch.Tensor, group_info: Dict, 
                           device: str) -> torch.Tensor:
        """Get ensemble probabilities from gating model"""
        gating_model.eval()
        
        with torch.no_grad():
            # Extract features
            num_groups = self.config['plugin']['groups']['num_groups']
            group_onehot = F.one_hot(group_ids, num_classes=num_groups).float()
            features = gating_model.feature_extractor.extract(model_probs_list, group_onehot)
            
            # Get gating weights
            gating_weights = gating_model.forward(features)
            
            # Compute ensemble probabilities
            ensemble_probs = gating_model.compute_ensemble_probs(model_probs_list, gating_weights)
            
        return ensemble_probs
    
    def predict(self, gating_model: PACBayesGating, model_probs_list: List[torch.Tensor],
               plugin_rule: PluginRule, group_info: Dict, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on new data"""
        gating_model.eval()
        
        with torch.no_grad():
            # Dummy group IDs (we don't know true labels)
            batch_size = model_probs_list[0].size(0)
            num_groups = self.config['plugin']['groups']['num_groups']
            
            # Use uniform group distribution as default
            group_ids = torch.zeros(batch_size, dtype=torch.long).to(device)
            group_onehot = F.one_hot(group_ids, num_classes=num_groups).float()
            
            # Extract features
            features = gating_model.feature_extractor.extract(model_probs_list, group_onehot)
            
            # Get gating weights
            gating_weights = gating_model.forward(features)
            
            # Compute ensemble probabilities
            ensemble_probs = gating_model.compute_ensemble_probs(model_probs_list, gating_weights)
            
            # Apply plugin rule
            predictions, rejections = plugin_rule.forward(ensemble_probs)
            
        return predictions, rejections


def save_inference_results(results: Dict, save_path: str):
    """Save inference results"""
    # Convert tensors to numpy for saving
    save_dict = {}
    
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            save_dict[key] = value.cpu().numpy()
        elif isinstance(value, dict):
            save_dict[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    save_dict[key][sub_key] = sub_value.cpu().numpy()
                else:
                    save_dict[key][sub_key] = sub_value
        else:
            save_dict[key] = value
    
    torch.save(save_dict, save_path)


def load_inference_results(load_path: str, device: str = 'cuda') -> Dict:
    """Load inference results"""
    save_dict = torch.load(load_path, map_location=device)
    
    # Convert numpy arrays back to tensors
    results = {}
    for key, value in save_dict.items():
        if isinstance(value, np.ndarray):
            results[key] = torch.from_numpy(value).to(device)
        elif isinstance(value, dict):
            results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    results[key][sub_key] = torch.from_numpy(sub_value).to(device)
                else:
                    results[key][sub_key] = sub_value
        else:
            results[key] = value
            
    return results
