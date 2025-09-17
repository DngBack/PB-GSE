"""
Sampling strategies for long-tail datasets
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Dict, Optional


class ClassAwareSampler(Sampler):
    """Class-aware sampling: sample class uniformly, then sample within class"""
    
    def __init__(self, targets: List[int], num_samples: int, replacement: bool = True):
        self.targets = np.array(targets)
        self.num_samples = num_samples
        self.replacement = replacement
        
        # Group indices by class
        self.class_indices = {}
        for idx, target in enumerate(targets):
            if target not in self.class_indices:
                self.class_indices[target] = []
            self.class_indices[target].append(idx)
            
        self.num_classes = len(self.class_indices)
        
    def __iter__(self):
        # Sample classes uniformly
        class_samples = np.random.choice(
            list(self.class_indices.keys()), 
            size=self.num_samples, 
            replace=True
        )
        
        # Sample within each selected class
        indices = []
        for class_id in class_samples:
            class_idx_list = self.class_indices[class_id]
            idx = np.random.choice(class_idx_list)
            indices.append(idx)
            
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


class BalancedBatchSampler(Sampler):
    """Ensure each batch has balanced representation from head/tail"""
    
    def __init__(self, targets: List[int], group_info: Dict, batch_size: int, 
                 min_tail_ratio: float = 0.5, drop_last: bool = False):
        self.targets = np.array(targets)
        self.batch_size = batch_size
        self.min_tail_ratio = min_tail_ratio
        self.drop_last = drop_last
        
        # Separate head and tail indices
        class_to_group = group_info['class_to_group']
        self.head_indices = []
        self.tail_indices = []
        
        for idx, target in enumerate(targets):
            if class_to_group[target] == 0:  # head
                self.head_indices.append(idx)
            else:  # tail
                self.tail_indices.append(idx)
                
        self.head_indices = np.array(self.head_indices)
        self.tail_indices = np.array(self.tail_indices)
        
        # Calculate batch composition
        self.min_tail_per_batch = int(batch_size * min_tail_ratio)
        self.min_head_per_batch = batch_size - self.min_tail_per_batch
        
        # Calculate number of batches
        total_samples = len(targets)
        self.num_batches = total_samples // batch_size
        if not drop_last and total_samples % batch_size > 0:
            self.num_batches += 1
            
    def __iter__(self):
        # Shuffle indices
        np.random.shuffle(self.head_indices)
        np.random.shuffle(self.tail_indices)
        
        head_ptr = 0
        tail_ptr = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add minimum tail samples
            for _ in range(self.min_tail_per_batch):
                if tail_ptr < len(self.tail_indices):
                    batch_indices.append(self.tail_indices[tail_ptr])
                    tail_ptr += 1
                else:
                    # Wrap around if needed
                    tail_ptr = 0
                    batch_indices.append(self.tail_indices[tail_ptr])
                    tail_ptr += 1
                    
            # Add head samples
            for _ in range(self.min_head_per_batch):
                if head_ptr < len(self.head_indices):
                    batch_indices.append(self.head_indices[head_ptr])
                    head_ptr += 1
                else:
                    # Wrap around if needed
                    head_ptr = 0
                    batch_indices.append(self.head_indices[head_ptr])
                    head_ptr += 1
                    
            # Shuffle batch
            np.random.shuffle(batch_indices)
            yield batch_indices
            
    def __len__(self):
        return self.num_batches


class SquareRootSampler(Sampler):
    """Square-root sampling: probability ∝ √n_y"""
    
    def __init__(self, targets: List[int], num_samples: int):
        self.targets = np.array(targets)
        self.num_samples = num_samples
        
        # Calculate class frequencies
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        
        # Calculate square-root weights
        sqrt_weights = np.sqrt(class_counts)
        class_probs = sqrt_weights / sqrt_weights.sum()
        
        # Create sampling probabilities for each sample
        self.sample_weights = np.zeros(len(targets))
        for class_id, prob in zip(unique_classes, class_probs):
            class_mask = (self.targets == class_id)
            # Distribute class probability equally among class samples
            self.sample_weights[class_mask] = prob / class_counts[class_id]
            
    def __iter__(self):
        indices = np.random.choice(
            len(self.targets),
            size=self.num_samples,
            replace=True,
            p=self.sample_weights
        )
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


def get_sampler(config: Dict, dataset, targets: List[int], 
               group_info: Dict, split: str = 'train') -> Optional[Sampler]:
    """Get appropriate sampler based on config"""
    
    if split != 'train':
        return None  # Use default sampler for non-training splits
        
    sampling_config = config['data']['sampling']
    method = sampling_config['method']
    batch_size = config['data']['batch_size']
    
    if method == 'class_aware':
        return ClassAwareSampler(
            targets=targets,
            num_samples=len(dataset),
            replacement=True
        )
    elif method == 'square_root':
        return SquareRootSampler(
            targets=targets,
            num_samples=len(dataset)
        )
    elif method == 'balanced_batch':
        return BalancedBatchSampler(
            targets=targets,
            group_info=group_info,
            batch_size=batch_size,
            min_tail_ratio=sampling_config.get('balanced_batch_ratio', 0.5)
        )
    else:
        return None  # Use default sampler
