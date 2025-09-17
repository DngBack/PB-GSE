"""
Long-tail dataset implementations
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional
import pickle


class LongTailDataset:
    """Base class for long-tail datasets"""
    
    def __init__(self, root: str, imbalance_factor: int = 100, seed: int = 42):
        self.root = root
        self.imbalance_factor = imbalance_factor
        self.seed = seed
        self.num_classes = None
        self.class_to_group = {}
        self.group_to_classes = {}
        
    def create_imbalanced_indices(self, targets: List[int]) -> List[int]:
        """Create imbalanced dataset indices following exponential decay"""
        np.random.seed(self.seed)
        
        # Count samples per class
        class_counts = np.bincount(targets)
        num_classes = len(class_counts)
        
        # Calculate samples per class with exponential decay
        max_samples = class_counts.max()
        imb_ratio = 1.0 / self.imbalance_factor
        
        samples_per_class = []
        for i in range(num_classes):
            n_i = int(max_samples * (imb_ratio ** (i / (num_classes - 1))))
            samples_per_class.append(max(n_i, 1))  # At least 1 sample per class
            
        # Sample indices for each class
        selected_indices = []
        for class_idx in range(num_classes):
            class_indices = [i for i, t in enumerate(targets) if t == class_idx]
            n_samples = min(samples_per_class[class_idx], len(class_indices))
            selected = np.random.choice(class_indices, n_samples, replace=False)
            selected_indices.extend(selected)
            
        return sorted(selected_indices)
    
    def create_groups(self, targets: List[int], tail_threshold: int = 50) -> Dict:
        """Create head/tail groups based on class frequency"""
        class_counts = np.bincount(targets)
        
        head_classes = []
        tail_classes = []
        
        for class_idx, count in enumerate(class_counts):
            if count > tail_threshold:
                head_classes.append(class_idx)
                self.class_to_group[class_idx] = 0  # head = 0
            else:
                tail_classes.append(class_idx)
                self.class_to_group[class_idx] = 1  # tail = 1
                
        self.group_to_classes = {0: head_classes, 1: tail_classes}
        
        return {
            'head_classes': head_classes,
            'tail_classes': tail_classes,
            'class_to_group': self.class_to_group,
            'group_to_classes': self.group_to_classes
        }


class CIFAR10LT(LongTailDataset):
    """CIFAR-10 Long-tail dataset"""
    
    def __init__(self, root: str, imbalance_factor: int = 100, seed: int = 42):
        super().__init__(root, imbalance_factor, seed)
        self.num_classes = 10
        
    def get_datasets(self, transform_train=None, transform_test=None):
        """Get train and test datasets"""
        # Load original CIFAR-10
        train_dataset = datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=transform_test
        )
        
        # Create imbalanced training set
        train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
        imbalanced_indices = self.create_imbalanced_indices(train_targets)
        
        # Create groups
        imbalanced_targets = [train_targets[i] for i in imbalanced_indices]
        group_info = self.create_groups(imbalanced_targets)
        
        # Create subset
        train_imbalanced = Subset(train_dataset, imbalanced_indices)
        
        return train_imbalanced, test_dataset, group_info


class CIFAR100LT(LongTailDataset):
    """CIFAR-100 Long-tail dataset"""
    
    def __init__(self, root: str, imbalance_factor: int = 100, seed: int = 42):
        super().__init__(root, imbalance_factor, seed)
        self.num_classes = 100
        
    def get_datasets(self, transform_train=None, transform_test=None):
        """Get train and test datasets"""
        train_dataset = datasets.CIFAR100(
            root=self.root, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=self.root, train=False, download=True, transform=transform_test
        )
        
        train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
        imbalanced_indices = self.create_imbalanced_indices(train_targets)
        
        imbalanced_targets = [train_targets[i] for i in imbalanced_indices]
        group_info = self.create_groups(imbalanced_targets)
        
        train_imbalanced = Subset(train_dataset, imbalanced_indices)
        
        return train_imbalanced, test_dataset, group_info


class DataSplitter:
    """Split dataset into train/cal/val/test"""
    
    def __init__(self, dataset, targets: List[int], splits: Dict[str, float], seed: int = 42):
        self.dataset = dataset
        self.targets = targets
        self.splits = splits
        self.seed = seed
        
    def split(self) -> Dict[str, Subset]:
        """Split dataset maintaining class balance"""
        np.random.seed(self.seed)
        
        # Group indices by class
        class_indices = {}
        for idx, target in enumerate(self.targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)
            
        # Split each class according to ratios
        split_indices = {split_name: [] for split_name in self.splits.keys()}
        
        for class_idx, indices in class_indices.items():
            np.random.shuffle(indices)
            n_samples = len(indices)
            
            start = 0
            for split_name, ratio in self.splits.items():
                end = start + int(n_samples * ratio)
                if split_name == list(self.splits.keys())[-1]:  # last split gets remainder
                    end = n_samples
                split_indices[split_name].extend(indices[start:end])
                start = end
                
        # Create subsets
        split_datasets = {}
        for split_name, indices in split_indices.items():
            split_datasets[split_name] = Subset(self.dataset, indices)
            
        return split_datasets


def save_group_info(group_info: Dict, save_path: str):
    """Save group information to file"""
    with open(save_path, 'w') as f:
        json.dump(group_info, f, indent=2)


def load_group_info(load_path: str) -> Dict:
    """Load group information from file"""
    with open(load_path, 'r') as f:
        return json.load(f)


def get_dataset_and_splits(config: Dict) -> Tuple[Dict, Dict]:
    """Main function to get dataset and splits"""
    data_name = config['data']['name']
    root = config['data']['root']
    imbalance_factor = config['data']['imbalance_factor']
    seed = config['data']['seed']
    splits = config['data']['splits']
    tail_threshold = config['data']['groups']['tail_threshold']
    
    # Create dataset
    if data_name == 'cifar10_lt':
        dataset_class = CIFAR10LT
    elif data_name == 'cifar100_lt':
        dataset_class = CIFAR100LT
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
        
    # Get transforms (will implement in transforms.py)
    from .transforms import get_transforms
    transform_train, transform_test = get_transforms(config)
    
    # Create dataset
    lt_dataset = dataset_class(root, imbalance_factor, seed)
    train_dataset, test_dataset, group_info = lt_dataset.get_datasets(
        transform_train, transform_test
    )
    
    # Split training set
    train_targets = [train_dataset.dataset[train_dataset.indices[i]][1] 
                    for i in range(len(train_dataset))]
    
    splitter = DataSplitter(train_dataset, train_targets, splits, seed)
    split_datasets = splitter.split()
    
    # Add test set
    split_datasets['test'] = test_dataset
    
    # Save group info
    os.makedirs(os.path.dirname(f"{root}/group_info.json"), exist_ok=True)
    save_group_info(group_info, f"{root}/group_info.json")
    
    return split_datasets, group_info
