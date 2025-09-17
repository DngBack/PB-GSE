"""
Data transforms and augmentations
"""

import torch
import torchvision.transforms as transforms
from typing import Tuple, Dict


class RandAugment:
    """Simplified RandAugment implementation"""
    
    def __init__(self, n: int = 2, m: int = 10):
        self.n = n
        self.m = m
        
    def __call__(self, img):
        # Simplified implementation - use torchvision's RandAugment
        return transforms.RandAugment(num_ops=self.n, magnitude=self.m)(img)


def get_transforms(config: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and test transforms based on config"""
    
    data_name = config['data']['name']
    aug_config = config['data']['augmentation']
    
    # Base normalization for CIFAR
    if 'cifar' in data_name.lower():
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        image_size = 32
    else:
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image_size = 224
    
    # Training transforms
    train_transforms = []
    
    if aug_config['train'].get('random_resized_crop', False):
        train_transforms.append(
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
        )
    else:
        train_transforms.extend([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip()
        ])
    
    # RandAugment
    if 'rand_augment' in aug_config['train']:
        ra_config = aug_config['train']['rand_augment']
        train_transforms.append(RandAugment(ra_config['n'], ra_config['m']))
    
    # Color jitter
    if aug_config['train'].get('color_jitter', 0) > 0:
        jitter_strength = aug_config['train']['color_jitter']
        train_transforms.append(
            transforms.ColorJitter(
                brightness=jitter_strength,
                contrast=jitter_strength,
                saturation=jitter_strength,
                hue=jitter_strength/2
            )
        )
    
    train_transforms.extend([
        transforms.ToTensor(),
        normalize
    ])
    
    # Test transforms
    test_transforms = []
    
    if aug_config['test'].get('center_crop', False):
        test_transforms.append(transforms.CenterCrop(image_size))
    else:
        test_transforms.append(transforms.Resize(image_size))
    
    test_transforms.extend([
        transforms.ToTensor(),
        normalize
    ])
    
    return (
        transforms.Compose(train_transforms),
        transforms.Compose(test_transforms)
    )


class MixUp:
    """MixUp data augmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix data augmentation"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        # Generate random bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()
        
        cx = torch.randint(0, W, (1,))
        cy = torch.randint(0, H, (1,))
        
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
