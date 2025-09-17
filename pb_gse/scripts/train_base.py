"""
Training script for base models
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import get_dataset_and_splits
from data.samplers import get_sampler
from models.backbones import get_backbone, EMAModel
from models.losses_lt import get_loss_function, MixUpLoss, CutMixLoss
from data.transforms import MixUp, CutMix


def setup_logging(log_dir: str, model_name: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{model_name}_train.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_class_frequencies(dataset, group_info):
    """Get class frequencies from dataset"""
    class_counts = {}
    for i in range(len(dataset)):
        _, target = dataset[i]
        if isinstance(target, torch.Tensor):
            target = target.item()
        if target not in class_counts:
            class_counts[target] = 0
        class_counts[target] += 1
    
    # Convert to list
    num_classes = max(class_counts.keys()) + 1
    frequencies = [class_counts.get(i, 0) for i in range(num_classes)]
    
    return frequencies


def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Setup data augmentation
    use_mixup = config['base_model'].get('use_mixup', False)
    use_cutmix = config['base_model'].get('use_cutmix', False)
    
    mixup = MixUp(alpha=0.2) if use_mixup else None
    cutmix = CutMix(alpha=0.2) if use_cutmix else None
    
    # Wrap criterion for augmentation
    if use_mixup and mixup is not None:
        augmented_criterion = MixUpLoss(criterion)
    elif use_cutmix and cutmix is not None:
        augmented_criterion = CutMixLoss(criterion)
    else:
        augmented_criterion = criterion
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply data augmentation
        if use_mixup and mixup is not None and torch.rand(1) < 0.5:
            inputs, targets_a, targets_b, lam = mixup(inputs, targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = augmented_criterion(outputs, targets_a, targets_b, lam)
        elif use_cutmix and cutmix is not None and torch.rand(1) < 0.5:
            inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = augmented_criterion(outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy (only for non-augmented batches)
        if not (use_mixup or use_cutmix) or torch.rand(1) >= 0.5:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def save_model_probs(model, dataloader, save_path, device):
    """Save model probabilities for ensemble"""
    model.eval()
    all_probs = []
    all_targets = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu())
            all_targets.append(targets)
            
            # Store indices for this batch
            batch_size = inputs.size(0)
            start_idx = batch_idx * dataloader.batch_size
            indices = torch.arange(start_idx, start_idx + batch_size)
            all_indices.append(indices)
    
    # Concatenate all results
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    # Save to file
    torch.save({
        'probs': all_probs,
        'targets': all_targets,
        'indices': all_indices
    }, save_path)
    
    logging.info(f"Saved probabilities to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train base model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model_config', type=str, required=True, help='Model config file')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load configs
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge configs
    config.update(model_config)
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_name = config['base_model']['name']
    
    # Setup logging
    setup_logging(os.path.join(args.save_dir, 'logs'), model_name)
    logging.info(f"Training {model_name} model")
    logging.info(f"Config: {config}")
    
    # Load dataset
    split_datasets, group_info = get_dataset_and_splits(config)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['val']
    cal_dataset = split_datasets['cal']
    test_dataset = split_datasets['test']
    
    # Get class frequencies
    class_frequencies = get_class_frequencies(train_dataset, group_info)
    logging.info(f"Class frequencies: {class_frequencies}")
    
    # Create data loaders
    train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    train_sampler = get_sampler(config, train_dataset, train_targets, group_info, 'train')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create model
    num_classes = len(class_frequencies)
    backbone_config = config['base_model'].copy()
    backbone_config['num_classes'] = num_classes
    
    model = get_backbone(backbone_config).to(device)
    
    # Setup EMA if enabled
    ema_model = None
    if config['base_model'].get('ema', False):
        ema_model = EMAModel(model, decay=config['base_model'].get('ema_decay', 0.999))
    
    # Create optimizer
    optimizer_name = config['base_model'].get('optimizer', 'sgd')
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['base_model']['lr'],
            momentum=config['base_model'].get('momentum', 0.9),
            weight_decay=config['base_model'].get('weight_decay', 5e-4)
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['base_model']['lr'],
            weight_decay=config['base_model'].get('weight_decay', 1e-4)
        )
    
    # Create scheduler
    scheduler_name = config['base_model'].get('scheduler', 'cosine')
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['base_model']['epochs']
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.1
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_acc = 0.0
    epochs = config['base_model']['epochs']
    
    for epoch in range(epochs):
        # Update loss function (for methods like LDAM with DRW)
        criterion = get_loss_function(config, class_frequencies, epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        # Update EMA
        if ema_model:
            ema_model.update(model)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        logging.info(f'Epoch {epoch+1}/{epochs}:')
        logging.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'models', f'{model_name}_best.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config
            }, save_path)
            
            logging.info(f'Saved best model with val_acc: {val_acc:.2f}%')
    
    # Load best model for probability extraction
    best_model_path = os.path.join(args.save_dir, 'models', f'{model_name}_best.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply EMA if used
    if ema_model:
        ema_model.apply_shadow()
    
    # Save probabilities for all splits
    prob_save_dir = os.path.join(args.save_dir, 'probs', model_name)
    os.makedirs(prob_save_dir, exist_ok=True)
    
    save_model_probs(model, cal_loader, os.path.join(prob_save_dir, 'cal.pth'), device)
    save_model_probs(model, val_loader, os.path.join(prob_save_dir, 'val.pth'), device)
    save_model_probs(model, test_loader, os.path.join(prob_save_dir, 'test.pth'), device)
    
    logging.info(f'Training completed for {model_name}')


if __name__ == '__main__':
    main()
