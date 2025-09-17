"""
Calibration script for base models
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import get_dataset_and_splits
from models.backbones import get_backbone
from models.calibration import ModelCalibrator


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'calibration.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_model(model_path: str, config: Dict, num_classes: int, device: str):
    """Load trained model"""
    # Create model
    backbone_config = config['base_model'].copy()
    backbone_config['num_classes'] = num_classes
    model = get_backbone(backbone_config).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def calibrate_model(model, cal_loader, group_info, config, device):
    """Calibrate model using calibration set"""
    num_classes = len(set(group_info['class_to_group'].keys()))
    num_groups = config['data']['groups']['num_groups']
    
    # Create calibrator
    calibrator = ModelCalibrator(
        model=model,
        num_classes=num_classes,
        num_groups=num_groups,
        method='temperature',  # or 'vector'
        group_aware=True
    )
    
    # Calibrate
    calibrator.calibrate(cal_loader, group_info, device)
    
    return calibrator


def save_calibrated_probs(model, calibrator, dataloader, group_info, save_path, device):
    """Save calibrated probabilities"""
    model.eval()
    calibrator.calibrator.eval()
    
    all_probs = []
    all_targets = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get group IDs
            group_ids = torch.tensor([
                group_info['class_to_group'][target.item()] 
                for target in targets
            ]).to(device)
            
            # Get calibrated probabilities
            probs = calibrator.get_calibrated_probs(inputs, group_ids)
            
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            
            # Store indices
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
    
    logging.info(f"Saved calibrated probabilities to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate base models')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory with trained models')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    setup_logging(os.path.join(args.save_dir, 'logs'))
    logging.info("Starting model calibration")
    
    # Load dataset
    split_datasets, group_info = get_dataset_and_splits(config)
    cal_dataset = split_datasets['cal']
    val_dataset = split_datasets['val']
    test_dataset = split_datasets['test']
    
    # Create data loaders
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Get number of classes
    num_classes = len(set(group_info['class_to_group'].keys()))
    
    # List of model names to calibrate
    model_names = ['cRT', 'LDAM_DRW', 'CB_Focal']  # Add more as needed
    
    for model_name in model_names:
        logging.info(f"Calibrating {model_name}")
        
        # Load model
        model_path = os.path.join(args.models_dir, f'{model_name}_best.pth')
        if not os.path.exists(model_path):
            logging.warning(f"Model {model_path} not found, skipping...")
            continue
        
        # Load model config for this specific model
        model_config_path = f"pb_gse/configs/base_{model_name.lower()}.yaml"
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            config.update(model_config)
        
        model = load_model(model_path, config, num_classes, device)
        
        # Calibrate
        calibrator = calibrate_model(model, cal_loader, group_info, config, device)
        
        # Save calibrator
        calibrator_save_path = os.path.join(args.save_dir, 'calibrators', f'{model_name}_calibrator.pth')
        os.makedirs(os.path.dirname(calibrator_save_path), exist_ok=True)
        calibrator.save_calibrator(calibrator_save_path)
        
        # Save calibrated probabilities
        prob_save_dir = os.path.join(args.save_dir, 'probs_calibrated', model_name)
        os.makedirs(prob_save_dir, exist_ok=True)
        
        save_calibrated_probs(model, calibrator, cal_loader, group_info,
                            os.path.join(prob_save_dir, 'cal.pth'), device)
        save_calibrated_probs(model, calibrator, val_loader, group_info,
                            os.path.join(prob_save_dir, 'val.pth'), device)
        save_calibrated_probs(model, calibrator, test_loader, group_info,
                            os.path.join(prob_save_dir, 'test.pth'), device)
        
        logging.info(f"Completed calibration for {model_name}")
    
    logging.info("Calibration completed for all models")


if __name__ == '__main__':
    main()
