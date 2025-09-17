"""
Main experiment runner for PB-GSE
"""

import os
import sys
import torch
import yaml
import argparse
import logging
import numpy as np
from pathlib import Path
import subprocess
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import get_dataset_and_splits
from models.inference import PBGSEInference, save_inference_results
from models.metrics import SelectiveMetrics, compute_metrics_at_coverage, MetricsLogger
from models.gating import PACBayesGating


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'experiment.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_stage(stage_name: str, script_path: str, args: list):
    """Run a specific stage of the experiment"""
    logging.info(f"Running stage: {stage_name}")
    
    cmd = ['python', script_path] + args
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Stage {stage_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Stage {stage_name} failed with return code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False


def load_model_probs(probs_dir: str, model_names: list, split: str) -> tuple:
    """Load model probabilities"""
    model_probs_list = []
    targets = None
    
    for model_name in model_names:
        prob_path = os.path.join(probs_dir, model_name, f'{split}.pth')
        data = torch.load(prob_path)
        model_probs_list.append(data['probs'])
        
        if targets is None:
            targets = data['targets']
    
    return model_probs_list, targets


def run_inference_and_evaluation(config: dict, args):
    """Run inference and evaluation"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset info
    _, group_info = get_dataset_and_splits(config)
    
    # Model names
    model_names = ['cRT', 'LDAM_DRW', 'CB_Focal']
    
    # Load gating model
    gating_path = os.path.join(args.save_dir, 'gating', 'best_gating.pth')
    if not os.path.exists(gating_path):
        logging.error(f"Gating model not found: {gating_path}")
        return False
    
    gating_checkpoint = torch.load(gating_path, map_location=device)
    
    # Reconstruct gating model (simplified)
    input_dim = 100  # This should match the actual feature dimension
    num_models = len(model_names)
    gating_model = PACBayesGating(input_dim, num_models, config).to(device)
    gating_model.load_state_dict(gating_checkpoint['model_state_dict'])
    
    # Create inference engine
    inference_engine = PBGSEInference(config)
    
    # Load test data probabilities
    probs_dir = os.path.join(args.save_dir, 'probs_calibrated')
    test_probs, test_targets = load_model_probs(probs_dir, model_names, 'test')
    
    # Move to device
    test_probs = [probs.to(device) for probs in test_probs]
    test_targets = test_targets.to(device)
    
    # Run inference
    logging.info("Running PB-GSE inference...")
    results = inference_engine.inference(
        gating_model, test_probs, test_targets, group_info, device
    )
    
    # Save results
    results_path = os.path.join(args.save_dir, 'results', 'inference_results.pth')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    save_inference_results(results, results_path)
    
    # Compute metrics
    logging.info("Computing evaluation metrics...")
    
    predictions = results['predictions']
    rejections = results['rejections']
    ensemble_probs = results['ensemble_probs']
    
    # Compute group IDs
    class_to_group = group_info['class_to_group']
    group_ids = torch.tensor([
        class_to_group[target.item()] for target in test_targets
    ]).to(device)
    
    # Initialize metrics computer
    num_groups = config['plugin']['groups']['num_groups']
    metrics_computer = SelectiveMetrics(num_groups)
    
    # Compute all metrics
    all_metrics = metrics_computer.compute_all_metrics(
        predictions, test_targets, rejections, group_ids, ensemble_probs
    )
    
    # Compute metrics at specific coverage levels
    coverage_levels = config['plugin']['coverage_levels']
    coverage_metrics = {}
    
    for coverage in coverage_levels:
        coverage_metrics[f'metrics_at_{coverage}'] = compute_metrics_at_coverage(
            predictions, test_targets, rejections, group_ids, 
            ensemble_probs, coverage, num_groups
        )
    
    # Combine all metrics
    final_metrics = {**all_metrics, **coverage_metrics}
    
    # Save metrics
    metrics_path = os.path.join(args.save_dir, 'results', 'metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, torch.Tensor):
                json_metrics[key] = value.cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                json_metrics[key] = [v.cpu().numpy().tolist() for v in value]
            else:
                json_metrics[key] = value
        
        json.dump(json_metrics, f, indent=2)
    
    # Log key results
    logging.info("=== PB-GSE Results ===")
    logging.info(f"Overall Coverage: {final_metrics['coverage']:.3f}")
    logging.info(f"Balanced Selective Error: {final_metrics['balanced_selective_error']:.3f}")
    logging.info(f"Worst-Group Selective Error: {final_metrics['worst_group_selective_error']:.3f}")
    logging.info(f"AURC: {final_metrics['aurc']:.3f}")
    logging.info(f"Overall ECE: {final_metrics['overall_ece']:.3f}")
    
    # Coverage-specific results
    for coverage in coverage_levels:
        metrics_key = f'metrics_at_{coverage}'
        if metrics_key in final_metrics:
            m = final_metrics[metrics_key]
            logging.info(f"At {coverage*100}% coverage:")
            logging.info(f"  BSE: {m['balanced_selective_error']:.3f}")
            logging.info(f"  WGSE: {m['worst_group_selective_error']:.3f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run PB-GSE experiment')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--skip_training', action='store_true', help='Skip training stages')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(os.path.join(args.save_dir, 'logs'))
    logging.info("Starting PB-GSE experiment")
    logging.info(f"Config: {config}")
    
    # Set random seeds for reproducibility
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if config['experiment']['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    success = True
    
    if not args.skip_training:
        # Stage 1: Train base models
        if config['experiment']['stages']['train_base_models']:
            logging.info("=== Stage 1: Training Base Models ===")
            
            model_configs = ['base_crt.yaml', 'base_ldam.yaml', 'base_cbfocal.yaml']
            
            for model_config in model_configs:
                model_config_path = os.path.join('pb_gse', 'configs', model_config)
                stage_args = [
                    '--config', args.config,
                    '--model_config', model_config_path,
                    '--save_dir', args.save_dir,
                    '--device', args.device
                ]
                
                if not run_stage(f"Train {model_config}", 'pb_gse/scripts/train_base.py', stage_args):
                    success = False
                    break
        
        # Stage 2: Calibrate models
        if success and config['experiment']['stages']['calibrate_models']:
            logging.info("=== Stage 2: Calibrating Models ===")
            
            stage_args = [
                '--config', args.config,
                '--models_dir', os.path.join(args.save_dir, 'models'),
                '--save_dir', args.save_dir,
                '--device', args.device
            ]
            
            if not run_stage("Calibration", 'pb_gse/scripts/calibrate.py', stage_args):
                success = False
        
        # Stage 3: Train gating network
        if success and config['experiment']['stages']['train_gating']:
            logging.info("=== Stage 3: Training Gating Network ===")
            
            stage_args = [
                '--config', args.config,
                '--probs_dir', os.path.join(args.save_dir, 'probs_calibrated'),
                '--save_dir', args.save_dir,
                '--device', args.device
            ]
            
            if not run_stage("Gating Training", 'pb_gse/scripts/train_gating_pacbayes.py', stage_args):
                success = False
    
    # Stage 4: Run inference and evaluation
    if success and config['experiment']['stages']['run_plugin']:
        logging.info("=== Stage 4: Running Inference and Evaluation ===")
        
        if not run_inference_and_evaluation(config, args):
            success = False
    
    if success:
        logging.info("=== Experiment completed successfully! ===")
    else:
        logging.error("=== Experiment failed! ===")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
