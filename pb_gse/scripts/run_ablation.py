"""
Ablation study script for PB-GSE
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
import itertools

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def create_ablation_configs(base_config_path: str, output_dir: str):
    """Create configuration files for ablation study"""
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define ablation parameters
    ablations = {
        'calibration': [True, False],
        'pac_bayes_method': ['gaussian', 'deterministic'], 
        'group_aware_prior': [True, False],
        'worst_group_enabled': [True, False],
        'num_models': [2, 3, 4],
        'rejection_costs': [0.05, 0.1, 0.2, 0.5]
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    config_id = 0
    
    # Generate all combinations (subset for practical purposes)
    key_ablations = ['calibration', 'pac_bayes_method', 'group_aware_prior']
    
    for values in itertools.product(*[ablations[key] for key in key_ablations]):
        config = base_config.copy()
        
        # Apply ablation settings
        calibration, pac_bayes_method, group_aware_prior = values
        
        # Modify config based on ablation
        config['experiment']['name'] = f"ablation_{config_id:03d}"
        
        # Calibration ablation
        if not calibration:
            config['experiment']['stages']['calibrate_models'] = False
        
        # PAC-Bayes method
        config['gating']['pac_bayes']['method'] = pac_bayes_method
        config['gating']['pac_bayes']['group_aware_prior'] = group_aware_prior
        
        # Save config
        config_path = os.path.join(output_dir, f'ablation_{config_id:03d}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        config_id += 1
    
    print(f"Generated {config_id} ablation configurations in {output_dir}")
    
    return config_id


def run_ablation_experiments(config_dir: str, output_base_dir: str):
    """Run all ablation experiments"""
    
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        exp_name = config_file.replace('.yaml', '')
        output_dir = os.path.join(output_base_dir, exp_name)
        
        print(f"\n=== Running {exp_name} ===")
        
        # Run experiment
        cmd = [
            'python', 'pb_gse/scripts/run_experiment.py',
            '--config', config_path,
            '--save_dir', output_dir
        ]
        
        try:
            import subprocess
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ {exp_name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ {exp_name} failed: {e}")
            continue


def collect_ablation_results(output_base_dir: str, results_file: str):
    """Collect and summarize ablation results"""
    
    import json
    import pandas as pd
    
    results = []
    
    # Iterate through experiment directories
    for exp_dir in os.listdir(output_base_dir):
        exp_path = os.path.join(output_base_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        metrics_file = os.path.join(exp_path, 'results', 'metrics.json')
        if not os.path.exists(metrics_file):
            continue
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract key metrics
        result = {
            'experiment': exp_dir,
            'coverage': metrics.get('coverage', 0.0),
            'balanced_selective_error': metrics.get('balanced_selective_error', 1.0),
            'worst_group_selective_error': metrics.get('worst_group_selective_error', 1.0),
            'aurc': metrics.get('aurc', 1.0),
            'overall_ece': metrics.get('overall_ece', 1.0)
        }
        
        # Add coverage-specific metrics
        for coverage in [0.7, 0.8, 0.9]:
            key = f'metrics_at_{coverage}'
            if key in metrics:
                result[f'bse_at_{coverage}'] = metrics[key].get('balanced_selective_error', 1.0)
                result[f'wgse_at_{coverage}'] = metrics[key].get('worst_group_selective_error', 1.0)
        
        results.append(result)
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    
    # Print summary
    print(f"\n=== Ablation Results Summary ===")
    print(f"Total experiments: {len(results)}")
    print("\nBest performers by metric:")
    
    key_metrics = ['balanced_selective_error', 'worst_group_selective_error', 'aurc']
    for metric in key_metrics:
        if metric in df.columns:
            best_idx = df[metric].idxmin()
            best_exp = df.loc[best_idx, 'experiment']
            best_val = df.loc[best_idx, metric]
            print(f"  {metric}: {best_exp} ({best_val:.3f})")
    
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Run PB-GSE ablation study')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--output_dir', type=str, default='./ablation_outputs', help='Output directory')
    parser.add_argument('--only_generate', action='store_true', help='Only generate configs')
    parser.add_argument('--only_collect', action='store_true', help='Only collect results')
    
    args = parser.parse_args()
    
    config_dir = os.path.join(args.output_dir, 'configs')
    results_dir = os.path.join(args.output_dir, 'results')
    
    if not args.only_collect:
        # Generate ablation configurations
        print("Generating ablation configurations...")
        num_configs = create_ablation_configs(args.base_config, config_dir)
        
        if args.only_generate:
            print(f"Generated {num_configs} configurations. Exiting.")
            return 0
        
        # Run experiments
        print("Running ablation experiments...")
        run_ablation_experiments(config_dir, results_dir)
    
    # Collect results
    print("Collecting results...")
    results_file = os.path.join(args.output_dir, 'ablation_summary.csv')
    collect_ablation_results(results_dir, results_file)
    
    print("Ablation study completed!")
    return 0


if __name__ == '__main__':
    exit(main())
