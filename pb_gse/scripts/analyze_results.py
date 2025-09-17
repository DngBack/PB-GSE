"""
Analyze and debug PB-GSE results
"""

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def analyze_paper_results(results_path: str):
    """Analyze paper results and identify issues"""

    print("=== PB-GSE Results Analysis ===")

    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    overall = results["overall_metrics"]

    print(f"📊 Overall Performance:")
    print(
        f"  Coverage: {overall['coverage']:.3f} ({'LOW' if overall['coverage'] < 0.5 else 'OK'})"
    )
    print(f"  BSE: {overall['balanced_selective_error']:.3f}")
    print(f"  WGSE: {overall['worst_group_selective_error']:.3f}")

    # Group analysis
    print(f"\n🏷️ Group Analysis:")
    print(f"  Head coverage: {overall['group_coverage'][0]:.3f}")
    print(f"  Tail coverage: {overall['group_coverage'][1]:.3f}")
    print(f"  Head error: {overall['group_errors'][0]:.3f}")
    print(f"  Tail error: {overall['group_errors'][1]:.3f}")

    # Plugin parameters
    plugin_params = results["plugin_parameters"]
    print(f"\n⚙️ Plugin Parameters:")
    print(f"  Alpha: {plugin_params['alpha']}")
    print(f"  Mu: {plugin_params['mu']}")

    # Diagnosis
    print(f"\n🔍 Diagnosis:")

    if overall["coverage"] < 0.1:
        print("❌ CRITICAL: Coverage too low (<10%)")
        print("   Possible causes:")
        print("   - Plugin rule too conservative (rejection cost too low)")
        print("   - Alpha parameters too high")
        print("   - Model probabilities not well calibrated")

    if overall["balanced_selective_error"] > 0.4:
        print("❌ WARNING: BSE too high (>40%)")
        print("   Possible causes:")
        print("   - Models not trained well")
        print("   - Calibration not effective")
        print("   - Gating network not working")

    if abs(overall["group_coverage"][0] - overall["group_coverage"][1]) > 0.3:
        print("❌ WARNING: Large coverage imbalance between groups")
        print("   Possible causes:")
        print("   - Group-aware calibration not working")
        print("   - Plugin rule bias toward one group")

    # Recommendations
    print(f"\n💡 Recommendations:")
    print("1. Increase rejection_cost from 0.1 to 0.3-0.5")
    print("2. Reduce fixed_point max_iterations to avoid divergence")
    print("3. Check model training logs for convergence")
    print("4. Verify calibration effectiveness")


def suggest_fixes():
    """Suggest configuration fixes"""

    print("\n🔧 SUGGESTED FIXES:")

    print("\n1. Fix rejection cost (increase coverage):")
    print("   rejection_cost: 0.3  # was 0.1")

    print("\n2. Fix plugin optimization:")
    print("   fixed_point:")
    print("     max_iterations: 10  # was 50")
    print("     lambda_grid: [-1.0, -0.5, 0.0, 0.5, 1.0]  # smaller range")

    print("\n3. Fix base model training:")
    print("   epochs: 50  # increase from 10 for better convergence")

    print("\n4. Alternative: Use deterministic gating:")
    print("   pac_bayes:")
    print("     method: 'deterministic'  # instead of 'gaussian'")


def create_fixed_config():
    """Create fixed configuration"""

    fixed_config = {
        "data": {
            "name": "cifar10_lt",
            "root": "./data",
            "imbalance_factor": 100,
            "seed": 42,
            "splits": {"train": 0.8, "cal": 0.1, "val": 0.05, "test": 0.05},
            "groups": {"num_groups": 2, "tail_threshold": 50},
            "augmentation": {
                "train": {
                    "rand_augment": {"n": 2, "m": 10},
                    "mixup": 0.2,
                    "cutmix": 0.2,
                    "random_resized_crop": True,
                    "color_jitter": 0.1,
                },
                "test": {"center_crop": True},
            },
            "sampling": {"method": "class_aware", "balanced_batch_ratio": 0.5},
            "batch_size": 256,
            "num_workers": 2,
            "pin_memory": False,
        },
        "gating": {
            "network": {"hidden_dims": [64, 32], "dropout": 0.1, "activation": "relu"},
            "features": {
                "use_probs": True,
                "use_entropy": True,
                "use_max_prob": True,
                "use_disagreement": True,
                "use_group_onehot": True,
            },
            "pac_bayes": {
                "method": "deterministic",  # Changed from gaussian
                "prior_std": 1.0,
                "posterior_std_init": 0.1,
            },
            "epochs": 10,
            "lr": 1e-3,
            "optimizer": "adam",
            "batch_size": 256,
            "early_stopping": True,
            "patience": 5,
        },
        "plugin": {
            "rejection_cost": 0.3,  # Increased from 0.1
            "fixed_point": {
                "max_iterations": 10,  # Reduced from 50
                "tolerance": 1e-6,
                "lambda_grid": [-1.0, -0.5, 0.0, 0.5, 1.0],  # Smaller range
            },
            "groups": {"num_groups": 2, "group_names": ["head", "tail"]},
            "worst_group": {"enabled": False},  # Disable for now
            "coverage_levels": [0.7, 0.8, 0.9],
        },
        "experiment": {
            "name": "pbgse_paper_fixed",
            "seed": 42,
            "device": "cuda",
            "deterministic": True,
        },
    }

    return fixed_config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze PB-GSE results")
    parser.add_argument(
        "--results_path", type=str, default="./paper_results/paper_results.json"
    )
    parser.add_argument("--create_fixed_config", action="store_true")

    args = parser.parse_args()

    # Analyze results
    analyze_paper_results(args.results_path)

    # Suggest fixes
    suggest_fixes()

    # Create fixed config if requested
    if args.create_fixed_config:
        fixed_config = create_fixed_config()

        import yaml

        fixed_config_path = "./pb_gse/configs/experiment_fixed.yaml"
        with open(fixed_config_path, "w") as f:
            yaml.dump(fixed_config, f, default_flow_style=False, indent=2)

        print(f"\n✅ Created fixed config: {fixed_config_path}")
        print("\n🚀 To run with fixed config:")
        print(
            "python pb_gse/scripts/run_experiment.py --config pb_gse/configs/experiment_fixed.yaml --pbgse_only"
        )


if __name__ == "__main__":
    main()
