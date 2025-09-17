"""
Check and fix import issues in the codebase
"""

import os
import sys
from pathlib import Path

def check_imports():
    """Check if all imports work correctly"""
    
    print("Checking imports...")
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        # Test data imports
        print("✓ Testing data imports...")
        from data.datasets import CIFAR10LT, CIFAR100LT, get_dataset_and_splits
        from data.transforms import get_transforms, MixUp, CutMix
        from data.samplers import ClassAwareSampler, BalancedBatchSampler
        
        # Test model imports
        print("✓ Testing model imports...")
        from models.backbones import ResNet32, get_backbone, EMAModel
        from models.losses_lt import get_loss_function, BalancedSoftmaxLoss, LDAMLoss
        from models.calibration import ModelCalibrator, TemperatureScaling
        from models.gating import PACBayesGating, FeatureExtractor
        from models.plugin_rule import PluginRule, PluginOptimizer, FixedPointSolver
        from models.inference import PBGSEInference, ExponentiatedGradient
        from models.metrics import SelectiveMetrics, compute_metrics_at_coverage
        
        print("✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        return False


def create_init_files():
    """Create missing __init__.py files"""
    
    directories = [
        'pb_gse',
        'pb_gse/data',
        'pb_gse/models', 
        'pb_gse/scripts'
    ]
    
    for directory in directories:
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            print(f"Creating {init_file}")
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')


def main():
    print("=== PB-GSE Import Checker ===")
    
    # Create missing __init__.py files
    create_init_files()
    
    # Check imports
    if check_imports():
        print("\n✓ All checks passed! The codebase is ready to use.")
    else:
        print("\n✗ Some imports failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
