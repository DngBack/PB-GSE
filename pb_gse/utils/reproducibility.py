"""
Reproducibility utilities
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility"""
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: This may reduce performance
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device_str: str = 'auto') -> torch.device:
    """Get appropriate device"""
    
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    
    return device


def print_system_info():
    """Print system information for reproducibility"""
    
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"NumPy version: {np.__version__}")
    
    # Python version
    import sys
    print(f"Python version: {sys.version}")
    
    print("=" * 30)
