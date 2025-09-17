"""
Setup script for Google Colab
"""

import os
import sys
import subprocess


def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
        "tqdm",
    ]

    print("Installing required packages...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")

    print("✓ Installation completed!")


def setup_environment():
    """Setup environment for PB-GSE"""

    print("=== Setting up PB-GSE for Google Colab ===")

    # Install requirements
    install_requirements()

    # Create necessary directories
    directories = ["data", "outputs", "outputs/logs", "outputs/models", "outputs/probs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    # Add current directory to Python path
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
        print("✓ Added current directory to Python path")

    print("\n=== Environment setup completed! ===")
    print("You can now run:")
    print("  python pb_gse/scripts/colab_demo.py")


def main():
    setup_environment()


if __name__ == "__main__":
    main()
