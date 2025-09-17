# PB-GSE: PAC-Bayes Group-Selective Ensemble for Long-tail Classification with Abstention

This repository implements the PB-GSE method for long-tail classification with selective prediction (abstention). The method uses PAC-Bayes theory to optimize ensemble gating while maintaining group-fairness across head and tail classes.

## Overview

PB-GSE addresses the challenge of selective prediction on long-tail datasets by:

1. **Diverse Base Models**: Training multiple models with different long-tail-aware loss functions (cRT, LDAM-DRW, CB-Focal)
2. **Group-aware Calibration**: Calibrating models separately for head and tail classes
3. **PAC-Bayes Gating**: Learning ensemble weights that minimize a PAC-Bayes bound on balanced selective risk
4. **Plug-in Optimal Rule**: Using Theorem 1 to make optimal classification and rejection decisions
5. **Worst-group Extension**: Optional EG-based optimization for worst-group performance

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/PB-GSE.git
cd PB-GSE

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Demo

Run a quick demo on CIFAR-10-LT:

```bash
python pb_gse/scripts/demo.py
```

### Full Experiment

Run the complete PB-GSE pipeline:

```bash
python pb_gse/scripts/run_experiment.py --config pb_gse/configs/experiment.yaml
```

## Project Structure

```
pb_gse/
├── configs/                 # Configuration files
│   ├── data.yaml           # Data configuration
│   ├── base_*.yaml         # Base model configurations
│   ├── gating.yaml         # Gating network configuration
│   ├── plugin.yaml         # Plugin rule configuration
│   └── experiment.yaml     # Main experiment configuration
├── data/                   # Data processing modules
│   ├── datasets.py         # Long-tail dataset implementations
│   ├── transforms.py       # Data augmentations
│   └── samplers.py         # Sampling strategies
├── models/                 # Model implementations
│   ├── backbones.py        # Neural network architectures
│   ├── losses_lt.py        # Long-tail loss functions
│   ├── calibration.py      # Calibration methods
│   ├── gating.py           # Gating network and PAC-Bayes
│   ├── plugin_rule.py      # Theorem 1 implementation
│   ├── inference.py        # Inference and worst-group EG
│   └── metrics.py          # Evaluation metrics
├── scripts/                # Training and evaluation scripts
│   ├── train_base.py       # Train base models
│   ├── calibrate.py        # Calibrate models
│   ├── train_gating_pacbayes.py  # Train gating network
│   ├── run_experiment.py   # Main experiment runner
│   └── demo.py             # Quick demo
└── outputs/                # Output directory
    ├── models/             # Trained models
    ├── probs/              # Model probabilities
    ├── gating/             # Gating networks
    └── results/            # Final results
```

## Usage

### 1. Training Base Models

Train multiple base models with different loss functions:

```bash
# Train cRT model
python pb_gse/scripts/train_base.py \
    --config pb_gse/configs/experiment.yaml \
    --model_config pb_gse/configs/base_crt.yaml

# Train LDAM-DRW model
python pb_gse/scripts/train_base.py \
    --config pb_gse/configs/experiment.yaml \
    --model_config pb_gse/configs/base_ldam.yaml

# Train CB-Focal model
python pb_gse/scripts/train_base.py \
    --config pb_gse/configs/experiment.yaml \
    --model_config pb_gse/configs/base_cbfocal.yaml
```

### 2. Model Calibration

Calibrate models using group-aware temperature scaling:

```bash
python pb_gse/scripts/calibrate.py \
    --config pb_gse/configs/experiment.yaml \
    --models_dir outputs/models \
    --save_dir outputs
```

### 3. Gating Network Training

Train the PAC-Bayes gating network:

```bash
python pb_gse/scripts/train_gating_pacbayes.py \
    --config pb_gse/configs/experiment.yaml \
    --probs_dir outputs/probs_calibrated \
    --save_dir outputs
```

### 4. Full Pipeline

Run the complete experiment:

```bash
python pb_gse/scripts/run_experiment.py \
    --config pb_gse/configs/experiment.yaml \
    --save_dir outputs
```

## Configuration

### Data Configuration (`data.yaml`)

- Dataset selection (CIFAR-10/100-LT, ImageNet-LT)
- Imbalance factors and grouping
- Data augmentation settings
- Sampling strategies

### Base Model Configuration (`base_*.yaml`)

- Model architectures
- Loss function parameters
- Training hyperparameters
- Two-stage training settings

### Gating Configuration (`gating.yaml`)

- Network architecture
- Feature extraction settings
- PAC-Bayes parameters
- Training settings

### Plugin Configuration (`plugin.yaml`)

- Rejection cost
- Fixed-point iteration parameters
- Worst-group extension settings
- Evaluation metrics

## Key Methods

### 1. Balanced Linear Risk

The method optimizes a linearized version of the balanced selective risk:

```
R_α^lin(h,r) = Σ_k (1/α_k) Pr(y≠h(x), r(x)=0, y∈G_k) + c Pr(r(x)=1)
```

### 2. PAC-Bayes Bound

The gating network minimizes a PAC-Bayes bound:

```
E_θ~Q[R̂_α,S^lin] + L_α√(KL(Q||Π) + ln(2n/δ)) / (2(n-1))
```

### 3. Plug-in Optimal Rule (Theorem 1)

Classification and rejection decisions:

```
h_θ(x) = argmax_y p_Qθ,y(x)/α[y]
r_θ(x) = 1 ⟺ max_y p_Qθ,y(x)/α[y] < Σ_j (1/α[j] - μ[j])p_Qθ,j(x) - c
```

## Evaluation Metrics

- **BSE**: Balanced Selective Error
- **WGSE**: Worst-Group Selective Error
- **AURC**: Area Under Risk-Coverage curve
- **ECE@group**: Expected Calibration Error per group
- **Coverage@group**: Acceptance rates per group

## Datasets

Supported datasets:

- CIFAR-10-LT (IF=100, 200)
- CIFAR-100-LT (IF=100, 200)
- ImageNet-LT
- iNaturalist-LT

## Results

The method demonstrates improved performance on:

- Balanced selective error across head/tail groups
- Worst-group selective error
- Calibration quality per group
- Coverage fairness

## Citation

If you use this code, please cite:

```bibtex
@article{pbgse2024,
  title={PAC-Bayes Group-Selective Ensemble for Long-tail Classification with Abstention},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
