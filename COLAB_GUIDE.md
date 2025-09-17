# PB-GSE Google Colab Guide

Hướng dẫn chạy PB-GSE trên Google Colab một cách đầy đủ.

## 🚀 Quick Start

### 1. Setup trong Colab

```python
# Clone repository
!git clone https://github.com/your-repo/PB-GSE.git
%cd PB-GSE

# Setup environment
!python setup_colab.py

# Test installation
!python pb_gse/scripts/colab_demo.py
```

### 2. Chạy Demo nhanh

```python
# Chạy demo với synthetic data
!python pb_gse/scripts/colab_demo.py
```

### 3. Chạy thí nghiệm đầy đủ

```python
# Chạy experiment pipeline hoàn chỉnh
!python pb_gse/scripts/run_experiment.py --config pb_gse/configs/experiment.yaml --save_dir ./colab_outputs
```

## 📋 Chi tiết từng bước

### Bước 1: Cài đặt và Setup

```python
import os
import sys

# Clone repo nếu chưa có
if not os.path.exists('PB-GSE'):
    !git clone https://github.com/your-repo/PB-GSE.git

%cd PB-GSE

# Install requirements
!pip install torch torchvision numpy matplotlib scikit-learn PyYAML tqdm

# Setup directories
!mkdir -p data outputs outputs/logs outputs/models outputs/probs

# Add to Python path
sys.path.append('/content/PB-GSE')
```

### Bước 2: Test Import và Dataset

```python
# Test imports
from pb_gse.data.datasets import CIFAR10LT
from pb_gse.models.backbones import ResNet32
from pb_gse.models.gating import PACBayesGating
from pb_gse.models.plugin_rule import PluginOptimizer
from pb_gse.models.metrics import SelectiveMetrics

print("✓ All imports successful!")

# Test dataset
import torch
from pb_gse.data.transforms import get_transforms

config = {
    'data': {
        'name': 'cifar10_lt',
        'augmentation': {
            'train': {'rand_augment': {'n': 2, 'm': 10}},
            'test': {'center_crop': True}
        }
    }
}

transform_train, transform_test = get_transforms(config)
lt_dataset = CIFAR10LT(root='./data', imbalance_factor=100, seed=42)
train_dataset, test_dataset, group_info = lt_dataset.get_datasets(transform_train, transform_test)

print(f"✓ Dataset: {len(train_dataset)} train, {len(test_dataset)} test")
print(f"✓ Groups: {group_info}")
```

### Bước 3: Chạy Pipeline đơn giản

```python
# Chạy demo pipeline
!python pb_gse/scripts/colab_demo.py
```

### Bước 4: Training Base Models (tuỳ chọn)

```python
# Train từng model riêng lẻ với epochs ngắn cho demo
configs = ['base_crt.yaml', 'base_ldam.yaml', 'base_cbfocal.yaml']

for config_file in configs:
    print(f"Training with {config_file}...")
    !python pb_gse/scripts/train_base.py \
        --config pb_gse/configs/experiment.yaml \
        --model_config pb_gse/configs/{config_file} \
        --save_dir ./colab_outputs \
        --device cuda
```

### Bước 5: Full Pipeline

```python
# Chạy toàn bộ pipeline
!python pb_gse/scripts/run_experiment.py \
    --config pb_gse/configs/experiment.yaml \
    --save_dir ./colab_outputs \
    --device cuda
```

## ⚙️ Cấu hình cho Colab

### Điều chỉnh config cho Colab

```python
import yaml

# Load và modify config cho Colab
with open('pb_gse/configs/experiment.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Giảm epochs cho demo nhanh
config['data']['batch_size'] = 128  # Giảm batch size
config['data']['num_workers'] = 2   # Giảm workers
config['data']['pin_memory'] = False # Tắt pin_memory

# Sửa base model epochs
base_configs = ['pb_gse/configs/base_crt.yaml',
                'pb_gse/configs/base_ldam.yaml',
                'pb_gse/configs/base_cbfocal.yaml']

for config_path in base_configs:
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Giảm epochs cho demo
    if 'stage1' in base_config['base_model']:
        base_config['base_model']['stage1']['epochs'] = 10
    if 'stage2' in base_config['base_model']:
        base_config['base_model']['stage2']['epochs'] = 5
    else:
        base_config['base_model']['epochs'] = 10

    with open(config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False)

print("✓ Configs updated for Colab")
```

## 📊 Xem kết quả

```python
import json
import matplotlib.pyplot as plt

# Load results
results_path = 'colab_outputs/results/metrics.json'
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    print("=== PB-GSE Results ===")
    print(f"Coverage: {results['coverage']:.3f}")
    print(f"Balanced Selective Error: {results['balanced_selective_error']:.3f}")
    print(f"Worst-Group Error: {results['worst_group_selective_error']:.3f}")
    print(f"AURC: {results['aurc']:.3f}")

    # Plot results
    metrics = ['coverage', 'balanced_selective_error', 'worst_group_selective_error', 'aurc']
    values = [results[m] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values)
    plt.title('PB-GSE Results')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **Out of Memory:**

   ```python
   # Giảm batch size
   config['data']['batch_size'] = 64
   config['gating']['batch_size'] = 256
   ```

2. **Dataset download chậm:**

   ```python
   # CIFAR-10 sẽ tự động download, chờ khoảng 5-10 phút
   ```

3. **Import errors:**
   ```python
   # Ensure Python path
   import sys
   sys.path.append('/content/PB-GSE')
   ```

### Performance Tips:

1. **Sử dụng GPU:**

   ```python
   # Kiểm tra GPU
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
   ```

2. **Tăng tốc training:**

   ```python
   # Sử dụng mixed precision
   config['use_amp'] = True

   # Giảm epochs cho demo
   config['demo_mode'] = True
   ```

## 📝 Lưu kết quả

```python
# Download results
from google.colab import files

# Zip results
!zip -r colab_results.zip colab_outputs/

# Download
files.download('colab_results.zip')
```

## 🎯 Next Steps

Sau khi chạy thành công trên Colab:

1. **Điều chỉnh hyperparameters** trong config files
2. **Thử các datasets khác** (CIFAR-100-LT)
3. **Chạy ablation studies**
4. **So sánh với baselines**

```python
# Chạy ablation study
!python pb_gse/scripts/run_ablation.py \
    --base_config pb_gse/configs/experiment.yaml \
    --output_dir ./ablation_outputs \
    --only_generate

# Chạy baseline comparison
!python pb_gse/scripts/evaluate_baselines.py \
    --config pb_gse/configs/experiment.yaml \
    --probs_dir colab_outputs/probs_calibrated \
    --output_dir ./baseline_results
```

## 📚 Resources

- **Paper**: [Link to paper when published]
- **GitHub**: https://github.com/your-repo/PB-GSE
- **Documentation**: See README.md
- **Issues**: Report on GitHub Issues

---

**🎉 Chúc bạn thành công với PB-GSE trên Google Colab!**
