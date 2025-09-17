"""
Backbone network architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture for CIFAR"""
    
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Store feature dimension
        self.feature_dim = 512 * block.expansion
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = F.avg_pool2d(out, 4)
        features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_features(self, x):
        """Extract features without classification"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features = F.avg_pool2d(out, 4)
        features = features.view(features.size(0), -1)
        return features


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


class EMAModel:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def get_backbone(config: Dict[str, Any]) -> nn.Module:
    """Get backbone model based on config"""
    backbone_name = config['backbone']
    num_classes = config.get('num_classes', 10)
    
    if backbone_name == 'resnet18':
        return ResNet18(num_classes)
    elif backbone_name == 'resnet32':
        return ResNet32(num_classes)
    elif backbone_name == 'resnet34':
        return ResNet34(num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")


class BaseModel(nn.Module):
    """Base model wrapper with additional functionality"""
    
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_dim = backbone.feature_dim
        
    def forward(self, x, return_features=False):
        return self.backbone(x, return_features)
    
    def get_features(self, x):
        return self.backbone.get_features(x)
    
    def get_logits(self, x):
        return self.backbone(x)
    
    def get_probs(self, x, temperature=1.0):
        """Get softmax probabilities with temperature scaling"""
        logits = self.get_logits(x)
        return F.softmax(logits / temperature, dim=1)
