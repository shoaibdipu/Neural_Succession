#!/usr/bin/env python3
"""Prefetch optional torchvision pretrained backbones with network access."""
from torchvision.models import resnet50,ResNet50_Weights,vit_b_16,ViT_B_16_Weights
print('prefetch ResNet50...'); resnet50(weights=ResNet50_Weights.DEFAULT)
print('prefetch ViT-B/16...'); vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
print('PRETRAINED_CACHE_READY')
