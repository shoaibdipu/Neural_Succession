from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18

class SmallCNN(nn.Module):
    def __init__(self, num_classes:int=10, feat_dim:int=64):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Linear(64,feat_dim)
        self.fc=nn.Linear(feat_dim,num_classes)
    def forward_features(self,x):
        z=self.features(x).flatten(1); return self.proj(z)
    def forward(self,x): return self.fc(self.forward_features(x))

class ResNet18CL(nn.Module):
    def __init__(self,num_classes:int=10, pretrained:bool=False):
        super().__init__()
        # weights intentionally not downloaded by default; set pretrained only when requested.
        from torchvision.models import ResNet18_Weights
        m=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        d=m.fc.in_features; m.fc=nn.Identity()
        self.backbone=m; self.fc=nn.Linear(d,num_classes)
    def forward_features(self,x): return self.backbone(x)
    def forward(self,x): return self.fc(self.forward_features(x))


def build_model(arch:str,num_classes:int,pretrained:bool=False):
    if arch=="smallcnn": return SmallCNN(num_classes)
    if arch=="resnet18": return ResNet18CL(num_classes,pretrained)
    raise ValueError(arch)
