#!/usr/bin/env python3
"""Prefetch every dataset used by both original and planned scripts.
Run this once with network access before launching experiments.
"""
import os
from pathlib import Path
root=Path(os.environ.get('SLT_DATA','./data')).resolve(); root.mkdir(parents=True,exist_ok=True)
print('torchvision data root:',root)
from torchvision import datasets as tv
for D in [tv.MNIST,tv.CIFAR10,tv.CIFAR100]:
    for train in [True,False]:
        d=D(str(root),train=train,download=True); print(D.__name__, 'train' if train else 'test', len(d))
print('\nPrefetching Hugging Face datasets used by unchanged legacy scripts...')
from datasets import load_dataset
for name in ['ylecun/mnist','uoft-cs/cifar10']:
    ds=load_dataset(name); print(name,{k:len(v) for k,v in ds.items()})
print('DONE')
