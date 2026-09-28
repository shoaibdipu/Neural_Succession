from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from .data import CIFAR10_TASKS, semantic_cifar100_tasks, load_dataset, task_subset, loader, synthetic_tasks
from .models import build_model
from .metrics import accuracy
from .train import train_task


class World:
    def __init__(self,dataset='cifar10',data_root='./data',arch='resnet18',device='cpu',
                 batch_size=128,workers=2,seed=0,smoke=False):
        self.dataset=dataset; self.data_root=data_root; self.arch=arch; self.device=torch.device(device)
        self.batch_size=batch_size; self.workers=workers; self.seed=seed; self.smoke=smoke
        if smoke:
            ds,tasks,sub=synthetic_tasks(seed=seed,n_per_class=64,n_classes=6)
            self.train_ds=ds; self.train_eval_ds=ds; self.test_ds=ds; self.tasks=tasks
            self._sub_train=lambda t:sub(t); self._sub_train_eval=lambda t:sub(t); self._sub_test=lambda t:sub(t)
            self.num_classes=6; self.arch='smallcnn'; self.workers=0
        else:
            self.train_ds=load_dataset(dataset,data_root,True,True,augment=True)
            # Same training examples/labels, deterministic transforms for validation/probes.
            self.train_eval_ds=load_dataset(dataset,data_root,True,True,augment=False)
            self.test_ds=load_dataset(dataset,data_root,False,True,augment=False)
            if dataset=='cifar10': self.tasks=CIFAR10_TASKS; self.num_classes=10
            elif dataset=='cifar100': self.tasks=semantic_cifar100_tasks(self.train_ds); self.num_classes=100
            else: raise ValueError(dataset)
            self._sub_train=lambda t:task_subset(self.train_ds,t)
            self._sub_train_eval=lambda t:task_subset(self.train_eval_ds,t)
            self._sub_test=lambda t:task_subset(self.test_ds,t)

    def model(self,pretrained=False):
        return build_model(self.arch,self.num_classes,pretrained).to(self.device)

    def _train_val_indices(self,task,val_fraction=0.1):
        ds=self._sub_train(task); ids=np.asarray(ds.indices,dtype=int); n=len(ids)
        nv=max(1,int(round(n*val_fraction))); g=torch.Generator().manual_seed(self.seed+1000+task.task_id)
        perm=torch.randperm(n,generator=g).numpy(); val_pos=perm[:nv]; tr_pos=perm[nv:]
        return ids[tr_pos].tolist(), ids[val_pos].tolist()

    def train_val_subsets(self,task,val_fraction=0.1):
        tr_ids,va_ids=self._train_val_indices(task,val_fraction)
        return Subset(self.train_ds,tr_ids), Subset(self.train_eval_ds,va_ids)

    def train_loader(self,task,shuffle=True,batch_size=None,seed=None):
        tr,_=self.train_val_subsets(task)
        return loader(tr,batch_size or self.batch_size,shuffle,self.workers,self.seed if seed is None else seed)

    def val_loader(self,task,batch_size=None):
        _,va=self.train_val_subsets(task)
        return loader(va,batch_size or self.batch_size,False,self.workers,self.seed)

    def probe_loader(self,task,batch_size=None,seed=None):
        """Deterministic transform on the incoming training split; never test data."""
        return loader(self._sub_train_eval(task),batch_size or self.batch_size,False,self.workers,
                      self.seed if seed is None else seed)

    def full_train_loader(self,task,shuffle=False,batch_size=None,seed=None):
        # Training-transform loader used by optimization/buffer population.
        return loader(self._sub_train(task),batch_size or self.batch_size,shuffle,self.workers,
                      self.seed if seed is None else seed)

    def test_loader(self,task,batch_size=None):
        return loader(self._sub_test(task),batch_size or self.batch_size,False,self.workers,self.seed)

    def train_task(self,model,task,epochs=10,lr=0.05,target_acc=None,max_epochs=None,
                   target_tolerance=None,restore_closest=False,weight_decay=5e-4,momentum=0.9):
        return train_task(model,self.train_loader(task,True),self.device,epochs=epochs,lr=lr,
                          weight_decay=weight_decay,momentum=momentum,
                          task_classes=task.classes,val_dl=self.val_loader(task),target_acc=target_acc,
                          max_epochs=max_epochs,target_tolerance=target_tolerance,
                          restore_closest=restore_closest)

    def acc(self,model,task,task_il=True,allowed_classes=None):
        """Evaluate a task with an explicit class-IL seen-class mask when requested.

        task-IL masks to the task's own classes.  class-IL should pass the union
        of classes observed up to the current training step via ``allowed_classes``.
        This matches the frozen manuscript protocol and prevents future, untrained
        logits from contaminating intermediate class-IL measurements.
        """
        mask_classes = task.classes if task_il else allowed_classes
        return accuracy(model,self.test_loader(task),self.device,mask_classes)

    def val_acc(self,model,task,task_il=True,allowed_classes=None):
        mask_classes = task.classes if task_il else allowed_classes
        return accuracy(model,self.val_loader(task),self.device,mask_classes)
