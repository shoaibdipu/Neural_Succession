#!/usr/bin/env python3
"""Continual-learning baseline helpers for the confirmatory E5 benchmark.

The goal is transparent, auditable implementations with identical train/validation
handling across methods.  PackNet is task-IL only because its task-specific mask
requires task identity at inference.  ``ewc_dr`` is explicitly implemented as an
online/normalized-Fisher EWC operationalization; the exact choice is written into
result metadata rather than hidden behind the literature label.
"""
from __future__ import annotations
import copy, os, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from slt_common import DEVICE, _reshape_for, _set_trainable, compute_fisher, ewc_penalty


def _head(model,t): return model.heads[t] if model.scenario=='task' else model.head


def make_opt(params, *, optimizer='adam', lr=1e-3, wd=1e-4, momentum=0.0):
    """Construct an optimizer explicitly; class-IL callers use the frozen SGD regime.

    Keeping optimizer choice explicit prevents the previous class-IL failure mode in
    which every baseline silently inherited Adam from the task-IL implementation.
    """
    params=list(params)
    name=str(optimizer).lower()
    if name=='adam':
        return torch.optim.Adam(params,lr=lr,weight_decay=wd)
    if name in ('sgd','plain_sgd'):
        return torch.optim.SGD(params,lr=lr,momentum=0.0,weight_decay=wd)
    if name in ('sgd_momentum','momentum'):
        return torch.optim.SGD(params,lr=lr,momentum=float(momentum or 0.9),weight_decay=wd)
    raise ValueError(f'unknown optimizer {optimizer}')


def train_plain(model,t,X,y,*,arch,dataset,epochs,lr,batch,wd,global_labels=False,class_offset=0,extra=None,optimizer='adam',momentum=0.0):
    _set_trainable(model,t); Xr=_reshape_for(arch,dataset,X); yy=y+class_offset if global_labels else y
    dl=DataLoader(TensorDataset(torch.tensor(Xr),torch.tensor(yy)),batch_size=batch,shuffle=True,drop_last=(len(Xr)%batch==1))
    opt=make_opt([p for p in model.parameters() if p.requires_grad],optimizer=optimizer,lr=lr,wd=wd,momentum=momentum)
    model.train(); losses=[]
    for _ in range(epochs):
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); feat=model.features(xb); logits=_head(model,t)(feat)
            loss=F.cross_entropy(logits,yb)
            if extra is not None: loss=loss+extra(model,feat,logits,xb,yb)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
    return {'loss_mean':float(np.mean(losses)) if losses else None,'steps':len(losses)}


def fisher_normalize(fisher):
    vals=torch.cat([v.detach().reshape(-1) for v in fisher.values()])
    scale=vals.mean().clamp_min(1e-12)
    return {k:v/scale for k,v in fisher.items()}


def online_ewc_update(prev_fisher,new_fisher,gamma=1.0):
    nf=fisher_normalize(new_fisher)
    if prev_fisher is None: return {k:v.clone() for k,v in nf.items()}
    out={}
    for k,v in nf.items(): out[k]=gamma*prev_fisher.get(k,torch.zeros_like(v))+v
    return out


def train_agem(model,t,X,y,mem,*,arch,dataset,epochs,lr,batch,wd,global_labels=False,class_offset=0,seed=0,optimizer='adam',momentum=0.0):
    _set_trainable(model,t); Xr=_reshape_for(arch,dataset,X); yy=y+class_offset if global_labels else y
    dl=DataLoader(TensorDataset(torch.tensor(Xr),torch.tensor(yy)),batch_size=batch,shuffle=True,drop_last=(len(Xr)%batch==1))
    opt=make_opt([p for p in model.parameters() if p.requires_grad],optimizer=optimizer,lr=lr,wd=wd,momentum=momentum)
    params=[p for p in model.parameters() if p.requires_grad]; rng=np.random.RandomState(seed); model.train()
    for _ in range(epochs):
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(); F.cross_entropy(_head(model,t)(model.features(xb)),yb).backward()
            gcur=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            if mem['X'] is not None and len(mem['X']):
                opt.zero_grad(); idx=rng.choice(len(mem['X']),min(batch,len(mem['X'])),replace=False)
                xm=torch.tensor(_reshape_for(arch,dataset,mem['X'][idx]),dtype=torch.float32,device=DEVICE)
                ym=torch.tensor(mem['y'][idx],dtype=torch.long,device=DEVICE); tm=mem['t'][idx]
                ml=torch.tensor(0.,device=DEVICE)
                for tt in np.unique(tm):
                    mask=tm==tt
                    if mask.sum(): ml=ml+F.cross_entropy(_head(model,int(tt))(model.features(xm[mask])),ym[mask])
                ml.backward(); gref=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                dot=sum((a*b).sum() for a,b in zip(gcur,gref)); rn=sum((b*b).sum() for b in gref)+1e-12
                if dot<0: gcur=[a-(dot/rn)*b for a,b in zip(gcur,gref)]
            opt.zero_grad()
            for p,g in zip(params,gcur): p.grad=g
            opt.step()


def train_derpp(model,t,X,y,buf,*,arch,dataset,epochs,lr,batch,wd,global_labels=False,class_offset=0,
                alpha=.5,beta=.5,seed=0,extra=None,optimizer='adam',momentum=0.0):
    _set_trainable(model,t); Xr=_reshape_for(arch,dataset,X); yy=y+class_offset if global_labels else y
    dl=DataLoader(TensorDataset(torch.tensor(Xr),torch.tensor(yy)),batch_size=batch,shuffle=True,drop_last=(len(Xr)%batch==1))
    opt=make_opt([p for p in model.parameters() if p.requires_grad],optimizer=optimizer,lr=lr,wd=wd,momentum=momentum)
    rng=np.random.RandomState(seed); model.train(); has=buf['X'] is not None and len(buf['X'])
    for _ in range(epochs):
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); feat=model.features(xb); loss=F.cross_entropy(_head(model,t)(feat),yb)
            if has:
                idx=rng.choice(len(buf['X']),min(batch,len(buf['X'])),replace=True)
                xr=torch.tensor(_reshape_for(arch,dataset,buf['X'][idx]),dtype=torch.float32,device=DEVICE)
                yr=torch.tensor(buf['y'][idx],dtype=torch.long,device=DEVICE); lg=torch.tensor(buf['logits'][idx],dtype=torch.float32,device=DEVICE); tr=buf['t'][idx]
                for tt in np.unique(tr):
                    mask=tr==tt
                    if mask.sum():
                        out=_head(model,int(tt))(model.features(xr[mask]))
                        loss=loss+alpha*F.mse_loss(out,lg[mask])+beta*F.cross_entropy(out,yr[mask])
            if extra is not None: loss=loss+extra(model,feat,None,xb,yb)
            opt.zero_grad(); loss.backward(); opt.step()


def backbone_named_params(model):
    return [(n,p) for n,p in model.named_parameters() if 'head' not in n]


def _flat_grads(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for _,p in params])


def _write_flat_grads(params,g):
    off=0
    for _,p in params:
        n=p.numel(); p.grad=g[off:off+n].view_as(p).clone(); off+=n


def project_gradient(g,basis):
    if not basis: return g
    out=g
    for b in basis: out=out-torch.dot(out,b)*b
    return out


def build_ogd_basis(model,t,X,y,basis,*,arch,dataset,n_vectors=10,batch=32,global_labels=False,class_offset=0,seed=0):
    rng=np.random.RandomState(seed); bp=backbone_named_params(model); model.eval()
    for _ in range(n_vectors):
        idx=rng.choice(len(X),min(batch,len(X)),replace=False)
        xb=torch.tensor(_reshape_for(arch,dataset,X[idx]),dtype=torch.float32,device=DEVICE)
        yy=y[idx]+class_offset if global_labels else y[idx]; yb=torch.tensor(yy,dtype=torch.long,device=DEVICE)
        model.zero_grad(); F.cross_entropy(_head(model,t)(model.features(xb)),yb).backward(); g=_flat_grads(bp).detach()
        g=project_gradient(g,basis); n=g.norm()
        if n>1e-8: basis.append((g/n).detach().clone())
    return basis


def train_ogd(model,t,X,y,basis,*,arch,dataset,epochs,lr,batch,wd,global_labels=False,class_offset=0,optimizer='adam',momentum=0.0):
    _set_trainable(model,t); Xr=_reshape_for(arch,dataset,X); yy=y+class_offset if global_labels else y
    dl=DataLoader(TensorDataset(torch.tensor(Xr),torch.tensor(yy)),batch_size=batch,shuffle=True,drop_last=(len(Xr)%batch==1))
    opt=make_opt([p for p in model.parameters() if p.requires_grad],optimizer=optimizer,lr=lr,wd=wd,momentum=momentum); bp=backbone_named_params(model); model.train()
    for _ in range(epochs):
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(); F.cross_entropy(_head(model,t)(model.features(xb)),yb).backward()
            if basis:
                g=_flat_grads(bp); _write_flat_grads(bp,project_gradient(g,basis))
            opt.step()


class PackNetState:
    """Minimal task-IL PackNet-style magnitude allocation over backbone weights."""
    def __init__(self,model):
        self.assignment={n:torch.zeros_like(p,dtype=torch.int16,device=p.device) for n,p in backbone_named_params(model)}

    def mask_frozen_grads(self,model,current_task):
        for n,p in backbone_named_params(model):
            if p.grad is not None: p.grad[self.assignment[n]>0]=0

    def allocate(self,model,task_id,keep_fraction=.5):
        label=task_id+1
        for n,p in backbone_named_params(model):
            free=self.assignment[n]==0
            vals=p.detach().abs()[free]
            if vals.numel()==0: continue
            k=max(1,int(round(keep_fraction*vals.numel())))
            if k>=vals.numel(): chosen=free
            else:
                thr=torch.topk(vals,k,largest=True).values.min(); chosen=free & (p.detach().abs()>=thr)
            self.assignment[n][chosen]=label
            # Unallocated free weights are zeroed for the next task.
            p.data[free & ~chosen]=0

    def apply_eval_mask(self,model,task_id):
        backup={}
        for n,p in backbone_named_params(model):
            backup[n]=p.data.clone(); allowed=(self.assignment[n]>0)&(self.assignment[n]<=task_id+1); p.data.mul_(allowed)
        return backup

    def restore(self,model,backup):
        for n,p in backbone_named_params(model): p.data.copy_(backup[n])


def train_packnet(model,state,t,X,y,*,arch,dataset,epochs,lr,batch,wd,keep_fraction=.5):
    _set_trainable(model,t); Xr=_reshape_for(arch,dataset,X); dl=DataLoader(TensorDataset(torch.tensor(Xr),torch.tensor(y)),batch_size=batch,shuffle=True,drop_last=(len(Xr)%batch==1))  # BatchNorm rejects a size-1 tail in train mode
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=wd); model.train()
    for _ in range(epochs):
        for xb,yb in dl:
            if xb.size(0) < 2: continue   # BatchNorm needs >1 sample in train mode
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(); F.cross_entropy(model.heads[t](model.features(xb)),yb).backward(); state.mask_frozen_grads(model,t); opt.step()
    state.allocate(model,t,keep_fraction=keep_fraction)
