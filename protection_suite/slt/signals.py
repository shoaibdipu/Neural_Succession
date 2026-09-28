from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .probes import extract_features, extract_class_balanced_features, slt_compatibility_from_features


def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def batch_grad(model,x,y,device,task_classes=None):
    model.zero_grad(set_to_none=True); model.train(False)
    x=x.to(device); y=y.to(device); z=model(x)
    if task_classes is not None:
        mask=torch.full((z.shape[1],),False,device=device,dtype=torch.bool); mask[list(task_classes)]=True
        z=z.masked_fill(~mask[None,:],-1e9)
    loss=F.cross_entropy(z,y)
    ps=[p for p in model.parameters() if p.requires_grad]
    gs=torch.autograd.grad(loss,ps,retain_graph=False,create_graph=False,allow_unused=True)
    return flatten_grads([g for g in gs if g is not None]).detach(), float(loss.item())


def gradient_cosine(g1,g2):
    den=(g1.norm()*g2.norm()).clamp_min(1e-12)
    return float(torch.dot(g1,g2)/den)


def linear_cka(X,Y):
    X=X-X.mean(0,keepdims=True); Y=Y-Y.mean(0,keepdims=True)
    hs=np.linalg.norm(X.T@Y,'fro')**2
    den=np.linalg.norm(X.T@X,'fro')*np.linalg.norm(Y.T@Y,'fro')
    return float(hs/max(den,1e-12))


def first_order_safety_floor(gR,gB):
    """Special-case q* for update (1-q)gB + q gR under plain SGD."""
    c=torch.clamp(-torch.dot(gR,gB),min=0.0)
    den=gR.pow(2).sum()+c+1e-12
    return float((c/den).clamp(0,1).item())


def slt_compatibility(model,incoming_loader,device,n_shuffles=20,max_n=256,seed=0):
    # Incoming labels are from the training split only. Feature extraction stops
    # once the class-balanced probe quota is filled.
    X,y=extract_class_balanced_features(model,incoming_loader,device,n_total=max_n,seed=seed)
    return slt_compatibility_from_features(X,y,n_shuffles=n_shuffles,seed=seed)


def feature_displacement(model_before,model_after,dl,device,max_n=512):
    X,_=extract_features(model_before,dl,device,max_n=max_n)
    Y,_=extract_features(model_after,dl,device,max_n=max_n)
    n=min(len(X),len(Y)); X=X[:n]; Y=Y[:n]
    return float(np.sqrt(np.mean(np.sum((Y-X)**2,axis=1))))


def covariance_similarity(X,Y):
    X=X-X.mean(0,keepdims=True); Y=Y-Y.mean(0,keepdims=True)
    CX=(X.T@X)/max(len(X)-1,1); CY=(Y.T@Y)/max(len(Y)-1,1)
    a=CX.reshape(-1); b=CY.reshape(-1)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def one_step_loss_change(model, old_batch, new_batch, device, lr=1e-3, old_classes=None, new_classes=None):
    """Transparent one-step loss-change baseline; not labeled as official SPOT."""
    from copy import deepcopy
    m=deepcopy(model).to(device); m.train(False)
    xa,ya=old_batch[0].to(device),old_batch[1].to(device)
    xb,yb=new_batch[0].to(device),new_batch[1].to(device)
    za=m(xa); base=F.cross_entropy(za,ya)
    zb=m(xb); lb=F.cross_entropy(zb,yb)
    ps=[p for p in m.parameters() if p.requires_grad]
    gs=torch.autograd.grad(lb,ps)
    with torch.no_grad():
        for p,g in zip(ps,gs): p.add_(g,alpha=-lr)
        after=F.cross_entropy(m(xa),ya)
    return float((after-base).item()), float(base.item()), float(after.item())
