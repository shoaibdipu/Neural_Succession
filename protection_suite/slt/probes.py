from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _dataset_targets(ds):
    """Return labels for a Dataset/Subset without running image transforms."""
    if isinstance(ds, Subset):
        base = _dataset_targets(ds.dataset)
        return np.asarray(base)[np.asarray(ds.indices, dtype=int)]
    if isinstance(ds, TensorDataset):
        if len(ds.tensors) < 2:
            raise ValueError("TensorDataset has no label tensor")
        return ds.tensors[1].detach().cpu().numpy()
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)
    raise TypeError(f"Cannot recover labels from dataset type {type(ds).__name__}")


@torch.no_grad()
def extract_features(model, dl, device, max_n:int|None=None):
    model.eval(); xs=[]; ys=[]; n=0
    for x,y in dl:
        f=model.forward_features(x.to(device)).detach().cpu().numpy()
        xs.append(f); ys.append(y.numpy()); n += len(y)
        if max_n is not None and n>=max_n: break
    if not xs:
        return np.empty((0,0)), np.empty((0,), dtype=int)
    X=np.concatenate(xs); Y=np.concatenate(ys)
    if max_n is not None:
        X=X[:max_n]; Y=Y[:max_n]
    return X,Y


@torch.no_grad()
def extract_class_balanced_features(model, dl, device, n_total:int, seed:int=0):
    """Extract at most ``n_total`` frozen features with an exact class quota.

    Class labels are read from the underlying dataset metadata, so feature
    extraction stops once the requested quota is filled rather than forwarding
    the entire incoming task. This keeps P1/P2/P6 CPU/GPU cost bounded while
    preserving the manuscript's class-balanced probe protocol.
    """
    targets=_dataset_targets(dl.dataset)
    classes=np.unique(targets)
    if len(classes)==0:
        raise ValueError("empty incoming dataset")
    per=n_total//len(classes); rem=n_total-per*len(classes)
    quota={int(c): per+(1 if j<rem else 0) for j,c in enumerate(classes)}
    rng=np.random.default_rng(seed)
    # Randomize which class receives the remainder without changing quotas.
    if rem:
        shuffled=classes.copy(); rng.shuffle(shuffled)
        quota={int(c):per for c in classes}
        for c in shuffled[:rem]: quota[int(c)]+=1
    xs=[]; ys=[]; have={int(c):0 for c in classes}
    model.eval()
    for x,y in dl:
        take=[]
        for i,yi in enumerate(y.tolist()):
            yi=int(yi)
            if yi in quota and have[yi] < quota[yi]:
                take.append(i); have[yi]+=1
        if take:
            idx=torch.as_tensor(take,dtype=torch.long)
            f=model.forward_features(x.index_select(0,idx).to(device)).detach().cpu().numpy()
            xs.append(f); ys.append(y.index_select(0,idx).numpy())
        if all(have[c] >= quota[c] for c in quota):
            break
    if not xs:
        raise RuntimeError("failed to extract probe features")
    X=np.concatenate(xs); Y=np.concatenate(ys)
    # If a tiny/smoke dataset cannot fill the nominal quota, keep all available.
    order=np.arange(len(Y)); rng.shuffle(order)
    return X[order],Y[order]


def linear_probe_error(X:np.ndarray,y:np.ndarray,seed:int=0,folds:int=5,C:float=1.0)->float:
    clf=LogisticRegression(C=C,max_iter=1000,solver="lbfgs")
    counts=np.unique(y,return_counts=True)[1]
    n_splits=max(2,min(folds,int(counts.min())))
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    acc=cross_val_score(clf,X,y,cv=cv,scoring="accuracy").mean()
    return float(1.0-acc)


def slt_compatibility_from_features(X:np.ndarray,y:np.ndarray,n_shuffles:int=20,seed:int=0):
    """rho = 1 - sqrt(e_B/e_B,sf), matching the manuscript definition.

    e_B is cross-validated linear-probe error on frozen resident features.
    e_B,sf is the mean error after label permutation. All labels come from the
    incoming training subset; benchmark test labels are never used.
    """
    e=linear_probe_error(X,y,seed=seed)
    rng=np.random.default_rng(seed+17); es=[]
    for s in range(n_shuffles):
        yp=rng.permutation(y)
        es.append(linear_probe_error(X,yp,seed=seed+100+s))
    esf=float(np.mean(es))
    rho=1.0-np.sqrt(max(e,1e-8)/max(esf,1e-8))
    return {"rho":float(rho),"omega":float(1-rho),"probe_error":e,
            "shuffle_error":esf,"shuffle_sd":float(np.std(es,ddof=1) if len(es)>1 else 0),
            "probe_n_used":int(len(y))}
