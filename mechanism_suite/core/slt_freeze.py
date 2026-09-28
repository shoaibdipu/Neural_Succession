#!/usr/bin/env python3
"""Frozen-protocol helpers for the 2026-09-17 EST/SLT campaign.

This module contains *shared infrastructure only*.  No experiment reads another
experiment's result file.  Every experiment constructs its own datasets, models,
probes, and statistics so experiments can be launched independently.
"""
from __future__ import annotations

import copy, json, math, os, random, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

from slt_common import (DEVICE, _reshape_for, _set_trainable, accuracy, feature_matrix,
                        make_model, set_seed, subsample)
from slt_confirmatory import (freeze_batchnorm_stats, json_dump_safe, run_metadata,
                              stratified_probe, stratified_train_val)


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, '1' if default else '0').lower() in {'1','true','yes','y'}


def exp_outdir(root: Path, exp: str) -> Path:
    base = Path(os.environ.get('SLT_RESULTS_ROOT', str(root / 'results')))
    out = base / exp
    out.mkdir(parents=True, exist_ok=True)
    return out


def dump_result(root: Path, exp: str, obj: dict, filename: Optional[str] = None) -> Path:
    out = exp_outdir(root, exp)
    p = out / (filename or f'{exp}.json')
    json_dump_safe(obj, p)
    return p


def fixed_pairs(n_tasks: int, limit: Optional[int] = None) -> List[Tuple[int,int]]:
    """Balanced deterministic directed-pair order."""
    pairs=[]
    for d in range(1,n_tasks):
        for a in range(n_tasks):
            pairs.append((a,(a+d)%n_tasks))
    return pairs if limit is None else pairs[:int(limit)]


def pearson_safe(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12: return (None,None)
    r,p=stats.pearsonr(x[ok],y[ok]); return float(r),float(p)


def spearman_safe(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12: return (None,None)
    r,p=stats.spearmanr(x[ok],y[ok]); return float(r),float(p)


def auc_safe(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); ok=np.isfinite(s)
    if ok.sum()<2 or len(np.unique(y[ok]))<2: return None
    return float(roc_auc_score(y[ok],s[ok]))


def bootstrap_slope(rows: Sequence[dict], x_key: str, y_key: str,
                    task_keys=('task_a','task_b'), n_boot=5000, seed=42):
    """Task-cluster bootstrap: resample task identities, then keep pair rows whose
    endpoints are represented. Multiplicity is induced by sampled task counts.
    """
    xs=np.asarray([r[x_key] for r in rows],float); ys=np.asarray([r[y_key] for r in rows],float)
    ta=np.asarray([r[task_keys[0]] for r in rows],int); tb=np.asarray([r[task_keys[1]] for r in rows],int)
    ok=np.isfinite(xs)&np.isfinite(ys); xs,ys,ta,tb=xs[ok],ys[ok],ta[ok],tb[ok]
    tasks=np.unique(np.r_[ta,tb]); rng=np.random.RandomState(seed)
    if len(tasks)<2: return {'slope':None,'ci95':None,'n_boot':0}
    slope=float(np.polyfit(xs,ys,1)[0]) if np.std(xs)>1e-12 else None
    bs=[]
    for _ in range(int(n_boot)):
        draw=rng.choice(tasks,size=len(tasks),replace=True)
        cnt={int(t):int(np.sum(draw==t)) for t in tasks}
        w=np.asarray([cnt.get(int(a),0)*cnt.get(int(b),0) for a,b in zip(ta,tb)],float)
        sel=w>0
        if sel.sum()<3 or np.std(xs[sel])<1e-12: continue
        # weighted linear regression with intercept
        X=np.c_[np.ones(sel.sum()),xs[sel]]; W=np.sqrt(w[sel])[:,None]
        beta=np.linalg.lstsq(X*W,ys[sel]*W[:,0],rcond=None)[0]
        bs.append(float(beta[1]))
    ci=[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))] if bs else None
    return {'slope':slope,'ci95':ci,'n_boot':len(bs),'bootstrap':'task-cluster'}


def loto_incoming_signs(rows: Sequence[dict], x_key: str, y_key='forgetting', n_tasks=None):
    """Train a one-dimensional linear predictor excluding each incoming task and
    report the held-out correlation/sign. Used only as a stability diagnostic.
    """
    if n_tasks is None:
        n_tasks=1+max(max(r['task_a'],r['task_b']) for r in rows)
    folds=[]; pred=[]; true=[]
    for b in range(int(n_tasks)):
        tr=[r for r in rows if r['task_b']!=b]; te=[r for r in rows if r['task_b']==b]
        if len(tr)<3 or len(te)<1: continue
        x=np.asarray([r[x_key] for r in tr],float); y=np.asarray([r[y_key] for r in tr],float)
        ok=np.isfinite(x)&np.isfinite(y)
        if ok.sum()<3 or np.std(x[ok])<1e-12: continue
        m=LinearRegression().fit(x[ok,None],y[ok])
        p=m.predict(np.asarray([r[x_key] for r in te],float)[:,None])
        yte=np.asarray([r[y_key] for r in te],float)
        pred.extend(p.tolist()); true.extend(yte.tolist())
        rfold,_=pearson_safe(p,yte)
        folds.append({'incoming_task':b,'n':len(te),'r':rfold,'slope':float(m.coef_[0])})
    rall,_=pearson_safe(pred,true)
    # The expected direction for compatibility rho -> forgetting is negative.
    neg=sum(1 for f in folds if f['slope']<0)
    return {'folds':folds,'overall_r':rall,'n_negative_slope_folds':neg,'n_folds':len(folds)}


# ---------------------------------------------------------------------------
# CIFAR-100 superclass task definition (standard fine-class names)
# ---------------------------------------------------------------------------
CIFAR100_COARSE = {
'aquatic_mammals':['beaver','dolphin','otter','seal','whale'],
'fish':['aquarium_fish','flatfish','ray','shark','trout'],
'flowers':['orchids','poppies','roses','sunflowers','tulips'],
'food_containers':['bottles','bowls','cans','cups','plates'],
'fruit_and_vegetables':['apples','mushrooms','oranges','pears','sweet_peppers'],
'household_electrical_devices':['clock','computer_keyboard','lamp','telephone','television'],
'household_furniture':['bed','chair','couch','table','wardrobe'],
'insects':['bee','beetle','butterfly','caterpillar','cockroach'],
'large_carnivores':['bear','leopard','lion','tiger','wolf'],
'large_man_made_outdoor_things':['bridge','castle','house','road','skyscraper'],
'large_natural_outdoor_scenes':['cloud','forest','mountain','plain','sea'],
'large_omnivores_and_herbivores':['camel','cattle','chimpanzee','elephant','kangaroo'],
'medium_sized_mammals':['fox','porcupine','possum','raccoon','skunk'],
'non_insect_invertebrates':['crab','lobster','snail','spider','worm'],
'people':['baby','boy','girl','man','woman'],
'reptiles':['crocodile','dinosaur','lizard','snake','turtle'],
'small_mammals':['hamster','mouse','rabbit','shrew','squirrel'],
'trees':['maple_tree','oak_tree','palm_tree','pine_tree','willow_tree'],
'vehicles_1':['bicycle','bus','motorcycle','pickup_truck','train'],
'vehicles_2':['lawn_mower','rocket','streetcar','tank','tractor'],
}
# Frozen pairing of 20 superclasses into ten 10-way tasks. Pairing is based on
# semantic coherence and is not changed after outcomes are inspected.
CIFAR100_SEM_PAIRS = [
 ('aquatic_mammals','fish'),
 ('large_carnivores','medium_sized_mammals'),
 ('large_omnivores_and_herbivores','small_mammals'),
 ('insects','non_insect_invertebrates'),
 ('reptiles','people'),
 ('flowers','trees'),
 ('fruit_and_vegetables','food_containers'),
 ('household_electrical_devices','household_furniture'),
 ('large_man_made_outdoor_things','large_natural_outdoor_scenes'),
 ('vehicles_1','vehicles_2'),
]


# Torchvision's CIFAR-100 fine-label names are SINGULAR ('orchid','poppy','apple','keyboard', ...); the plural
# forms used in CIFAR100_COARSE above come from the dataset web page and would raise a name mismatch at run time.
# The frozen partition is therefore resolved from the dataset's own COARSE labels (pickle 'coarse_labels'), which
# is exact and name-independent.  CIFAR100_COARSE is kept only as documentation of the intended membership.
_COARSE_ALIASES={'large_man_made_outdoor_things':'large_man-made_outdoor_things','medium_sized_mammals':'medium_mammals',
                 'non_insect_invertebrates':'non-insect_invertebrates'}
CIFAR100_COARSE_NAMES=['aquatic_mammals','fish','flowers','food_containers','fruit_and_vegetables','household_electrical_devices',
 'household_furniture','insects','large_carnivores','large_man-made_outdoor_things','large_natural_outdoor_scenes',
 'large_omnivores_and_herbivores','medium_mammals','non-insect_invertebrates','people','reptiles','small_mammals','trees','vehicles_1','vehicles_2']


def cifar100_fine_to_coarse(root=None):
    """(fine->coarse id array, coarse names, fine names) from the torchvision CIFAR-100 pickle.
    SLT_SYNTHETIC=1 (smoke only) returns a deterministic 5-per-superclass mapping."""
    import pickle
    if os.environ.get('SLT_SYNTHETIC','0')=='1':
        return np.repeat(np.arange(20),5),list(CIFAR100_COARSE_NAMES),[f'fine{i}' for i in range(100)]
    root=root or os.environ.get('SLT_DATA','./data'); base=Path(root)/'cifar-100-python'
    if not (base/'train').exists():
        from torchvision import datasets; datasets.CIFAR100(root,train=True,download=True)
    d=pickle.load(open(base/'train','rb'),encoding='latin1'); meta=pickle.load(open(base/'meta','rb'),encoding='latin1')
    fine=np.asarray(d['fine_labels']); coarse=np.asarray(d['coarse_labels']); f2c=np.zeros(100,int)
    for c in range(100): f2c[c]=int(np.bincount(coarse[fine==c]).argmax())
    return f2c,list(meta['coarse_label_names']),list(meta['fine_label_names'])


def cifar100_semantic_splits(class_names=None, root=None):
    """Ten 10-way tasks from CIFAR100_SEM_PAIRS, resolved through the dataset's coarse labels."""
    f2c,cnames,_=cifar100_fine_to_coarse(root); cid={n:i for i,n in enumerate(cnames)}
    splits=[]
    for a,b in CIFAR100_SEM_PAIRS:
        ids=[cid[_COARSE_ALIASES.get(n,n)] for n in (a,b)]
        splits.append(sorted(int(c) for c in range(100) if f2c[c] in ids))
    flat=[i for s in splits for i in s]
    if len(flat)!=100 or len(set(flat))!=100:
        raise RuntimeError('semantic CIFAR-100 split must cover each fine class exactly once')
    return splits


def load_cifar100_semantic(root=None):
    """Return normalized numpy data plus the frozen semantic task splits (via slt_common.load_np)."""
    from slt_common import load_np
    Xtr,ytr,Xte,yte=load_np('cifar100',root); splits=cifar100_semantic_splits(root=root)
    _,_,fine_names=cifar100_fine_to_coarse(root)
    return Xtr,ytr,Xte,yte,splits,fine_names


# ---------------------------------------------------------------------------
# Layer features / Jacobians
# ---------------------------------------------------------------------------
def layer_features(model, xb: torch.Tensor, layer: str) -> torch.Tensor:
    if layer in {'feat','penultimate','features'}: return model.features(xb)
    d=model.features_at(xb)
    if layer not in d: raise KeyError(f'layer {layer!r} not in {list(d)}')
    z=d[layer]
    return z if z.ndim==2 else z.flatten(1)


def layer_feature_jacobians(model, X, *, arch, dataset, layer='feat', n_probe=64,
                            max_features=None, feature_indices=None, device=DEVICE):
    model.eval().to(device)
    Xs=np.asarray(X)[:min(int(n_probe),len(X))]
    xr=_reshape_for(arch,dataset,Xs)
    xb=torch.as_tensor(xr,dtype=torch.float32,device=device).requires_grad_(True)
    z=layer_features(model,xb,layer)
    if feature_indices is None:
        K=z.shape[1] if max_features is None else min(z.shape[1],int(max_features))
        ids=list(range(K))
    else:
        ids=[int(i) for i in feature_indices]
        if max_features is not None: ids=ids[:int(max_features)]
    rows=[]
    for j,i in enumerate(ids):
        g=torch.autograd.grad(z[:,i].sum(),xb,retain_graph=(j < len(ids)-1),create_graph=False)[0]
        rows.append(g.mean(0).detach().cpu().reshape(-1).numpy())
    return np.stack(rows,0)


def layer_strength(model, X, *, arch, dataset, layer='feat', max_n=2000, device=DEVICE):
    model.eval().to(device); Xs=np.asarray(X)[:min(len(X),int(max_n))]; xr=_reshape_for(arch,dataset,Xs)
    out=[]
    with torch.no_grad():
        for i in range(0,len(xr),256):
            xb=torch.as_tensor(xr[i:i+256],dtype=torch.float32,device=device)
            out.append(layer_features(model,xb,layer).cpu())
    z=torch.cat(out,0)
    return (z.mean(0)**2).numpy()


# ---------------------------------------------------------------------------
# Training helpers with explicit optimizer/update access
# ---------------------------------------------------------------------------
def train_task_optimizer(model, task_id, X, y, *, arch, dataset, optimizer='adam',
                         lr=1e-3, epochs=20, batch=128, wd=1e-4, seed=42,
                         freeze_bn=False, global_labels=False, class_offset=0,
                         step_hook=None):
    """Train a task with deterministic numpy shuffling. step_hook, if supplied,
    is called after backward and before optimizer.step as hook(ctx), and after
    the update as hook(ctx | {'phase':'after','delta':...}).
    """
    _set_trainable(model,task_id); params=[p for p in model.parameters() if p.requires_grad]
    if optimizer.lower()=='sgd': opt=torch.optim.SGD(params,lr=lr,momentum=0.0,weight_decay=wd)
    elif optimizer.lower()=='sgd_momentum': opt=torch.optim.SGD(params,lr=lr,momentum=.9,weight_decay=wd)
    elif optimizer.lower()=='adam': opt=torch.optim.Adam(params,lr=lr,weight_decay=wd)
    else: raise ValueError(optimizer)
    xr=_reshape_for(arch,dataset,X); yy=np.asarray(y)+(class_offset if global_labels else 0); rng=np.random.RandomState(seed)
    step=0; losses=[]
    for ep in range(int(epochs)):
        perm=rng.permutation(len(xr))
        for i in range(0,len(xr),int(batch)):
            ids=perm[i:i+int(batch)]; xb=torch.as_tensor(xr[ids],dtype=torch.float32,device=DEVICE); yb=torch.as_tensor(yy[ids],dtype=torch.long,device=DEVICE)
            model.train();
            if freeze_bn: freeze_batchnorm_stats(model)
            opt.zero_grad(set_to_none=True); f=model.features(xb); logits=model.heads[task_id](f) if model.scenario=='task' else model.head(f); loss=F.cross_entropy(logits,yb); loss.backward()
            before=None
            if step_hook is not None:
                before=[p.detach().clone() for p in params]; step_hook({'phase':'before','step':step,'epoch':ep,'model':model,'xb':xb,'yb':yb,'loss':loss,'params':params})
            opt.step()
            if step_hook is not None:
                delta=[p.detach()-q for p,q in zip(params,before)]; step_hook({'phase':'after','step':step,'epoch':ep,'model':model,'xb':xb,'yb':yb,'loss':loss.detach(),'params':params,'delta':delta})
            losses.append(float(loss.detach().cpu())); step+=1
    return {'steps':step,'loss_mean':float(np.mean(losses)),'loss_last':losses[-1] if losses else None,'optimizer':optimizer,'lr':lr,'epochs':epochs}


def backbone_named_parameters(model):
    return [(n,p) for n,p in model.named_parameters() if ('head' not in n)]


def flat_grad(loss, named_params, *, retain_graph=False, create_graph=False):
    params=[p for _,p in named_params]
    gs=torch.autograd.grad(loss,params,retain_graph=retain_graph,create_graph=create_graph,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])


def task_loss(model, task_id, X, y, *, arch, dataset, max_n=None, seed=0, freeze_bn=False,
              global_labels=False, class_offset=0):
    Xs,ys=subsample(np.asarray(X),np.asarray(y),max_n,seed=seed); xr=_reshape_for(arch,dataset,Xs)
    xb=torch.as_tensor(xr,dtype=torch.float32,device=DEVICE); yy=torch.as_tensor(ys+(class_offset if global_labels else 0),dtype=torch.long,device=DEVICE)
    model.train()
    if freeze_bn: freeze_batchnorm_stats(model)
    f=model.features(xb); logits=model.heads[task_id](f) if model.scenario=='task' else model.head(f)
    return F.cross_entropy(logits,yy), xb, yy


def backbone_gradient(model, task_id, X, y, *, arch, dataset, max_n=128, seed=0, freeze_bn=True,
                      global_labels=False, class_offset=0):
    model.zero_grad(set_to_none=True)
    loss,_,_=task_loss(model,task_id,X,y,arch=arch,dataset=dataset,max_n=max_n,seed=seed,freeze_bn=freeze_bn,
                       global_labels=global_labels,class_offset=class_offset)
    named=backbone_named_parameters(model); g=flat_grad(loss,named)
    return g.detach(),float(loss.detach().cpu())


def linear_probe_loss_and_acc(model, X_fit, y_fit, X_eval, y_eval, *, arch, dataset,
                              max_iter=1000, C=1.0):
    """Fit a multinomial logistic probe on frozen features and evaluate on a
    disjoint set. Returns eval cross entropy and accuracy.
    """
    Ftr=feature_matrix(model,X_fit,arch=arch,dataset=dataset,max_n=len(X_fit),device=DEVICE).cpu().numpy()
    Fte=feature_matrix(model,X_eval,arch=arch,dataset=dataset,max_n=len(X_eval),device=DEVICE).cpu().numpy()
    clf=LogisticRegression(max_iter=max_iter,C=C,solver='lbfgs',random_state=0).fit(Ftr,y_fit)
    prob=clf.predict_proba(Fte); classes=clf.classes_.astype(int); cmap={c:i for i,c in enumerate(classes)}
    idx=np.asarray([cmap[int(v)] for v in y_eval],int); loss=float(-np.log(np.clip(prob[np.arange(len(idx)),idx],1e-12,1)).mean())
    acc=float((clf.predict(Fte)==y_eval).mean())
    return loss,acc,clf


def copy_probe_to_head(model, task_id, clf):
    """Initialize a torch linear head from sklearn multinomial logistic regression.
    Handles binary sklearn's one-row coefficient convention exactly.
    """
    h=model.heads[task_id]
    W=np.asarray(clf.coef_,dtype=np.float32); b=np.asarray(clf.intercept_,dtype=np.float32)
    if W.shape[0]==1 and h.out_features==2:
        # sklearn uses logit(class1)=w x+b relative to class0. Symmetric two-logit form.
        W=np.vstack([-0.5*W[0],0.5*W[0]]); b=np.asarray([-0.5*b[0],0.5*b[0]],dtype=np.float32)
    if W.shape!=tuple(h.weight.shape):
        raise RuntimeError(f'probe/head shape mismatch {W.shape} vs {tuple(h.weight.shape)}')
    with torch.no_grad():
        h.weight.copy_(torch.as_tensor(W,device=h.weight.device)); h.bias.copy_(torch.as_tensor(b,device=h.bias.device))


def parameter_delta_vector(named_before, named_after):
    return torch.cat([(p1.detach()-p0.detach()).reshape(-1) for (_,p0),(_,p1) in zip(named_before,named_after)])


def snapshot_named(named):
    return [(n,p.detach().clone()) for n,p in named]


# ---------------------------------------------------------------------------
# Feature sampling helpers
# ---------------------------------------------------------------------------
def stratified_feature_indices(x0, eligible, n=64, seed=20260917):
    x0=np.asarray(x0,float); elig=np.where(np.asarray(eligible,bool))[0]
    if len(elig)<=n: return elig.tolist()
    # Quartiles are computed among eligible strengths. 16 per quartile, sampled at FROZEN random
    # within each quartile (not the lowest-valued members), with deterministic adjacent-quartile
    # redistribution if a quartile is short.
    order=elig[np.argsort(x0[elig])]; chunks=[np.asarray(c) for c in np.array_split(order,4)]
    rng=np.random.RandomState(seed); chunks=[c[rng.permutation(len(c))] for c in chunks]; target=[n//4]*4
    for i in range(n%4): target[i]+=1
    chosen=[]; leftover=[]
    for q,ch in enumerate(chunks):
        take=min(target[q],len(ch)); chosen.extend(ch[:take].tolist()); leftover.append(ch[take:].tolist()); target[q]-=take
    # Fill deficits from adjacent quartiles in deterministic distance order.
    for q in range(4):
        need=target[q]
        if need<=0: continue
        for d in (1,2,3):
            for qq in (q-d,q+d):
                if 0<=qq<4 and leftover[qq] and need>0:
                    k=min(need,len(leftover[qq])); chosen.extend(leftover[qq][:k]); leftover[qq]=leftover[qq][k:]; need-=k
            if need<=0: break
    return chosen[:n]


def feature_floor_mask(x0, q=.10, eps=1e-10):
    x0=np.asarray(x0,float); pos=x0[x0>1e-12]; floor=max(eps,float(np.quantile(pos,q)) if len(pos) else eps)
    return x0>=floor,floor


# ---------------------------------------------------------------------------
# Misc model statistics
# ---------------------------------------------------------------------------
def linear_cka_np(X,Y):
    X=np.asarray(X,float); Y=np.asarray(Y,float); n=min(len(X),len(Y)); X=X[:n]-X[:n].mean(0); Y=Y[:n]-Y[:n].mean(0)
    num=np.linalg.norm(X.T@Y,'fro')**2; den=np.linalg.norm(X.T@X,'fro')*np.linalg.norm(Y.T@Y,'fro')+1e-12
    return float(num/den)


def optimizer_update_dot(model, before_named):
    after=backbone_named_parameters(model); bdict={n:t for n,t in before_named};
    return torch.cat([(p.detach()-bdict[n].to(p.device)).reshape(-1) for n,p in after])


def quantile_ci(values, q=(.025,.975)):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    return None if not len(v) else [float(np.quantile(v,q[0])),float(np.quantile(v,q[1]))]

# ---------------------------------------------------------------------------
# Reviewer-facing task-cluster uncertainty helpers (2026-09-16 addendum)
# ---------------------------------------------------------------------------
def _weighted_pearson_np(x, y, w):
    x=np.asarray(x,float); y=np.asarray(y,float); w=np.asarray(w,float)
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    x,y,w=x[ok],y[ok],w[ok]
    if len(x)<3 or w.sum()<=0: return None
    w=w/w.sum(); mx=float(np.sum(w*x)); my=float(np.sum(w*y))
    vx=float(np.sum(w*(x-mx)**2)); vy=float(np.sum(w*(y-my)**2))
    if vx<=1e-18 or vy<=1e-18: return None
    return float(np.sum(w*(x-mx)*(y-my))/np.sqrt(vx*vy))


def _weighted_partial_r_np(x, y, controls, w):
    x=np.asarray(x,float); y=np.asarray(y,float); C=np.asarray(controls,float); w=np.asarray(w,float)
    if C.ndim==1: C=C[:,None]
    ok=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(C),axis=1)&np.isfinite(w)&(w>0)
    x,y,C,w=x[ok],y[ok],C[ok],w[ok]
    if len(x)<max(4,C.shape[1]+3): return None
    Z=np.c_[np.ones(len(C)),C]; sw=np.sqrt(w)[:,None]
    bx=np.linalg.lstsq(Z*sw,x*sw[:,0],rcond=None)[0]
    by=np.linalg.lstsq(Z*sw,y*sw[:,0],rcond=None)[0]
    return _weighted_pearson_np(x-Z@bx,y-Z@by,w)


def task_cluster_bootstrap_ci(rows: Sequence[dict], value_fn, *, task_keys=('task_a','task_b'),
                              n_boot=5000, seed=42, alpha=.05):
    """Cluster bootstrap over task identities for an arbitrary weighted statistic.

    ``value_fn(rows, weights)`` must return a scalar or None.  The task draw is
    converted to edge weights count(A)*count(B), preserving dyadic dependence.
    """
    rows=list(rows)
    if not rows: return {'estimate':None,'ci95':None,'n_boot':0,'bootstrap':'task-cluster'}
    ta=np.asarray([int(r[task_keys[0]]) for r in rows]); tb=np.asarray([int(r[task_keys[1]]) for r in rows])
    tasks=np.unique(np.r_[ta,tb]); rng=np.random.RandomState(seed)
    base_w=np.ones(len(rows),float); est=value_fn(rows,base_w)
    vals=[]
    for _ in range(int(n_boot)):
        draw=rng.choice(tasks,size=len(tasks),replace=True)
        cnt={int(t):int(np.sum(draw==t)) for t in tasks}
        w=np.asarray([cnt.get(int(a),0)*cnt.get(int(b),0) for a,b in zip(ta,tb)],float)
        if np.count_nonzero(w)>1:
            v=value_fn(rows,w)
            if v is not None and np.isfinite(v): vals.append(float(v))
    ci=None
    if vals:
        ci=[float(np.quantile(vals,alpha/2)),float(np.quantile(vals,1-alpha/2))]
    return {'estimate':None if est is None else float(est),'ci95':ci,'n_boot':len(vals),'bootstrap':'task-cluster'}


def bootstrap_pearson_ci(rows: Sequence[dict], x_key: str, y_key: str, *, n_boot=5000, seed=42):
    def fn(rs,w):
        return _weighted_pearson_np([r[x_key] for r in rs],[r[y_key] for r in rs],w)
    return task_cluster_bootstrap_ci(rows,fn,n_boot=n_boot,seed=seed)


def bootstrap_partial_corr_ci(rows: Sequence[dict], x_key: str, y_key: str, control_keys: Sequence[str], *,
                              n_boot=5000, seed=42):
    def fn(rs,w):
        C=np.asarray([[r[k] for k in control_keys] for r in rs],float)
        return _weighted_partial_r_np([r[x_key] for r in rs],[r[y_key] for r in rs],C,w)
    return task_cluster_bootstrap_ci(rows,fn,n_boot=n_boot,seed=seed)


def bootstrap_auc_ci(rows: Sequence[dict], score_key: str, outcome_key: str, *, n_boot=5000, seed=42):
    def fn(rs,w):
        y=np.asarray([int(r[outcome_key]) for r in rs]); s=np.asarray([float(r[score_key]) for r in rs])
        ok=np.isfinite(s)&(w>0)
        if ok.sum()<2 or len(np.unique(y[ok]))<2: return None
        return float(roc_auc_score(y[ok],s[ok],sample_weight=np.asarray(w)[ok]))
    return task_cluster_bootstrap_ci(rows,fn,n_boot=n_boot,seed=seed)


def train_probe_pool(X, y, *, n=256, seed=0):
    """Class-balanced labeled incoming-task probe drawn from TRAIN data only.

    This helper exists to make the pre-hoc supervision contract explicit and to
    prevent accidental use of benchmark test labels in rho/compatibility code.
    """
    Xp,yp,ids=stratified_probe(np.asarray(X),np.asarray(y),n=min(int(n),len(X)),seed=seed)
    return Xp,yp,ids
