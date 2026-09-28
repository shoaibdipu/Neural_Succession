#!/usr/bin/env python3
"""Utilities for the September-2026 EST revision experiments.

These helpers do NOT redefine EST.  They address specific reviewer-facing issues:
  * label-permutation invariance of the pre-hoc behavioral compatibility score;
  * task-identity-aware inference for dyadic task-pair data;
  * leave-one-task-out forecasting;
  * a signed, pre-hoc feature-pressure diagnostic consistent with gradient flow.

The original ``rho_pre`` is retained in ``slt_common`` for historical reproduction.
"""
from __future__ import annotations

import copy
import itertools
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from slt_common import DEVICE, _reshape_for, subsample


@torch.no_grad()
def task_predictions(model, task_id, X, *, arch, dataset, device=DEVICE, batch=256):
    """Return hard predictions from one task head on X."""
    model.eval().to(device)
    Xr = _reshape_for(arch, dataset, X)
    out = []
    for i in range(0, len(Xr), batch):
        xb = torch.as_tensor(Xr[i:i+batch], dtype=torch.float32, device=device)
        logits = model.heads[task_id](model.features(xb))
        out.append(logits.argmax(1).detach().cpu().numpy())
    return np.concatenate(out).astype(int)


def permutation_invariant_error_from_predictions(pred, y, n_classes=None):
    """Minimum classification error after an optimal one-to-one label assignment.

    This removes the arbitrary *local label index* problem when a head trained on
    Task A is evaluated on Task B.  For K classes, Hungarian assignment maximises
    matches between predicted head indices and Task-B class indices.
    """
    pred = np.asarray(pred, int); y = np.asarray(y, int)
    if n_classes is None:
        n_classes = int(max(pred.max(initial=0), y.max(initial=0)) + 1)
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, t in zip(pred, y):
        if 0 <= p < n_classes and 0 <= t < n_classes:
            C[p, t] += 1
    r, c = linear_sum_assignment(-C)
    matched = int(C[r, c].sum())
    return float(1.0 - matched / max(len(y), 1))


def rho_pre_permutation_invariant(modelA, taskA_id, X_B, y_B, *, arch, dataset,
                                  n_classes=None, n_samples=None, n_shuffle=20,
                                  device=DEVICE, seed=0):
    """Label-permutation-invariant analogue of historical rho_pre.

    rho_PI = 1 - sqrt(e_PI / e_PI,shuffle), where *both* numerator and shuffled
    baseline use the same optimal label assignment.  This keeps the historical
    normalization while removing arbitrary local class-index alignment.
    """
    Xs, ys = subsample(np.asarray(X_B), np.asarray(y_B), n_samples, seed=seed)
    pred = task_predictions(modelA, taskA_id, Xs, arch=arch, dataset=dataset, device=device)
    K = n_classes or int(max(pred.max(initial=0), ys.max(initial=0)) + 1)
    e = permutation_invariant_error_from_predictions(pred, ys, K)
    rng = np.random.RandomState(seed + 9173)
    sh = []
    for _ in range(int(n_shuffle)):
        yp = ys.copy(); rng.shuffle(yp)
        sh.append(permutation_invariant_error_from_predictions(pred, yp, K))
    e_sh = float(np.mean(sh))
    return float(1.0 - np.sqrt(e / (e_sh + 1e-8)))


@torch.no_grad()
def _feature_numpy(model, X, *, arch, dataset, device=DEVICE, batch=256):
    model.eval().to(device); Xr = _reshape_for(arch, dataset, X); out=[]
    for i in range(0, len(Xr), batch):
        xb = torch.as_tensor(Xr[i:i+batch], dtype=torch.float32, device=device)
        out.append(model.features(xb).detach().cpu().numpy())
    return np.concatenate(out, 0)


def frozen_probe_compatibility(modelA, X_B_train, y_B_train, X_B_test, y_B_test, *,
                               arch, dataset, n_classes, device=DEVICE, seed=0,
                               max_train=2000, max_test=2000, steps=200, lr=1e-2):
    """Label-invariant pre-hoc control: linear-probe transfer on frozen A features.

    No Task-B backbone training is performed.  Score is chance-normalised accuracy:
        (acc - 1/K)/(1 - 1/K).
    It is a *control metric*, not a replacement definition for the EST quantity.
    """
    Xtr, ytr = subsample(np.asarray(X_B_train), np.asarray(y_B_train), max_train, seed=seed)
    Xte, yte = subsample(np.asarray(X_B_test), np.asarray(y_B_test), max_test, seed=seed+1)
    ftr = torch.as_tensor(_feature_numpy(modelA, Xtr, arch=arch, dataset=dataset, device=device),
                          dtype=torch.float32, device=device)
    fte = torch.as_tensor(_feature_numpy(modelA, Xte, arch=arch, dataset=dataset, device=device),
                          dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    probe = nn.Linear(ftr.shape[1], int(n_classes)).to(device)
    g = torch.Generator(device='cpu'); g.manual_seed(int(seed))
    # deterministic initialisation independent of global model state
    torch.manual_seed(int(seed))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad(); loss = F.cross_entropy(probe(ftr), yt); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((probe(fte).argmax(1) == torch.as_tensor(yte, device=device)).float().mean().cpu())
    chance = 1.0 / float(n_classes)
    score = (acc - chance) / max(1.0 - chance, 1e-12)
    return float(score), float(acc)


def pearson_safe(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12:
        return float('nan'), float('nan')
    r,p=pearsonr(x[ok],y[ok]); return float(r),float(p)


def spearman_safe(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3: return float('nan'),float('nan')
    r,p=spearmanr(x[ok],y[ok]); return float(r),float(p)


def pair_means(rows, keys):
    """Average repeated seeds by directed task pair."""
    groups={}
    for r in rows:
        k=(int(r['task_a']),int(r['task_b'])); groups.setdefault(k,[]).append(r)
    out=[]
    for (a,b),rs in sorted(groups.items()):
        z={'task_a':a,'task_b':b,'n':len(rs)}
        for key in keys:
            vals=[float(r[key]) for r in rs if r.get(key) is not None and np.isfinite(float(r[key]))]
            z[key]=float(np.mean(vals)) if vals else None
        out.append(z)
    return out


def qap_task_label_permutation(pair_rows, x_key, y_key='forgetting', *,
                               n_tasks=None, n_perm=20000, seed=0, exact_max_tasks=8):
    """Task-label-aware QAP-style correlation test for directed dyadic data.

    One task-label permutation is applied simultaneously to the row/column labels
    of X while Y is fixed.  This preserves the dyadic dependency structure.
    For <= exact_max_tasks all task permutations are enumerated; otherwise Monte Carlo.
    """
    rows=pair_rows
    if n_tasks is None:
        n_tasks=max(max(int(r['task_a']),int(r['task_b'])) for r in rows)+1
    X=np.full((n_tasks,n_tasks),np.nan); Y=np.full_like(X,np.nan)
    for r in rows:
        a,b=int(r['task_a']),int(r['task_b'])
        if r.get(x_key) is not None: X[a,b]=float(r[x_key])
        if r.get(y_key) is not None: Y[a,b]=float(r[y_key])
    mask=np.isfinite(X)&np.isfinite(Y)&(~np.eye(n_tasks,dtype=bool))
    obs,_=pearson_safe(X[mask],Y[mask])
    perms = list(itertools.permutations(range(n_tasks))) if n_tasks<=exact_max_tasks else None
    rng=np.random.RandomState(seed)
    vals=[]
    iterator=perms if perms is not None else (rng.permutation(n_tasks) for _ in range(int(n_perm)))
    for p in iterator:
        p=np.asarray(p,int); Xp=X[np.ix_(p,p)]
        ok=np.isfinite(Xp)&np.isfinite(Y)&(~np.eye(n_tasks,dtype=bool))
        rp,_=pearson_safe(Xp[ok],Y[ok]); vals.append(rp)
    vals=np.asarray(vals,float); good=np.isfinite(vals)
    # inclusive finite-sample/randomisation p-value; exact case includes identity.
    pval=float(np.mean(np.abs(vals[good]) >= abs(obs)-1e-15)) if good.any() else float('nan')
    return {'r_observed':float(obs),'p_task_permutation':pval,'n_permutations':int(good.sum()),
            'exact':bool(perms is not None),'null_r_mean':float(np.nanmean(vals)),
            'null_r_std':float(np.nanstd(vals))}


def leave_one_task_out_forecast(pair_rows, x_key, y_key='forgetting', *, n_tasks=None,
                                incoming_only=True):
    """Forecast held-out-task transitions with zero task-identity leakage.

    For held-out task k, fitting excludes *every* edge touching k.  If
    incoming_only=True, evaluation uses A->k edges only, so each directed edge is
    evaluated once across folds.  The alternative evaluates all incident edges.
    """
    if n_tasks is None:
        n_tasks=max(max(int(r['task_a']),int(r['task_b'])) for r in pair_rows)+1
    folds=[]; all_y=[]; all_pred=[]
    for k in range(n_tasks):
        tr=[r for r in pair_rows if int(r['task_a'])!=k and int(r['task_b'])!=k]
        te=[r for r in pair_rows if (int(r['task_b'])==k if incoming_only else (int(r['task_a'])==k or int(r['task_b'])==k))]
        tr=[r for r in tr if r.get(x_key) is not None and r.get(y_key) is not None]
        te=[r for r in te if r.get(x_key) is not None and r.get(y_key) is not None]
        if len(tr)<3 or len(te)<1: continue
        X=np.asarray([[float(r[x_key])] for r in tr]); y=np.asarray([float(r[y_key]) for r in tr])
        mdl=LinearRegression().fit(X,y)
        Xt=np.asarray([[float(r[x_key])] for r in te]); yt=np.asarray([float(r[y_key]) for r in te])
        pr=mdl.predict(Xt); mae=float(np.mean(np.abs(pr-yt))); rr,_=pearson_safe(pr,yt)
        folds.append({'heldout_task':k,'n_train':len(tr),'n_test':len(te),'mae':mae,'r':rr,
                      'coef':float(mdl.coef_[0]),'intercept':float(mdl.intercept_)})
        all_y.extend(yt.tolist()); all_pred.extend(pr.tolist())
    r,_=pearson_safe(all_pred,all_y)
    return {'incoming_only':bool(incoming_only),'folds':folds,
            'overall_r':float(r),'overall_mae':float(np.mean(np.abs(np.asarray(all_pred)-np.asarray(all_y)))) if all_y else None,
            'n_predictions':len(all_y)}


def backbone_named_parameters(model):
    return [(n,p) for n,p in model.named_parameters() if ('heads' not in n and 'head' not in n)]


def backbone_l2_distance(model, reference):
    s=0.0
    ref=dict(reference.named_parameters())
    with torch.no_grad():
        for n,p in backbone_named_parameters(model):
            d=p.detach()-ref[n].detach().to(p.device); s += float((d*d).sum().cpu())
    return float(np.sqrt(s))


def train_frozen_probe(model, X_B, y_B, *, arch, dataset, n_classes, device=DEVICE,
                       max_train=1000, steps=200, lr=1e-2, seed=0):
    Xs,ys=subsample(np.asarray(X_B),np.asarray(y_B),max_train,seed=seed)
    f=torch.as_tensor(_feature_numpy(model,Xs,arch=arch,dataset=dataset,device=device),dtype=torch.float32,device=device)
    y=torch.as_tensor(ys,dtype=torch.long,device=device)
    torch.manual_seed(int(seed)); probe=nn.Linear(f.shape[1],int(n_classes)).to(device)
    opt=torch.optim.Adam(probe.parameters(),lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad(); loss=F.cross_entropy(probe(f),y); loss.backward(); opt.step()
    for p in probe.parameters(): p.requires_grad_(False)
    return probe


def feature_strength_tensor(model, X, *, arch, dataset, device=DEVICE):
    Xr=_reshape_for(arch,dataset,np.asarray(X))
    xb=torch.as_tensor(Xr,dtype=torch.float32,device=device)
    f=model.features(xb)
    return f.mean(0).pow(2)


def signed_feature_pressure_virtual_step(model, X_A_probe, X_B_probe, y_B_probe, *,
                                         arch, dataset, n_classes, X_B_probe_train=None,
                                         y_B_probe_train=None, eps=1e-4, device=DEVICE,
                                         seed=0):
    """Estimate signed pre-hoc pressure pi_i = <grad x_i^A, grad L_B>.

    A frozen linear Task-B probe supplies a meaningful pre-hoc B loss without
    updating the backbone.  A single virtual SGD step theta' = theta - eps*g_B
    gives
        pi_i ~= (x_i(theta) - x_i(theta')) / eps.
    Positive values predict resident-feature suppression; negative values predict
    facilitation/reinforcement.  Two eps values should be used by experiments as
    a finite-difference sensitivity check.
    """
    m=copy.deepcopy(model).to(device)
    XBt = X_B_probe if X_B_probe_train is None else X_B_probe_train
    yBt = y_B_probe if y_B_probe_train is None else y_B_probe_train
    probe=train_frozen_probe(m,XBt,yBt,arch=arch,dataset=dataset,n_classes=n_classes,
                             device=device,seed=seed)
    with torch.no_grad(): x0=feature_strength_tensor(m,X_A_probe,arch=arch,dataset=dataset,device=device).detach().cpu().numpy()
    Xr=_reshape_for(arch,dataset,np.asarray(X_B_probe)); xb=torch.as_tensor(Xr,dtype=torch.float32,device=device)
    yb=torch.as_tensor(y_B_probe,dtype=torch.long,device=device)
    m.zero_grad(set_to_none=True); loss=F.cross_entropy(probe(m.features(xb)),yb)
    named=backbone_named_parameters(m); grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
    grad_norm_sq=0.0
    with torch.no_grad():
        for (_,p),g in zip(named,grads):
            if g is not None:
                grad_norm_sq += float((g*g).sum().cpu()); p.add_(g,alpha=-float(eps))
        x1=feature_strength_tensor(m,X_A_probe,arch=arch,dataset=dataset,device=device).detach().cpu().numpy()
    pi=(x0-x1)/float(eps)
    return {'pi':pi,'x0':x0,'virtual_x1':x1,'eps':float(eps),
            'b_probe_loss':float(loss.detach().cpu()),'b_grad_norm':float(np.sqrt(grad_norm_sq))}


def partial_corr_residualized(x, y, controls):
    """Pearson partial correlation after linear residualization on controls.

    Returns a JSON-safe dictionary.  This is used as a scale-control diagnostic
    in theorem tests; it does not replace task-identity-aware inference.
    """
    x=np.asarray(x,float); y=np.asarray(y,float); C=np.asarray(controls,float)
    if C.ndim==1: C=C[:,None]
    ok=np.isfinite(x)&np.isfinite(y)&np.all(np.isfinite(C),axis=1)
    x=x[ok]; y=y[ok]; C=C[ok]
    if len(x)<4 or np.std(x)<1e-12 or np.std(y)<1e-12:
        return {'partial_r':None,'p':None,'n':int(len(x))}
    Z=np.concatenate([np.ones((len(C),1)),C],axis=1)
    bx=np.linalg.lstsq(Z,x,rcond=None)[0]; by=np.linalg.lstsq(Z,y,rcond=None)[0]
    rx=x-Z@bx; ry=y-Z@by
    if np.std(rx)<1e-12 or np.std(ry)<1e-12:
        return {'partial_r':None,'p':None,'n':int(len(x))}
    r,p=pearsonr(rx,ry)
    return {'partial_r':float(r),'p':float(p),'n':int(len(x))}


def shuffled_matrix_null(alpha, weights, observed, *, control=None, n_perm=1000,
                         seed=0, shuffle='columns'):
    """Null test for a matrix-derived pressure by disrupting alpha structure.

    ``pressure_i = sum_j alpha_ij * weights_j``.  The default column shuffle
    preserves each row's coefficient distribution while breaking the mapping
    between task-B feature identity and its strength.  If ``control`` is given,
    the test statistic is the partial correlation controlling that covariate;
    otherwise it is ordinary Pearson correlation.
    """
    A=np.asarray(alpha,float); w=np.asarray(weights,float); y=np.asarray(observed,float)
    if A.ndim!=2 or A.shape[1]!=len(w):
        raise ValueError('alpha/weights shape mismatch')
    rng=np.random.RandomState(seed)
    lam=A@w
    if control is None:
        obs,_=pearson_safe(lam,y)
    else:
        obs=partial_corr_residualized(lam,y,control)['partial_r']
        obs=float('nan') if obs is None else float(obs)
    vals=[]
    for _ in range(int(n_perm)):
        if shuffle=='columns':
            p=rng.permutation(A.shape[1]); Ap=A[:,p]
        elif shuffle=='entries_within_row':
            Ap=np.stack([row[rng.permutation(len(row))] for row in A],axis=0)
        else:
            raise ValueError(shuffle)
        lp=Ap@w
        if control is None:
            rr,_=pearson_safe(lp,y)
        else:
            rr=partial_corr_residualized(lp,y,control)['partial_r']
            rr=float('nan') if rr is None else float(rr)
        vals.append(rr)
    vals=np.asarray(vals,float); good=np.isfinite(vals)
    pval=float((1+np.sum(np.abs(vals[good])>=abs(obs)))/(1+good.sum())) if good.any() and np.isfinite(obs) else float('nan')
    return {'observed_r':float(obs) if np.isfinite(obs) else None,
            'p_shuffled_alpha':pval,'n_permutations':int(good.sum()),
            'null_mean':float(np.nanmean(vals)) if good.any() else None,
            'null_std':float(np.nanstd(vals)) if good.any() else None,
            'shuffle':shuffle}
