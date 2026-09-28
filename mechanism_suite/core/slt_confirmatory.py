#!/usr/bin/env python3
"""Utilities shared by the theorem-faithful E1/E2/E4 reruns.

These helpers implement the frozen protocol decisions without changing slt_common's
historical Adam training functions, so the legacy/pilot experiments remain auditable.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slt_common import DEVICE, _reshape_for, _set_trainable, accuracy


def stratified_probe(X, y, n=100, seed=0):
    """Deterministic class-balanced probe sampled from TRAIN data.

    Returns Xp, yp, indices.  If n is not divisible by the number of classes,
    the remainder is distributed deterministically over sorted class ids.
    """
    X = np.asarray(X); y = np.asarray(y)
    classes = np.unique(y)
    rng = np.random.RandomState(seed)
    base, rem = divmod(int(n), len(classes))
    ids = []
    for rank, c in enumerate(sorted(classes.tolist())):
        pool = np.where(y == c)[0]
        take = min(len(pool), base + (1 if rank < rem else 0))
        ids.extend(rng.choice(pool, size=take, replace=False).tolist())
    # If a class was too small, fill from the remaining pool without replacement.
    if len(ids) < min(n, len(X)):
        left = np.setdiff1d(np.arange(len(X)), np.asarray(ids, int), assume_unique=False)
        k = min(min(n, len(X)) - len(ids), len(left))
        ids.extend(rng.choice(left, size=k, replace=False).tolist())
    ids = np.asarray(ids, dtype=int)
    rng.shuffle(ids)
    return X[ids], y[ids], ids


def stratified_train_val(X, y, val_frac=0.1, seed=0):
    """Return train/validation splits using only the original training set."""
    X = np.asarray(X); y = np.asarray(y)
    rng = np.random.RandomState(seed)
    tr_ids, va_ids = [], []
    for c in sorted(np.unique(y).tolist()):
        pool = np.where(y == c)[0].copy(); rng.shuffle(pool)
        nv = max(1, int(round(val_frac * len(pool))))
        va_ids.extend(pool[:nv].tolist()); tr_ids.extend(pool[nv:].tolist())
    tr_ids = np.asarray(tr_ids, int); va_ids = np.asarray(va_ids, int)
    rng.shuffle(tr_ids); rng.shuffle(va_ids)
    return (X[tr_ids], y[tr_ids]), (X[va_ids], y[va_ids]), tr_ids, va_ids


def freeze_batchnorm_stats(model):
    """Keep BN affine parameters trainable but freeze running statistics."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_sgd_theory(model, task_id, X, y, *, arch, dataset, epochs=None,
                     max_steps=None, lr=0.05, batch=128, seed=0,
                     freeze_bn=False, device=DEVICE, class_offset=0,
                     global_labels=False, weight_decay=0.0,
                     extra_loss_fn=None, eval_data=None):
    """Vanilla-SGD path for theorem-validation experiments.

    No momentum, no Adam.  If ``max_steps`` is set, batches are cycled until
    exactly that many optimizer steps have been taken.  Returns a diagnostics
    dict including the step count and cumulative LV time ``tau=steps*lr``.
    """
    if epochs is None and max_steps is None:
        raise ValueError("provide epochs or max_steps")
    _set_trainable(model, task_id)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    Xr = _reshape_for(arch, dataset, X)
    yy = y + class_offset if global_labels else y
    rng = np.random.RandomState(seed)
    n = len(Xr)
    steps = 0; ep = 0; epoch_losses = []
    target_epochs = int(epochs or 10**9)
    while ep < target_epochs:
        perm = rng.permutation(n)
        losses = []
        model.train()
        if freeze_bn:
            freeze_batchnorm_stats(model)
        # include the final partial batch; this makes predicted and actual step counts explicit
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            if len(idx) == 0:
                continue
            xb = torch.tensor(Xr[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(yy[idx], dtype=torch.long, device=device)
            feats = model.features(xb)
            logits = model.heads[task_id](feats) if model.scenario == 'task' else model.head(feats)
            loss = F.cross_entropy(logits, yb)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model, feats, logits, xb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if max_steps is not None and steps >= int(max_steps):
                break
        epoch_losses.append(float(np.mean(losses)) if losses else float('nan'))
        ep += 1
        if max_steps is not None and steps >= int(max_steps):
            break
    model.eval()
    out = {
        'steps': int(steps), 'epochs_completed': int(ep), 'lr': float(lr),
        'tau': float(steps * lr), 'epoch_loss': epoch_losses,
        'last_loss': epoch_losses[-1] if epoch_losses else None,
        'loss_improvement_last2': (epoch_losses[-2] - epoch_losses[-1]) if len(epoch_losses) >= 2 else None,
        'freeze_bn_stats': bool(freeze_bn), 'optimizer': 'SGD', 'momentum': 0.0,
    }
    if eval_data is not None:
        Xe, ye = eval_data
        out['eval_accuracy'] = float(accuracy(model, task_id, Xe, ye, arch=arch, dataset=dataset,
                                              global_labels=global_labels, class_offset=class_offset))
    return out


@torch.no_grad()
def accuracy_from_features(model, task_id, feats, y, *, class_offset=0):
    """Accuracy from an already-computed feature matrix (task-IL only)."""
    if model.scenario != 'task':
        raise ValueError('accuracy_from_features currently task-IL only')
    logits = model.heads[task_id](feats)
    yy = torch.as_tensor(y, dtype=torch.long, device=logits.device)
    return float((logits.argmax(1) == yy).float().mean().cpu())


def estimate_local_margin_sensitivity(model, task_id, X_val, y_val, x0_A, eligible,
                                      *, arch, dataset, attenuations=(0.02, 0.05, 0.10),
                                      device=DEVICE):
    """Pre-B conservative estimator for the repaired Theorem-2 constant c_A.

    On held-out Task-A TRAIN-validation examples only, uniformly attenuate the
    eligible penultimate features by fixed amounts.  For each attenuation a,
    Definition-2 strength scales by (1-a)^2, giving a known total strength loss.
    c_A is the minimum non-negative observed accuracy-drop / strength-loss ratio.
    A zero estimate is retained (a vacuous bound) rather than fitted away.
    """
    model.eval()
    Xr = _reshape_for(arch, dataset, X_val)
    xb = torch.tensor(Xr, dtype=torch.float32, device=device)
    yb = torch.tensor(y_val, dtype=torch.long, device=device)
    with torch.no_grad():
        feats = model.features(xb)
        base = float((model.heads[task_id](feats).argmax(1) == yb).float().mean().cpu())
    elig_idx = np.where(np.asarray(eligible, bool))[0]
    total_x = float(np.asarray(x0_A)[elig_idx].sum())
    rows = []
    slopes = []
    for a in attenuations:
        f2 = feats.clone()
        if len(elig_idx):
            f2[:, elig_idx] *= (1.0 - float(a))
        with torch.no_grad():
            acc = float((model.heads[task_id](f2).argmax(1) == yb).float().mean().cpu())
        delta_acc = max(0.0, base - acc)
        delta_x = max(0.0, total_x * (1.0 - (1.0 - float(a))**2))
        slope = delta_acc / delta_x if delta_x > 0 else 0.0
        rows.append({'attenuation': float(a), 'accuracy': acc, 'delta_acc': delta_acc,
                     'delta_x_strength': delta_x, 'slope': float(slope)})
        slopes.append(max(0.0, float(slope)))
    c_A = float(min(slopes)) if slopes else 0.0
    return c_A, {'base_accuracy': base, 'c_A': c_A, 'rows': rows,
                 'estimator': 'minimum non-negative Task-A validation attenuation slope'}


def git_commit(root):
    try:
        return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_metadata(root, extra=None):
    d = {'synthetic': os.environ.get('SLT_SYNTHETIC', '0') == '1'}
    if extra: d.update(extra)
    return d


def json_dump_safe(obj, path):
    def conv(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if torch.is_tensor(x): return x.detach().cpu().tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    def _scrub(o):
        import math
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: _scrub(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_scrub(v) for v in o]
        return o
    Path(path).write_text(json.dumps(_scrub(obj), indent=2, default=conv, allow_nan=False))
