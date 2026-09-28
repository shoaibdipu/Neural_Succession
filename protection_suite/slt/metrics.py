from __future__ import annotations
from typing import Dict, Iterable, Sequence
import numpy as np
import torch

@torch.no_grad()
def accuracy(model, dl, device, task_classes: Sequence[int] | None=None) -> float:
    model.eval(); ok=0; n=0
    mask=None
    if task_classes is not None:
        mask=torch.full((model.fc.out_features,), False, dtype=torch.bool, device=device)
        mask[list(task_classes)]=True
    for x,y in dl:
        x=x.to(device); y=y.to(device); z=model(x)
        if mask is not None:
            z=z.masked_fill(~mask[None,:], -1e9)
        ok += (z.argmax(1)==y).sum().item(); n += len(y)
    return ok/max(n,1)

def forgetting(acc_before:float, acc_after:float) -> float:
    return float(acc_before-acc_after)

def bwt_from_matrix(R: np.ndarray) -> float:
    # R[t,j] accuracy on task j after training task t; t >= j.
    T=R.shape[0]
    vals=[R[T-1,j]-R[j,j] for j in range(T-1)]
    return float(np.mean(vals)) if vals else 0.0

def avg_accuracy(R:np.ndarray)->float:
    return float(np.nanmean(R[-1]))
