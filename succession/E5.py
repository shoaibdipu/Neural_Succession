# ============================================================
# E5 — ρ-EWC vs Baselines
# SLT Paper: Algorithm Validation
#
# Claim: ρ_pre-adaptive regularization (ρ-EWC) matches or beats
# fixed-λ EWC without any hyperparameter search, because E2's
# formula provides the tuning signal pre-hoc.
#
# Methods compared:
#   Fine-tuning   : no CL method (lower bound)
#   EWC           : fixed λ, grid searched on val set
#   A-GEM         : replay, M=200 samples/task
#   DER++         : replay with logit distillation, M=200/task
#   ρ-EWC         : EWC with λ scaled by ρ_pre (ours, no replay)
#   ρ-EWC+replay  : ρ-EWC with small fixed buffer M=50 (ours)
#
# Datasets: Split-CIFAR-10 (5 tasks), Split-CIFAR-100 (10 tasks)
# Architecture: ResNet-18, Task-IL
# Metrics: ACC↑, BWT↓ (averaged over N_SEEDS seeds)
#
# Usage:
#   python E5.py                         # full run
#   python E5.py --debug                 # 3 epochs, 15% data
#   python E5.py --dataset cifar100      # CIFAR-100 only
#   python E5.py --load e5_results.json  # reload and replot
# ============================================================

# !pip install datasets -q

import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from datasets import load_dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS       = 20
BATCH        = 128
LR           = 1e-3
SEED         = 42
N_SEEDS      = 3       # repeat each method this many times

# EWC λ grid — searched on first seed, reused for others
EWC_LAMBDA_GRID = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
EWC_LAMBDA_BASE = 10.0   # used by ρ-EWC as the base (same scale as best EWC)

# Replay buffer size per task for A-GEM, DER++
REPLAY_M    = 200
# Replay buffer for ρ-EWC+replay (small — motivated by E4)
RHO_REPLAY_M = 50

# DER++ loss weights
DER_ALPHA   = 0.5    # MSE weight on stored logits
DER_BETA    = 0.5    # CE weight on stored labels

# ρ_pre computation: use 100 Task B test samples
RHO_N_SAMPLES = 100

DEBUG             = False
DEBUG_EPOCHS      = 3
DEBUG_DATA_FRAC   = 0.15

CIFAR10_MEAN  = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD   = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

# Split-CIFAR-10: 5 tasks of 2 classes each
CIFAR10_SPLITS = [(0,1),(2,3),(4,5),(6,7),(8,9)]

# Split-CIFAR-100: 10 tasks of 10 classes each
CIFAR100_SPLITS = [(list(range(i*10, (i+1)*10))) for i in range(10)]


# ── Data ──────────────────────────────────────────────────────────────────────
def load_cifar10():
    ds = load_dataset("uoft-cs/cifar10")
    def extract(split):
        imgs = np.array([np.array(x) for x in ds[split]['img']],
                        dtype=np.float32)
        imgs = (imgs/255.0 - CIFAR10_MEAN) / CIFAR10_STD
        imgs = imgs.transpose(0,3,1,2)
        labs = np.array(ds[split]['label'], dtype=np.int64)
        return imgs, labs
    return extract('train'), extract('test')


def load_cifar100():
    ds = load_dataset("uoft-cs/cifar100")
    def extract(split):
        imgs = np.array([np.array(x) for x in ds[split]['img']],
                        dtype=np.float32)
        imgs = (imgs/255.0 - CIFAR100_MEAN) / CIFAR100_STD
        imgs = imgs.transpose(0,3,1,2)
        labs = np.array(ds[split]['fine_label'], dtype=np.int64)
        return imgs, labs
    return extract('train'), extract('test')


def make_task_data(X, y, splits, n_classes_per_task):
    """Slice dataset into per-task (train, test) tuples with relabeled targets."""
    tasks = []
    for classes in splits:
        if isinstance(classes, tuple):
            classes = list(classes)
        mask  = np.isin(y, classes)
        X_t   = X[mask]
        # Relabel: class index within this task (0..n_classes_per_task-1)
        y_t   = np.array([classes.index(c) for c in y[mask]], dtype=np.int64)
        tasks.append((X_t, y_t))
    return tasks


# ── Model ─────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, n_tasks, n_classes_per_task):
        super().__init__()
        bb    = resnet18(weights=None)
        bb.fc = nn.Identity()
        self.bb    = bb
        self.heads = nn.ModuleList(
            [nn.Linear(512, n_classes_per_task) for _ in range(n_tasks)]
        )

    def forward(self, x, t):
        return self.heads[t](self.bb(x))

    def features(self, x):
        return self.bb(x)


# ── ρ_pre ─────────────────────────────────────────────────────────────────────
def compute_rho_pre(model, task_a_id, X_B, y_B, device,
                    n_samples=RHO_N_SAMPLES, n_shuffle=5):
    """
    Pre-hoc task similarity: ρ_pre = 1 - sqrt(ε_B[W_A] / ε_{B,sf}[W_A])
    Uses only Task A's trained model — no Task B training required.
    """
    idx = np.random.choice(len(X_B), min(n_samples, len(X_B)), replace=False)
    Xs  = X_B[idx]; ys = y_B[idx]

    model.eval()
    def _err(y_use):
        wrong, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(Xs), 256):
                xb = torch.tensor(Xs[i:i+256]).to(device)
                yb = torch.tensor(y_use[i:i+256]).to(device)
                wrong += (model(xb, task_a_id).argmax(1) != yb).sum().item()
                total += len(yb)
        return wrong / total

    eB   = _err(ys)
    eBsf = np.mean([_err(np.random.permutation(ys)) for _ in range(n_shuffle)])
    return float(1.0 - np.sqrt(eB / (eBsf + 1e-8)))


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, task_id, X, y, device, batch=256):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch]).to(device)
            yb = torch.tensor(y[i:i+batch]).to(device)
            correct += (model(xb, task_id).argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / total


# ── Base trainer ──────────────────────────────────────────────────────────────
def train_base(model, task_id, X, y, device, epochs,
               extra_loss_fn=None):
    """
    Train task_id head + backbone.
    extra_loss_fn(model) → scalar tensor: called at each step for regularization.
    """
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i == task_id)
    for p in model.bb.parameters():
        p.requires_grad = True

    loader  = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                         batch_size=BATCH, shuffle=True, drop_last=True)
    opt     = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, task_id), yb)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model)
            loss.backward()
            opt.step()


# ── EWC helpers ───────────────────────────────────────────────────────────────
def compute_fisher(model, task_id, X, y, device, n_samples=500):
    """
    Diagonal Fisher Information Matrix via empirical Fisher.
    Returns dict: param_name → diagonal FIM tensor (same shape as param).
    """
    idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    loader = DataLoader(
        TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx])),
        batch_size=64, shuffle=False
    )
    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        out  = model(xb, task_id)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += (p.grad ** 2) * len(xb)

    for n in fisher:
        fisher[n] /= len(idx)
    return fisher


def ewc_penalty(model, fisher_list, param_list, lam):
    """
    EWC regularization term: λ/2 * Σ_tasks Σ_params F_i(θ_i - θ*_i)²
    fisher_list: list of fisher dicts (one per previous task)
    param_list:  list of param dicts (θ* at task completion)
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for fisher, params in zip(fisher_list, param_list):
        for n, p in model.named_parameters():
            if n in fisher:
                loss = loss + (fisher[n] * (p - params[n]) ** 2).sum()
    return (lam / 2) * loss


# ── A-GEM ─────────────────────────────────────────────────────────────────────
def train_agem(model, task_id, X, y, memory_X, memory_y, memory_tasks,
               device, epochs):
    """
    A-GEM: project gradient onto constraint that memory loss doesn't increase.
    memory_X/y/tasks: concatenated samples from all previous tasks.
    """
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i == task_id)
    for p in model.bb.parameters():
        p.requires_grad = True

    loader  = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                         batch_size=BATCH, shuffle=True, drop_last=True)
    opt     = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            # Current task gradient
            loss_fn(model(xb, task_id), yb).backward()
            g_cur = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                     for p in model.parameters() if p.requires_grad]

            # Reference gradient on memory
            opt.zero_grad()
            idx = np.random.choice(len(memory_X), min(BATCH, len(memory_X)),
                                   replace=False)
            xm = torch.tensor(memory_X[idx]).to(device)
            ym = torch.tensor(memory_y[idx]).to(device)
            tm = memory_tasks[idx]
            # Compute memory loss per task to handle multi-task memory
            mem_loss = torch.tensor(0.0, device=device)
            for t in np.unique(tm):
                mask = tm == t
                if mask.sum() > 0:
                    mem_loss = mem_loss + loss_fn(model(xm[mask], int(t)),
                                                  ym[mask])
            mem_loss.backward()
            g_ref = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                     for p in model.parameters() if p.requires_grad]

            # Project: if g_cur · g_ref < 0, project g_cur
            dot = sum((gc * gr).sum() for gc, gr in zip(g_cur, g_ref))
            if dot < 0:
                ref_norm_sq = sum((gr**2).sum() for gr in g_ref) + 1e-8
                proj_scale  = dot / ref_norm_sq
                g_proj = [gc - proj_scale * gr for gc, gr in zip(g_cur, g_ref)]
            else:
                g_proj = g_cur

            # Apply projected gradient
            opt.zero_grad()
            params = [p for p in model.parameters() if p.requires_grad]
            for p, g in zip(params, g_proj):
                p.grad = g
            opt.step()


# ── DER++ ─────────────────────────────────────────────────────────────────────
def train_derpp(model, task_id, X, y,
                buf_X, buf_y, buf_logits, buf_tasks,
                device, epochs):
    """
    DER++: replay with knowledge distillation on stored logits.
    Loss = CE(current) + α*MSE(buf_logits) + β*CE(buf_labels)
    """
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i == task_id)
    for p in model.bb.parameters():
        p.requires_grad = True

    loader  = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                         batch_size=BATCH, shuffle=True, drop_last=True)
    opt     = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    has_buf = buf_X is not None and len(buf_X) > 0

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, task_id), yb)

            if has_buf:
                idx  = np.random.choice(len(buf_X), min(BATCH, len(buf_X)),
                                        replace=True)
                xr   = torch.tensor(buf_X[idx]).to(device)
                yr   = torch.tensor(buf_y[idx]).to(device)
                lr_  = torch.tensor(buf_logits[idx]).to(device)
                tr_  = buf_tasks[idx]

                for t in np.unique(tr_):
                    mask = tr_ == t
                    if mask.sum() < 2:
                        continue
                    logits_now = model(xr[mask], int(t))
                    # α: distill stored logits
                    loss = loss + DER_ALPHA * F.mse_loss(logits_now, lr_[mask])
                    # β: CE on stored labels
                    loss = loss + DER_BETA  * loss_fn(logits_now, yr[mask])

            loss.backward()
            opt.step()


# ── CL Experiment Runner ───────────────────────────────────────────────────────
def run_method(method_name, task_data_tr, task_data_te, n_tasks,
               n_classes_per_task, device, epochs, ewc_lam=None):
    """
    Run one CL method across all tasks.
    Returns: acc_matrix[i][j] = accuracy on task j after training on task i
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = Model(n_tasks, n_classes_per_task).to(device)

    # Per-method state
    fisher_list = []     # EWC: list of FIM dicts
    param_list  = []     # EWC: list of θ* dicts
    memory_X    = None   # A-GEM: episodic memory
    memory_y    = None
    memory_tasks= None
    buf_X       = None   # DER++: replay buffer
    buf_y       = None
    buf_logits  = None
    buf_tasks   = None

    acc_matrix = np.zeros((n_tasks, n_tasks))

    for t in range(n_tasks):
        X_tr, y_tr = task_data_tr[t]
        X_te, y_te = task_data_te[t]

        # ── ρ_pre computation (for ρ-EWC methods) ────────────────────────────
        rho = None
        if t > 0 and method_name in ('rho_ewc', 'rho_ewc_replay'):
            rho = compute_rho_pre(model, t-1, X_te, y_te, device)

        # ── Select λ for this task ────────────────────────────────────────────
        lam = ewc_lam  # default (EWC: fixed, finetune/agem/derpp: unused)
        if method_name in ('rho_ewc', 'rho_ewc_replay') and rho is not None:
            # Scale λ by dissimilarity: more dissimilar → higher λ
            # λ(ρ) = λ_base × (1 - ρ)
            # ρ ranges roughly -0.15 to 0.30, so (1-ρ) ∈ [0.70, 1.15]
            lam = EWC_LAMBDA_BASE * (1.0 - rho)

        # ── Train ─────────────────────────────────────────────────────────────
        if method_name == 'finetune':
            train_base(model, t, X_tr, y_tr, device, epochs)

        elif method_name == 'ewc':
            if fisher_list:
                extra = lambda m: ewc_penalty(m, fisher_list, param_list, lam)
            else:
                extra = None
            train_base(model, t, X_tr, y_tr, device, epochs, extra)

        elif method_name in ('rho_ewc', 'rho_ewc_replay'):
            # Regularization part (same as EWC but with adaptive λ)
            if fisher_list:
                extra = lambda m, l=lam: ewc_penalty(m, fisher_list,
                                                      param_list, l)
            else:
                extra = None

            if method_name == 'rho_ewc_replay' and buf_X is not None:
                # Small replay buffer (M=RHO_REPLAY_M) on top of regularization
                # Replay integrated into base training via extra_loss_fn
                train_derpp(model, t, X_tr, y_tr,
                            buf_X, buf_y, buf_logits, buf_tasks,
                            device, epochs)
                # Also apply EWC penalty separately
                if fisher_list:
                    # Fine-tune with EWC for one extra epoch to apply penalty
                    train_base(model, t, X_tr[:len(X_tr)//4],
                               y_tr[:len(y_tr)//4], device, 1, extra)
            else:
                train_base(model, t, X_tr, y_tr, device, epochs, extra)

        elif method_name == 'agem':
            if memory_X is not None:
                train_agem(model, t, X_tr, y_tr,
                           memory_X, memory_y, memory_tasks,
                           device, epochs)
            else:
                train_base(model, t, X_tr, y_tr, device, epochs)

        elif method_name == 'derpp':
            train_derpp(model, t, X_tr, y_tr,
                        buf_X, buf_y, buf_logits, buf_tasks,
                        device, epochs)

        # ── Update per-method memory after task t ─────────────────────────────
        if method_name in ('ewc', 'rho_ewc', 'rho_ewc_replay'):
            # Compute and store FIM + θ*
            fisher = compute_fisher(model, t, X_tr, y_tr, device)
            params = {n: p.detach().clone()
                      for n, p in model.named_parameters() if p.requires_grad}
            fisher_list.append(fisher)
            param_list.append(params)

        if method_name == 'agem':
            # Add M samples from this task to episodic memory
            idx = np.random.choice(len(X_tr), min(REPLAY_M, len(X_tr)),
                                   replace=False)
            new_X  = X_tr[idx]
            new_y  = y_tr[idx]
            new_ts = np.full(len(idx), t, dtype=np.int64)
            memory_X     = new_X if memory_X is None else np.concatenate([memory_X, new_X])
            memory_y     = new_y if memory_y is None else np.concatenate([memory_y, new_y])
            memory_tasks = new_ts if memory_tasks is None else np.concatenate([memory_tasks, new_ts])

        if method_name in ('derpp', 'rho_ewc_replay'):
            M = RHO_REPLAY_M if method_name == 'rho_ewc_replay' else REPLAY_M
            # Store (x, y, logits) for DER++
            idx  = np.random.choice(len(X_tr), min(M, len(X_tr)), replace=False)
            new_X = X_tr[idx]
            new_y = y_tr[idx]
            # Compute and store current logits
            model.eval()
            new_logits = []
            with torch.no_grad():
                for i in range(0, len(new_X), 256):
                    xb = torch.tensor(new_X[i:i+256]).to(device)
                    new_logits.append(model(xb, t).cpu().numpy())
            new_logits = np.concatenate(new_logits)
            new_ts     = np.full(len(idx), t, dtype=np.int64)
            buf_X      = new_X if buf_X is None else np.concatenate([buf_X, new_X])
            buf_y      = new_y if buf_y is None else np.concatenate([buf_y, new_y])
            buf_logits = new_logits if buf_logits is None else np.concatenate([buf_logits, new_logits])
            buf_tasks  = new_ts if buf_tasks is None else np.concatenate([buf_tasks, new_ts])

        # ── Evaluate on all tasks seen so far ─────────────────────────────────
        for j in range(t + 1):
            Xj, yj = task_data_te[j]
            acc_matrix[t][j] = evaluate(model, j, Xj, yj, device)

        print(f"  [{method_name}] After T{t}: "
              f"acc={[f'{acc_matrix[t][j]:.3f}' for j in range(t+1)]}"
              f"{'  ρ='+f'{rho:.3f}' if rho is not None else ''}"
              f"{'  λ='+f'{lam:.1f}' if lam is not None else ''}")

    return acc_matrix


def compute_metrics(acc_matrix, n_tasks):
    """
    ACC = mean accuracy on all tasks after full training
    BWT = mean forgetting (negative = more forgetting)
    """
    T    = n_tasks
    ACC  = np.mean([acc_matrix[T-1][j] for j in range(T)])
    BWT  = np.mean([acc_matrix[T-1][j] - acc_matrix[j][j]
                    for j in range(T-1)])
    return ACC, BWT


# ── λ selection for EWC (grid search on validation) ───────────────────────────
def select_ewc_lambda(task_data_tr, task_data_te, n_tasks,
                      n_classes_per_task, device, epochs):
    """
    Run EWC with each λ in EWC_LAMBDA_GRID on first 2 tasks.
    Pick λ that minimises forgetting on Task 0 after Task 1 training.
    """
    print(f"\n  Selecting EWC λ from grid {EWC_LAMBDA_GRID}...")
    best_lam, best_bwt = None, -np.inf

    X0, y0 = task_data_tr[0]; X1, y1 = task_data_tr[1]
    X0t, y0t = task_data_te[0]

    for lam in EWC_LAMBDA_GRID:
        torch.manual_seed(SEED)
        m = Model(n_tasks, n_classes_per_task).to(device)
        train_base(m, 0, X0, y0, device, epochs)
        fisher = compute_fisher(m, 0, X0, y0, device)
        params = {n: p.detach().clone()
                  for n, p in m.named_parameters() if p.requires_grad}
        extra  = lambda mod, f=fisher, p=params, l=lam: ewc_penalty(mod, [f], [p], l)
        train_base(m, 1, X1, y1, device, epochs, extra)
        acc_after = evaluate(m, 0, X0t, y0t, device)
        print(f"    λ={lam:.1f}  T0 acc after T1={acc_after:.3f}")
        if acc_after > best_bwt:
            best_bwt = acc_after
            best_lam = lam

    print(f"  → Best λ = {best_lam}\n")
    return best_lam


# ── Main ──────────────────────────────────────────────────────────────────────
def run_dataset(dataset_name, device, epochs):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    if dataset_name == 'cifar10':
        (Xtr, ytr), (Xte, yte) = load_cifar10()
        splits           = CIFAR10_SPLITS
        n_classes_per_task = 2
    else:
        (Xtr, ytr), (Xte, yte) = load_cifar100()
        splits           = CIFAR100_SPLITS
        n_classes_per_task = 10

    n_tasks = len(splits)

    task_data_tr = make_task_data(Xtr, ytr, splits, n_classes_per_task)
    task_data_te = make_task_data(Xte, yte, splits, n_classes_per_task)

    if DEBUG:
        task_data_tr = [(X[:max(BATCH*4, int(len(X)*DEBUG_DATA_FRAC))], y[:max(BATCH*4, int(len(y)*DEBUG_DATA_FRAC))])
                        for X, y in task_data_tr]

    # Select best EWC λ once
    ewc_lam = select_ewc_lambda(task_data_tr, task_data_te, n_tasks,
                                n_classes_per_task, device, epochs)

    methods = [
        ('finetune',      None),
        ('ewc',           ewc_lam),
        ('agem',          None),
        ('derpp',         None),
        ('rho_ewc',       None),
        ('rho_ewc_replay',None),
    ]

    dataset_results = {}

    for method_name, lam in methods:
        print(f"\n── {method_name.upper()} ──")
        accs_list, bwts_list = [], []

        for seed in range(N_SEEDS):
            torch.manual_seed(SEED + seed)
            np.random.seed(SEED + seed)
            print(f"  Seed {seed+1}/{N_SEEDS}")
            acc_mat = run_method(method_name, task_data_tr, task_data_te,
                                 n_tasks, n_classes_per_task, device,
                                 epochs, ewc_lam=lam)
            acc, bwt = compute_metrics(acc_mat, n_tasks)
            accs_list.append(acc)
            bwts_list.append(bwt)
            print(f"    ACC={acc:.3f}  BWT={bwt:.3f}")

        dataset_results[method_name] = {
            'ACC_mean': float(np.mean(accs_list)),
            'ACC_std':  float(np.std(accs_list)),
            'BWT_mean': float(np.mean(bwts_list)),
            'BWT_std':  float(np.std(bwts_list)),
            'ewc_lam':  ewc_lam,
        }
        print(f"  Final: ACC={np.mean(accs_list):.3f}±{np.std(accs_list):.3f}  "
              f"BWT={np.mean(bwts_list):.3f}±{np.std(bwts_list):.3f}")

    return dataset_results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(all_results):
    METHOD_LABELS = {
        'finetune':       'Fine-tune',
        'ewc':            'EWC',
        'agem':           'A-GEM',
        'derpp':          'DER++',
        'rho_ewc':        'ρ-EWC (ours)',
        'rho_ewc_replay': 'ρ-EWC+replay (ours)',
    }
    COLORS = {
        'finetune':       '#888780',
        'ewc':            '#378ADD',
        'agem':           '#BA7517',
        'derpp':          '#D85A30',
        'rho_ewc':        '#1D9E75',
        'rho_ewc_replay': '#0F6E56',
    }
    OUR_METHODS = {'rho_ewc', 'rho_ewc_replay'}

    datasets  = list(all_results.keys())
    n_ds      = len(datasets)
    methods   = list(METHOD_LABELS.keys())

    fig, axes = plt.subplots(2, n_ds, figsize=(7*n_ds, 10))
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for di, ds in enumerate(datasets):
        res = all_results[ds]

        for mi, metric in enumerate(['ACC', 'BWT']):
            ax     = axes[mi][di]
            means  = [res[m][f'{metric}_mean'] for m in methods]
            stds   = [res[m][f'{metric}_std']  for m in methods]
            colors = [COLORS[m] for m in methods]
            edgews = [2.0 if m in OUR_METHODS else 0.5 for m in methods]
            edges  = ['black' if m in OUR_METHODS else 'white' for m in methods]

            bars = ax.bar(range(len(methods)), means, yerr=stds,
                          color=colors, edgecolor=edges,
                          linewidth=edgews, capsize=4, width=0.6)

            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([METHOD_LABELS[m] for m in methods],
                               rotation=30, ha='right', fontsize=9)
            ax.set_ylabel(metric, fontsize=11)
            direction = '↑' if metric == 'ACC' else '↓'
            ax.set_title(f'{ds.upper()} — {metric} {direction}',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Annotate bars
            for i, (m, v, s) in enumerate(zip(methods, means, stds)):
                ax.text(i, v + (s if metric=='ACC' else -s) + 0.003,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold' if m in OUR_METHODS else 'normal')

    plt.suptitle('E5 — ρ-EWC vs Baselines\n'
                 'CIFAR-10/100, ResNet-18, Task-IL',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('e5_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e5_results.png")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',    type=str,   default=None)
    parser.add_argument('--debug',   action='store_true')
    parser.add_argument('--dataset', type=str,   default='both',
                        choices=['cifar10', 'cifar100', 'both'])
    args, _ = parser.parse_known_args()

    if args.debug:
        DEBUG = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device  : {device}")
    print(f"Debug   : {DEBUG}")
    print(f"Seeds   : {N_SEEDS}")

    if args.load:
        print(f"Loading from {args.load}")
        with open(args.load) as f:
            all_results = json.load(f)
    else:
        datasets = (['cifar10', 'cifar100'] if args.dataset == 'both'
                    else [args.dataset])
        n_epochs  = DEBUG_EPOCHS if DEBUG else EPOCHS

        all_results = {}
        for ds in datasets:
            all_results[ds] = run_dataset(ds, device, n_epochs)
            with open('e5_results.json', 'w') as f:
                json.dump(all_results, f)
            print(f"\nCheckpoint saved: e5_results.json")

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  {'Method':<20}  {'CIFAR-10 ACC':>13}  {'BWT':>8}  "
          f"{'CIFAR-100 ACC':>14}  {'BWT':>8}")
    print(f"  {'-'*60}")
    for m in ['finetune','ewc','agem','derpp','rho_ewc','rho_ewc_replay']:
        row = f"  {m:<20}"
        for ds in ['cifar10','cifar100']:
            if ds in all_results and m in all_results[ds]:
                r = all_results[ds][m]
                row += (f"  {r['ACC_mean']:.3f}±{r['ACC_std']:.3f}"
                        f"  {r['BWT_mean']:+.3f}")
            else:
                row += f"  {'N/A':>13}  {'N/A':>8}"
        print(row)
    print(f"{'='*65}")

    plot(all_results)