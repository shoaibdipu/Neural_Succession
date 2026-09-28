# ============================================================
# E4 v4 — Minimum Replay Buffer Size (Pairwise, Frozen Buffer)
# Theorem 4 (Revised) of SLT
#
# Why previous versions failed:
#   E4v1/v2/v3 resampled from the FULL Task A training set at
#   every step. At α=0.05 with BATCH=128 and 78 steps/epoch,
#   the model sees ~9k Task A samples per epoch — nearly the
#   entire dataset. Too much diversity, forgetting collapses
#   for any task pair.
#
# The fix — frozen buffer sampled ONCE:
#   After Task A training, sample exactly M samples from Task A
#   ONCE and freeze them. During Task B training, replay ONLY
#   from this frozen set. At M=50 the model sees the same 50
#   samples repeatedly — diversity is limited. Only pairs where
#   Task B is similar enough to leave Task A features intact
#   can survive with a small buffer.
#
# NOTE on replacement sampling when M < REPLAY_BATCH:
#   We always draw REPLAY_BATCH=128 samples from the buffer
#   using replacement. When M < 128, samples repeat within
#   each batch. This avoids BatchNorm errors (needs ≥2 samples)
#   and keeps gradient magnitude consistent. The diversity
#   penalty from small M is still felt — at M=10 you see the
#   same 10 samples ~13 times per batch, which is weaker
#   regularization than M=200 with genuine diversity.
#
# Design:
#   T_A = T0 (plane/auto) — fixed anchor, always first
#   T_B ∈ {T1, T2, T3, T4} — four subsequent tasks
#   Buffer sizes M ∈ BUFFER_SIZES
#   For each (T_B, M): deepcopy trained T_A model
#                    → train T_B with frozen buffer of size M
#                    → measure T_A forgetting
#   Find M*(T_B) = first M where forgetting < THRESHOLD
#   Plot ρ_pre(T0, T_B) vs M* — Theorem 4: negative correlation
#
# Dataset: Split-CIFAR-10, ResNet-18, Task-IL
# ============================================================

# !pip install datasets -q

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import json
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS          = 20
BATCH           = 128
REPLAY_BATCH    = 128
LR              = 1e-3
SEED            = 42
RELATIVE_THRESH = 0.10   # M* = first M where forgetting < RELATIVE_THRESH × R_AA

# Fine-grained sweep: resolution 25 over full range 0-1000
# Gives 41 values — enough to rank 20 pairs without tie artifacts
BUFFER_SIZES = list(range(0, 1001, 25))
TASK_ANCHORS = [0, 1, 2, 3, 4]

# Debug mode: same anchors and buffer sizes, just fewer epochs and less data
# Runs all 20 pairs to check the pipeline end-to-end — not a subset
# Run with: python e4.py --debug
DEBUG           = False
DEBUG_EPOCHS    = 3      # enough to see forgetting direction
DEBUG_DATA_FRAC = 0.15   # 15% of each task's training data

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

TASK_SPLITS = {
    0: (0, 1), 1: (2, 3), 2: (4, 5), 3: (6, 7), 4: (8, 9),
}
TASK_NAMES = {
    0: 'plane/auto', 1: 'bird/cat',
    2: 'deer/dog',   3: 'frog/horse', 4: 'ship/truck',
}


# ── Data ──────────────────────────────────────────────────────────────────────
def load_cifar10():
    ds = load_dataset("uoft-cs/cifar10")
    def extract(split):
        imgs = np.array([np.array(x) for x in ds[split]['img']],
                        dtype=np.float32)
        imgs = (imgs / 255.0 - CIFAR_MEAN) / CIFAR_STD
        imgs = imgs.transpose(0, 3, 1, 2)
        labs = np.array(ds[split]['label'], dtype=np.int64)
        return imgs, labs
    Xtr, ytr = extract('train')
    Xte, yte = extract('test')
    print(f"Train {Xtr.shape}  Test {Xte.shape}")
    return Xtr, ytr, Xte, yte


def task_subset(X, y, pair):
    c0, c1 = pair
    m = (y == c0) | (y == c1)
    return X[m], (y[m] == c1).astype(np.int64)


# ── Model ─────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, n_tasks=5):
        super().__init__()
        bb    = resnet18(weights=None)
        bb.fc = nn.Identity()
        self.bb    = bb
        self.heads = nn.ModuleList([nn.Linear(512, 2) for _ in range(n_tasks)])

    def forward(self, x, t):
        return self.heads[t](self.bb(x))


# ── ρ_pre ─────────────────────────────────────────────────────────────────────
def compute_rho_pre(model, task_a_id, X_B_te, y_B_te, device, n_shuffle=5):
    model.eval()
    def _err(y_use):
        wrong, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_B_te), 256):
                xb = torch.tensor(X_B_te[i:i+256]).to(device)
                yb = torch.tensor(y_use[i:i+256]).to(device)
                wrong += (model(xb, task_a_id).argmax(1) != yb).sum().item()
                total += len(yb)
        return wrong / total
    eB   = _err(y_B_te)
    eBsf = np.mean([_err(np.random.permutation(y_B_te))
                    for _ in range(n_shuffle)])
    return float(1.0 - np.sqrt(eB / (eBsf + 1e-8)))


# ── Train ─────────────────────────────────────────────────────────────────────
def train_task(model, task_id, X, y, device, epochs=EPOCHS):
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
            loss_fn(model(xb, task_id), yb).backward()
            opt.step()


def train_with_frozen_buffer(model, task_b_id, task_a_id,
                              X_B, y_B, buf_X, buf_y, device, epochs=EPOCHS):
    """
    Train Task B with frozen Task A buffer.
    buf_X, buf_y: frozen buffer of size M, sampled ONCE before this call.
    Samples with replacement so REPLAY_BATCH is always 128 — BatchNorm safe.
    """
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i in [task_a_id, task_b_id])
    for p in model.bb.parameters():
        p.requires_grad = True

    loader  = DataLoader(TensorDataset(torch.tensor(X_B), torch.tensor(y_B)),
                         batch_size=BATCH, shuffle=True, drop_last=True)
    opt     = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    has_buf = buf_X is not None and len(buf_X) > 0
    M       = len(buf_X) if has_buf else 0

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, task_b_id), yb)
            if has_buf:
                # replace=True: when M < REPLAY_BATCH, samples repeat
                # within batch — diversity limited by M, BatchNorm safe
                idx = np.random.choice(M, REPLAY_BATCH, replace=True)
                xr  = torch.tensor(buf_X[idx]).to(device)
                yr  = torch.tensor(buf_y[idx]).to(device)
                loss = loss + loss_fn(model(xr, task_a_id), yr)
            loss.backward()
            opt.step()


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


# ── Parallel worker — one per anchor ──────────────────────────────────────────
def run_anchor(task_a, task_data, n_epochs, buf_sizes, gpu_id, result_queue):
    """
    Runs all (T_A, T_B, M) combinations for one anchor task.
    Spawned as a separate process — each gets its own GPU fraction.

    gpu_id: which CUDA device to use (0..N_GPUs-1, or 'cpu')
    result_queue: multiprocessing.Queue to collect results
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) if gpu_id != 'cpu' else ''
    device = 'cuda:0' if gpu_id != 'cpu' and torch.cuda.is_available() else 'cpu'

    # Re-seed per process for reproducibility
    torch.manual_seed(SEED + task_a)
    np.random.seed(SEED + task_a)

    XA_tr, yA_tr = task_data[task_a]['train']
    XA_te, yA_te = task_data[task_a]['test']
    task_b_ids   = [t for t in TASK_SPLITS if t != task_a]

    print(f"[GPU {gpu_id}] Anchor T{task_a} ({TASK_NAMES[task_a]}) starting...")
    model_A = Model().to(device)
    train_task(model_A, task_a, XA_tr, yA_tr, device, epochs=n_epochs)
    R_AA = evaluate(model_A, task_a, XA_te, yA_te, device)
    print(f"[GPU {gpu_id}] T{task_a} R_AA={R_AA:.3f}")

    anchor_results = []

    for task_b in task_b_ids:
        XB_tr, yB_tr = task_data[task_b]['train']
        XB_te, yB_te = task_data[task_b]['test']
        rho       = compute_rho_pre(model_A, task_a, XB_te, yB_te, device)
        pair_runs = []
        max_M     = max(buf_sizes)
        max_idx   = np.random.choice(len(XA_tr), max_M, replace=False)

        print(f"[GPU {gpu_id}] T{task_a}→T{task_b} ρ={rho:+.3f}")

        for M in buf_sizes:
            buf_X = XA_tr[max_idx[:M]] if M > 0 else None
            buf_y = yA_tr[max_idx[:M]] if M > 0 else None

            model_B = copy.deepcopy(model_A)
            train_with_frozen_buffer(model_B, task_b, task_a,
                                     XB_tr, yB_tr,
                                     buf_X, buf_y, device,
                                     epochs=n_epochs)
            R_BA       = evaluate(model_B, task_a, XA_te, yA_te, device)
            R_BB       = evaluate(model_B, task_b, XB_te, yB_te, device)
            forgetting = R_AA - R_BA
            pair_runs.append({'M': M, 'R_BA': R_BA, 'R_BB': R_BB,
                               'forgetting': forgetting})

            # Early stopping once M* found
            if forgetting < RELATIVE_THRESH * R_AA:
                break

        rel_thresh = RELATIVE_THRESH * R_AA
        M_star     = next((r['M'] for r in pair_runs
                           if r['forgetting'] < rel_thresh), None)

        anchor_results.append({
            'task_a':  task_a, 'task_b':  task_b,
            'label':   f"T{task_a}→T{task_b}",
            'name_b':  TASK_NAMES[task_b],
            'rho_pre': rho, 'R_AA': R_AA,
            'runs':    pair_runs, 'M_star': M_star,
        })
        print(f"[GPU {gpu_id}] T{task_a}→T{task_b} M*={M_star}")

    result_queue.put(anchor_results)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(device):
    import torch.multiprocessing as mp

    Xtr, ytr, Xte, yte = load_cifar10()
    task_data = {
        t: {'train': task_subset(Xtr, ytr, p),
            'test':  task_subset(Xte, yte, p)}
        for t, p in TASK_SPLITS.items()
    }

    n_epochs  = DEBUG_EPOCHS if DEBUG else EPOCHS
    buf_sizes = BUFFER_SIZES

    if DEBUG:
        print(f"DEBUG MODE: epochs={n_epochs}, data={DEBUG_DATA_FRAC*100:.0f}%")
        for t in task_data:
            X, y = task_data[t]['train']
            n    = max(BATCH * 4, int(len(X) * DEBUG_DATA_FRAC))
            task_data[t]['train'] = (X[:n], y[:n])

    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
        # ── Multi-GPU: spawn one process per anchor ───────────────────────────
        # Each anchor gets its own GPU (round-robin if more anchors than GPUs)
        print(f"\nMulti-GPU mode: {n_gpus} GPUs detected")
        print(f"Spawning {len(TASK_ANCHORS)} workers...")

        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()
        processes    = []

        for i, task_a in enumerate(TASK_ANCHORS):
            gpu_id = i % n_gpus
            p = mp.Process(target=run_anchor,
                           args=(task_a, task_data, n_epochs,
                                 buf_sizes, gpu_id, result_queue))
            p.start()
            processes.append(p)
            print(f"  Started anchor T{task_a} on GPU {gpu_id}")

        for p in processes:
            p.join()

        # Collect and flatten results
        all_results = []
        while not result_queue.empty():
            all_results.extend(result_queue.get())

        # Sort by task_a, task_b for consistent ordering
        results = sorted(all_results, key=lambda r: (r['task_a'], r['task_b']))

    else:
        # ── Single GPU / CPU: sequential ─────────────────────────────────────
        if n_gpus == 1:
            print(f"\nSingle GPU mode")
        else:
            print(f"\nCPU mode — consider using Colab T4 for ~20h runtime")

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        results = []

        for task_a in TASK_ANCHORS:
            XA_tr, yA_tr = task_data[task_a]['train']
            XA_te, yA_te = task_data[task_a]['test']
            task_b_ids   = [t for t in TASK_SPLITS if t != task_a]

            print(f"\n{'='*55}")
            print(f"Anchor: T{task_a} ({TASK_NAMES[task_a]})")
            print(f"{'='*55}")
            model_A = Model().to(device)
            train_task(model_A, task_a, XA_tr, yA_tr, device, epochs=n_epochs)
            R_AA = evaluate(model_A, task_a, XA_te, yA_te, device)
            print(f"R_AA = {R_AA:.3f}\n")

            for task_b in task_b_ids:
                XB_tr, yB_tr = task_data[task_b]['train']
                XB_te, yB_te = task_data[task_b]['test']
                rho       = compute_rho_pre(model_A, task_a, XB_te, yB_te, device)
                pair_runs = []
                max_M     = max(buf_sizes)
                max_idx   = np.random.choice(len(XA_tr), max_M, replace=False)

                print(f"T{task_a}→T{task_b} ({TASK_NAMES[task_b]:<12})  "
                      f"ρ_pre={rho:+.3f}  R_AA={R_AA:.3f}")
                print(f"  {'M':>5}  {'forget':>8}  {'R_BA':>8}  "
                      f"{'R_BB':>8}  {'ratio':>8}")

                for M in buf_sizes:
                    buf_X = XA_tr[max_idx[:M]] if M > 0 else None
                    buf_y = yA_tr[max_idx[:M]] if M > 0 else None

                    model_B = copy.deepcopy(model_A)
                    train_with_frozen_buffer(model_B, task_b, task_a,
                                             XB_tr, yB_tr,
                                             buf_X, buf_y, device,
                                             epochs=n_epochs)
                    R_BA       = evaluate(model_B, task_a, XA_te, yA_te, device)
                    R_BB       = evaluate(model_B, task_b, XB_te, yB_te, device)
                    forgetting = R_AA - R_BA
                    pair_runs.append({'M': M, 'R_BA': R_BA, 'R_BB': R_BB,
                                       'forgetting': forgetting})
                    print(f"  M={M:>4}  forget={forgetting:.3f}  "
                          f"R_BA={R_BA:.3f}  R_BB={R_BB:.3f}  "
                          f"ratio={R_BA/R_AA:.3f}")

                    # Early stopping: M* found — no need to test larger buffers
                    if forgetting < RELATIVE_THRESH * R_AA:
                        print(f"  ✓ threshold reached at M={M}, stopping sweep")
                        break

                rel_thresh = RELATIVE_THRESH * R_AA
                M_star     = next((r['M'] for r in pair_runs
                                   if r['forgetting'] < rel_thresh), None)

                results.append({
                    'task_a':  task_a, 'task_b':  task_b,
                    'label':   f"T{task_a}→T{task_b}",
                    'name_b':  TASK_NAMES[task_b],
                    'rho_pre': rho, 'R_AA': R_AA,
                    'runs':    pair_runs, 'M_star': M_star,
                })
                print(f"  → M* = {M_star}\n")

                with open('e4_checkpoint.json', 'w') as f:
                    json.dump(results, f)

    # Save final checkpoint
    with open('e4_checkpoint.json', 'w') as f:
        json.dump(results, f)

    return results


# ── Analyze ───────────────────────────────────────────────────────────────────
def analyze(results):
    # Recompute M* using relative threshold
    for r in results:
        rel_thresh  = RELATIVE_THRESH * r['R_AA']
        r['M_star'] = next((run['M'] for run in r['runs']
                            if run['forgetting'] < rel_thresh), None)

    # ── Anchor-normalized M* ──────────────────────────────────────────────────
    # Anchor vulnerability (e.g. T1 has low R_AA so any Task B hurts it more)
    # is a confound unrelated to pairwise similarity. Normalize within anchor:
    #   M*_norm(tA→tB) = M*(tA→tB) / mean(M*(tA→*))
    # This isolates: within a fixed anchor, does more dissimilar Task B
    # require more replay? That's the direct test of Theorem 4.
    from collections import defaultdict
    anchor_groups = defaultdict(list)
    for r in results:
        if r['M_star'] is not None:
            anchor_groups[r['task_a']].append(r['M_star'])

    anchor_mean = {a: np.mean(ms) for a, ms in anchor_groups.items()}

    for r in results:
        if r['M_star'] is not None and anchor_mean.get(r['task_a'], 0) > 0:
            r['M_star_norm'] = r['M_star'] / anchor_mean[r['task_a']]
        else:
            r['M_star_norm'] = None

    # ── Fit β₀, β₁ from E4's own M=0 forgetting data ────────────────────────
    # Instead of hardcoding E2 values, derive from this experiment:
    # forgetting(M=0) = β₀ + β₁ × (1 - ρ_pre)
    # This makes E4 self-contained.
    rho_vals = np.array([r['rho_pre']              for r in results])
    fgt_vals = np.array([r['runs'][0]['forgetting'] for r in results])  # M=0
    dissim   = 1 - rho_vals

    # Linear regression: fgt = β₀ + β₁ × dissim
    A        = np.vstack([np.ones_like(dissim), dissim]).T
    beta, _  = np.linalg.lstsq(A, fgt_vals, rcond=None)[:2]
    BETA0_fit, BETA1_fit = float(beta[0]), float(beta[1])

    print(f"\n  β fit from E4 M=0 data: β₀={BETA0_fit:.3f}  β₁={BETA1_fit:.3f}")
    # Store on results for use in plot
    for r in results:
        r['_beta0'] = BETA0_fit
        r['_beta1'] = BETA1_fit

    print(f"\n{'='*60}")
    print(f"  Relative threshold : {RELATIVE_THRESH} × R_AA")
    print(f"\n  {'Pair':<18}  {'ρ_pre':>7}  {'R_AA':>6}  "
          f"{'M*':>6}  {'M*_norm':>8}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['label']:<18}  {r['rho_pre']:>+7.3f}  "
              f"{r['R_AA']:>6.3f}  "
              f"{'N/A':>6}  {'N/A':>8}" if r['M_star'] is None else
              f"  {r['label']:<18}  {r['rho_pre']:>+7.3f}  "
              f"{r['R_AA']:>6.3f}  "
              f"{r['M_star']:>6}  {r['M_star_norm']:>8.3f}")

    # Raw correlation
    valid_raw  = [(r['rho_pre'], r['M_star'])
                  for r in results if r['M_star'] is not None]
    # Normalized correlation
    valid_norm = [(r['rho_pre'], r['M_star_norm'])
                  for r in results if r['M_star_norm'] is not None]

    print(f"\n  --- Raw M* ---")
    if len(valid_raw) > 2:
        vx, vy       = zip(*valid_raw)
        r_raw, p_raw = pearsonr(vx, vy)
        sp_raw, _    = spearmanr(vx, vy)
        print(f"  Pearson r : {r_raw:.3f}  p={p_raw:.4f}")
        print(f"  Spearman ρ: {sp_raw:.3f}")

    # ── Replay efficiency η ───────────────────────────────────────────────────
    # η = (forgetting_at_M0 - forgetting_at_M25) / 25
    # Simple, robust: total forgetting drop over the first 25 replay samples.
    # Using M=25 because every pair has at least M=0 and M=25 in the sweep.
    # No linear fit — avoids noise amplification from intermediate fluctuations.
    # Hypothesis: dissimilar pairs (low ρ_pre) have higher η because
    # the steep forgetting gradient means each sample yields a larger recovery.
    for r in results:
        fgt0 = next((run['forgetting'] for run in r['runs'] if run['M'] == 0),  None)
        fgt25= next((run['forgetting'] for run in r['runs'] if run['M'] == 25), None)
        if fgt0 is not None and fgt25 is not None:
            r['eta'] = float((fgt0 - fgt25) / 25)
        else:
            r['eta'] = None

    # ── M* with fixed threshold 0.10 (not relative) ──────────────────────────
    # Fixed threshold makes M* comparable across anchors regardless of R_AA.
    FIXED_THRESH = 0.10
    for r in results:
        r['M_star_fixed'] = next((run['M'] for run in r['runs']
                                  if run['forgetting'] < FIXED_THRESH), None)

    print(f"\n  --- Replay efficiency η = (fgt_M0 - fgt_M25) / 25 ---")
    print(f"  Hypothesis: dissimilar pairs (low ρ_pre) → higher η")
    valid_eta = [(r['rho_pre'], r['eta']) for r in results
                 if r['eta'] is not None]
    print(f"\n  {'Pair':<18}  {'ρ_pre':>7}  {'M=0 fgt':>9}  {'η':>8}  {'M*(fix)':>8}")
    print(f"  {'-'*56}")
    for r in results:
        fgt0 = r['runs'][0]['forgetting']
        mfix = r.get('M_star_fixed')
        print(f"  {r['label']:<18}  {r['rho_pre']:>+7.3f}  "
              f"{fgt0:>9.3f}  "
              f"{'N/A':>8}  {'N/A':>8}" if r['eta'] is None else
              f"  {r['label']:<18}  {r['rho_pre']:>+7.3f}  "
              f"{fgt0:>9.3f}  {r['eta']:>8.5f}  "
              f"{'never' if mfix is None else mfix:>8}")

    if len(valid_eta) > 2:
        vx, vy       = zip(*valid_eta)
        r_eta, p_eta = pearsonr(vx, vy)
        sp_eta, _    = spearmanr(vx, vy)
        print(f"\n  Pearson r  (ρ_pre vs η): {r_eta:.3f}  p={p_eta:.4f}")
        print(f"  Spearman ρ (ρ_pre vs η): {sp_eta:.3f}")
        if r_eta < -0.3 and p_eta < 0.05:
            print(f"  ✅  Theory HOLDS")
        elif r_eta < 0:
            print(f"  ⚠️  Correct direction, p={p_eta:.3f}")
        else:
            print(f"  ❌  No support for η hypothesis")

    print(f"{'='*60}")
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(results):
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    TASK_A_SIZE = 10000
    rhos    = np.array([r['rho_pre'] for r in results])
    norm_fn = mcolors.Normalize(vmin=rhos.min(), vmax=rhos.max())
    cmap    = mcm.RdYlGn
    sm      = mcm.ScalarMappable(cmap=cmap, norm=norm_fn)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ── Panel 1: forgetting curves ────────────────────────────────────────────
    ax = axes[0]
    for r in results:
        col  = cmap(norm_fn(r['rho_pre']))
        Ms   = [run['M']          for run in r['runs']]
        fgts = [run['forgetting'] for run in r['runs']]
        ax.plot(Ms, fgts, '-', color=col, lw=1.2, alpha=0.7)
    plt.colorbar(sm, ax=ax).set_label('ρ_pre  (green=similar)', fontsize=9)
    ax.set_xlabel('Buffer size M (frozen samples)', fontsize=11)
    ax.set_ylabel('Task A forgetting', fontsize=11)
    ax.set_title('Forgetting vs buffer size\n20 pairs, colored by ρ_pre',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: ρ_pre vs α* — fixed threshold 0.10 ──────────────────────────
    valid_r = [r for r in results if r.get('M_star_fixed') is not None]
    ax2 = axes[1]
    if valid_r:
        vx       = np.array([r['rho_pre']       for r in valid_r])
        vy_alpha = np.array([r['M_star_fixed']   for r in valid_r], dtype=float) / TASK_A_SIZE
        cols     = [cmap(norm_fn(r['rho_pre'])) for r in valid_r]

        ax2.scatter(vx, vy_alpha, s=80, c=cols, edgecolors='black',
                    linewidths=0.6, zorder=4)

        for i in {int(np.argmax(vx)), int(np.argmin(vx)), int(np.argmax(vy_alpha))}:
            ax2.annotate(valid_r[i]['label'], (vx[i], vy_alpha[i]),
                         fontsize=8, fontweight='bold',
                         xytext=(6, 4), textcoords='offset points')

        xs = np.linspace(vx.min()-0.05, vx.max()+0.05, 200)
        m, b = np.polyfit(vx, vy_alpha, 1)
        ax2.plot(xs, m*xs+b, 'k--', lw=1.5)

        if len(valid_r) > 2:
            r_val, p_val = pearsonr(vx, vy_alpha)
            sp_val, _    = spearmanr(vx, vy_alpha)
            ax2.text(0.97, 0.97,
                     f'r={r_val:.3f}  p={p_val:.3f}\nSpearman={sp_val:.3f}',
                     transform=ax2.transAxes, fontsize=8,
                     va='top', ha='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('ρ_pre  (pre-hoc similarity)', fontsize=11)
    ax2.set_ylabel('α*  =  M* / |Task A|  (fixed thresh=0.10)', fontsize=11)
    ax2.set_title('ρ_pre vs α*\nfixed threshold 0.10',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # ── Panel 3: ρ_pre vs η (replay efficiency) ───────────────────────────────
    valid_eta = [r for r in results if r.get('eta') is not None]
    ax3 = axes[2]
    if valid_eta:
        vx3  = np.array([r['rho_pre'] for r in valid_eta])
        vy3  = np.array([r['eta']     for r in valid_eta])
        cols3 = [cmap(norm_fn(r['rho_pre'])) for r in valid_eta]

        ax3.scatter(vx3, vy3, s=80, c=cols3, edgecolors='black',
                    linewidths=0.6, zorder=4)

        for i in {int(np.argmax(vx3)), int(np.argmin(vx3)), int(np.argmax(vy3))}:
            ax3.annotate(valid_eta[i]['label'], (vx3[i], vy3[i]),
                         fontsize=8, fontweight='bold',
                         xytext=(6, 4), textcoords='offset points')

        xs3  = np.linspace(vx3.min()-0.05, vx3.max()+0.05, 200)
        m, b = np.polyfit(vx3, vy3, 1)
        ax3.plot(xs3, m*xs3+b, 'k--', lw=1.5)

        if len(valid_eta) > 2:
            r_e, p_e  = pearsonr(vx3, vy3)
            sp_e, _   = spearmanr(vx3, vy3)
            ax3.text(0.97, 0.97,
                     f'r={r_e:.3f}  p={p_e:.3f}\nSpearman={sp_e:.3f}',
                     transform=ax3.transAxes, fontsize=8,
                     va='top', ha='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.set_xlabel('ρ_pre  (pre-hoc similarity)', fontsize=11)
    ax3.set_ylabel('η  =  (fgt_M0 − fgt_M25) / 25', fontsize=11)
    ax3.set_title('ρ_pre vs replay efficiency η\n'
                  'η = forgetting drop per sample at M=25',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.suptitle('E4 — Replay Buffer Analysis\n'
                 'CIFAR-10, ResNet-18',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('e4_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e4_results.png")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None,
                        help='Load checkpoint JSON')
    parser.add_argument('--debug', action='store_true',
                        help='Quick run: 2 anchors, 3 buffers, 10% data')
    args, _ = parser.parse_known_args()  # ignore Colab kernel args

    if args.debug:
        DEBUG = True  # module-level, no global needed outside a function

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.load:
        print(f"Loading from {args.load}")
        with open(args.load) as f:
            results = json.load(f)
    else:
        print(f"Device       : {device}")
        print(f"Anchors      : {TASK_ANCHORS} — {len(TASK_ANCHORS)*4} pairs total")
        print(f"Threshold    : {RELATIVE_THRESH} × R_AA")
        print(f"Buffer sizes : {BUFFER_SIZES}\n")
        results = run(device)

    analyze(results)
    plot(results)