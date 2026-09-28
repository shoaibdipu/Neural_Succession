# ============================================================
# E1.5 — Pre-hoc vs Post-hoc Similarity Variants
#
# Question: Does the pre-hoc (one-directional) similarity
# predict forgetting as well as the full bidirectional ρ_AB?
#
# Three variants of similarity, all from Li & Hiratani Eq. 16:
#
#   ρ_full  (post-hoc, bidirectional):
#     1 - (1/2)(sqrt(ε_B[W_A]/ε_{B,sf}[W_A])
#              + sqrt(ε_A[W_B]/ε_{A,sf}[W_B]))
#     Requires both models trained. Best estimate of similarity.
#
#   ρ_pre   (pre-hoc, unidirectional):
#     1 - sqrt(ε_B[W_A]/ε_{B,sf}[W_A])
#     Only Task A model needed. Computable before Task B training.
#     The only genuinely pre-hoc variant.
#
#   ρ_rev   (reverse direction only):
#     1 - sqrt(ε_A[W_B]/ε_{A,sf}[W_B])
#     Only Task B model needed.
#     Isolated to understand contribution of each direction.
#
# Outputs:
#   1. Correlation of each variant with forgetting
#   2. Correlation between ρ_pre and ρ_full (are they redundant?)
#   3. Scatter grid: all three vs forgetting side by side
#
# Dataset: Split-CIFAR-10, ResNet-18, Task-IL
# ============================================================

# !pip install datasets -q

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

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS = 20
BATCH  = 128
LR     = 1e-3
SEED   = 42

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

TASK_SPLITS = {
    0: (0, 1), 1: (2, 3), 2: (4, 5), 3: (6, 7), 4: (8, 9),
}
TASK_NAMES = {
    0: 'plane/auto', 1: 'bird/cat',   2: 'deer/dog',
    3: 'frog/horse', 4: 'ship/truck',
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


# ── Train / Evaluate ──────────────────────────────────────────────────────────
def train(model, task_id, X, y, device):
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i == task_id)
    for p in model.bb.parameters():
        p.requires_grad = True

    loader  = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                         batch_size=BATCH, shuffle=True)
    opt     = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb, task_id), yb).backward()
            opt.step()


def error(model, task_id, X, y, device, batch=256):
    model.eval()
    wrong, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch]).to(device)
            yb = torch.tensor(y[i:i+batch]).to(device)
            wrong += (model(xb, task_id).argmax(1) != yb).sum().item()
            total += len(yb)
    return wrong / total


def shuffled_error(model, task_id, X, y, device, n=5):
    errs = []
    for _ in range(n):
        y_s = y.copy()
        np.random.shuffle(y_s)
        errs.append(error(model, task_id, X, y_s, device))
    return float(np.mean(errs))


# ── Three similarity variants ─────────────────────────────────────────────────
def compute_similarities(mA, mB, tA, tB,
                         XAte, yAte, XBte, yBte, device):
    """
    Returns all three similarity variants from a trained mA and mB.

    ρ_pre  — pre-hoc:  only uses Task A model (mA)
    ρ_rev  — reverse:  only uses Task B model (mB)
    ρ_full — post-hoc: average of both directions (Li & Hiratani Eq. 16)

    All three are on the same scale [~-0.4, 1.0].
    """
    # Forward term: Task A model on Task B data
    eB_WA  = error(mA, tA, XBte, yBte, device)
    eBs_WA = shuffled_error(mA, tA, XBte, yBte, device)
    t1     = np.sqrt(eB_WA / (eBs_WA + 1e-8))

    # Reverse term: Task B model on Task A data
    eA_WB  = error(mB, tB, XAte, yAte, device)
    eAs_WB = shuffled_error(mB, tB, XAte, yAte, device)
    t2     = np.sqrt(eA_WB / (eAs_WB + 1e-8))

    rho_pre  = float(1.0 - t1)           # pre-hoc: only forward direction
    rho_rev  = float(1.0 - t2)           # reverse direction only
    rho_full = float(1.0 - 0.5*(t1+t2)) # post-hoc: bidirectional average

    return rho_pre, rho_rev, rho_full


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(device):
    Xtr, ytr, Xte, yte = load_cifar10()

    task_data = {
        t: {'train': task_subset(Xtr, ytr, p),
            'test':  task_subset(Xte, yte, p)}
        for t, p in TASK_SPLITS.items()
    }

    all_pairs = [(tA, tB) for tA in range(5)
                          for tB in range(5) if tA != tB]
    results   = []
    total     = len(all_pairs)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{total} pairs\n")
    print(f"{'#':>3}  {'Pair':<12}  {'ρ_pre':>7}  {'ρ_rev':>7}  "
          f"{'ρ_full':>7}  {'forget':>8}")
    print("-" * 56)

    for idx, (tA, tB) in enumerate(all_pairs, 1):
        XAtr, yAtr = task_data[tA]['train']
        XAte, yAte = task_data[tA]['test']
        XBtr, yBtr = task_data[tB]['train']
        XBte, yBte = task_data[tB]['test']

        # Train Task A
        mA = Model().to(device)
        train(mA, tA, XAtr, yAtr, device)
        R_AA = 1 - error(mA, tA, XAte, yAte, device)

        # Train Task B (separate model — for similarity only)
        mB = Model().to(device)
        train(mB, tB, XBtr, yBtr, device)

        # All three similarity variants
        rho_pre, rho_rev, rho_full = compute_similarities(
            mA, mB, tA, tB, XAte, yAte, XBte, yBte, device
        )

        # Measure actual forgetting
        train(mA, tB, XBtr, yBtr, device)
        R_BA       = 1 - error(mA, tA, XAte, yAte, device)
        forgetting = R_AA - R_BA

        results.append({
            'label':     f"T{tA}→T{tB}",
            'tA': tA,    'tB': tB,
            'rho_pre':   rho_pre,
            'rho_rev':   rho_rev,
            'rho_full':  rho_full,
            'R_AA':      R_AA,
            'R_BA':      R_BA,
            'forgetting': forgetting,
        })

        print(f"[{idx:>2}/{total}]  "
              f"T{tA}→T{tB} ({TASK_NAMES[tA][:6]}→{TASK_NAMES[tB][:6]})  "
              f"ρ_pre={rho_pre:+.3f}  ρ_rev={rho_rev:+.3f}  "
              f"ρ_full={rho_full:+.3f}  forget={forgetting:.3f}")

        with open('e15_checkpoint.json', 'w') as f:
            json.dump(results, f)

    return results


# ── Analyze + Plot ────────────────────────────────────────────────────────────
def analyze(results):
    pre  = np.array([r['rho_pre']   for r in results])
    rev  = np.array([r['rho_rev']   for r in results])
    full = np.array([r['rho_full']  for r in results])
    fgt  = np.array([r['forgetting'] for r in results])
    lbls = [r['label'] for r in results]

    # Correlations with forgetting
    r_pre,  p_pre  = pearsonr(pre,  fgt)
    r_rev,  p_rev  = pearsonr(rev,  fgt)
    r_full, p_full = pearsonr(full, fgt)

    # Correlation between pre-hoc and full (are they measuring same thing?)
    r_cross, p_cross = pearsonr(pre, full)

    sp_pre,  _ = spearmanr(pre,  fgt)
    sp_full, _ = spearmanr(full, fgt)

    print(f"\n{'='*58}")
    print(f"  Correlation with forgetting:")
    print(f"    ρ_pre  (pre-hoc,  A→B only) : r={r_pre:.3f}  "
          f"p={p_pre:.4f}  Sp={sp_pre:.3f}")
    print(f"    ρ_rev  (reverse,  B→A only) : r={r_rev:.3f}  "
          f"p={p_rev:.4f}")
    print(f"    ρ_full (post-hoc, both dirs) : r={r_full:.3f}  "
          f"p={p_full:.4f}  Sp={sp_full:.3f}")
    print(f"\n  Correlation between ρ_pre and ρ_full:")
    print(f"    r={r_cross:.3f}  p={p_cross:.4f}")
    if r_cross > 0.85:
        print(f"    → Pre-hoc closely tracks post-hoc — "
              f"pre-hoc claim is justified")
    elif r_cross > 0.6:
        print(f"    → Moderate agreement — pre-hoc is noisier but useful")
    else:
        print(f"    → Weak agreement — pre-hoc misses important signal")
    print(f"{'='*58}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    variants = [
        (pre,  r_pre,  p_pre,  'ρ_pre  (pre-hoc: A→B only)',  'steelblue'),
        (rev,  r_rev,  p_rev,  'ρ_rev  (reverse: B→A only)',  'darkorange'),
        (full, r_full, p_full, 'ρ_full (post-hoc: both dirs)', 'crimson'),
    ]

    for ax, (rhos, r_v, p_v, title, col) in zip(axes, variants):
        ax.scatter(rhos, fgt, s=65, color=col,
                   edgecolors='white', linewidths=0.6, zorder=3)
        for lbl, rv, fv in zip(lbls, rhos, fgt):
            ax.annotate(lbl, (rv, fv), fontsize=6.5, alpha=0.75,
                        xytext=(4, 3), textcoords='offset points')
        xs   = np.linspace(rhos.min()*0.95, rhos.max()*1.05, 100)
        m, b = np.polyfit(rhos, fgt, 1)
        ax.plot(xs, m*xs+b, '--', color=col, lw=1.5,
                label=f'r={r_v:.3f} (p={p_v:.4f})')
        ax.set_xlabel('Similarity', fontsize=11)
        ax.set_ylabel('Forgetting  =  R_AA − R_BA', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.suptitle('E1.5 — Pre-hoc vs Post-hoc Similarity vs Forgetting\n'
                 '(CIFAR-10, Split-5 tasks, ResNet-18)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('e15_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e15_results.png")

    # ── Pre-hoc vs Full scatter ────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(pre, full, s=65, color='purple',
                edgecolors='white', linewidths=0.6, zorder=3)
    for lbl, pv, fv in zip(lbls, pre, full):
        ax2.annotate(lbl, (pv, fv), fontsize=6.5, alpha=0.75,
                     xytext=(4, 3), textcoords='offset points')
    xs2   = np.linspace(pre.min()*0.95, pre.max()*1.05, 100)
    m2, b2 = np.polyfit(pre, full, 1)
    ax2.plot(xs2, m2*xs2+b2, 'r--', lw=1.5,
             label=f'r={r_cross:.3f} (p={p_cross:.4f})')
    ax2.set_xlabel('ρ_pre  (pre-hoc, A→B only)', fontsize=11)
    ax2.set_ylabel('ρ_full  (post-hoc, both dirs)', fontsize=11)
    ax2.set_title('Pre-hoc vs Post-hoc similarity\n'
                  'High r here → pre-hoc is sufficient',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('e15_pre_vs_full.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e15_pre_vs_full.png")

    return {
        'r_pre': r_pre,   'p_pre': p_pre,
        'r_rev': r_rev,   'p_rev': p_rev,
        'r_full': r_full, 'p_full': p_full,
        'r_cross': r_cross,
    }


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    results = run(device)
    stats   = analyze(results)