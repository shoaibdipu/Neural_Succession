# ============================================================
# E2 — Pre-hoc Forgetting Prediction
# Theorem 2 of SLT
#
# Hypothesis: ρ_pre (computed before Task B training)
#             predicts actual forgetting with MAE < 10%
#             on Split-CIFAR-10.
#
# ρ_pre = 1 - sqrt(ε_B[W_A] / ε_{B,sf}[W_A])
#
#   ε_B[W_A]     : error of Task A model on Task B test data
#   ε_{B,sf}[W_A]: chance-level error (label-shuffled)
#
# Only Task A's trained model is needed — no Task B training.
# This is the first genuinely pre-hoc forgetting predictor
# in the continual learning literature.
#
# Prediction formula:
#   predicted_forgetting = β₀ + β₁ × ρ_pre
#   Fit via leave-one-out cross-validation across 20 pairs.
#   MAE and correlation reported on held-out predictions.
#
# Protocol:
#   - All 20 ordered pairs from Split-CIFAR-10
#   - Leave-one-out CV: fit on 19, predict on 1
#   - Report: MAE, Pearson r, scatter plot predicted vs actual
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
from sklearn.linear_model import LinearRegression
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
    """ResNet-18 backbone + per-task Linear(512,2) heads. Task-IL."""
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


# ── Pre-hoc similarity (unidirectional) ──────────────────────────────────────
def rho_pre(model_A, tA, XBte, yBte, device):
    """
    ρ_pre = 1 - sqrt(ε_B[W_A] / ε_{B,sf}[W_A])

    Pre-hoc: only Task A's trained model needed.
    Measures how well Task A's features transfer to Task B.
    High ρ_pre → tasks similar → predict low forgetting.
    Low ρ_pre  → tasks dissimilar → predict high forgetting.

    Computable before any Task B training begins.
    """
    eB   = error(model_A, tA, XBte, yBte, device)
    eBsf = shuffled_error(model_A, tA, XBte, yBte, device)
    return float(1.0 - np.sqrt(eB / (eBsf + 1e-8)))


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(device):
    """
    For each of 20 pairs:
      1. Train Task A
      2. Compute ρ_pre — pre-hoc, no Task B training
      3. Train Task B → measure actual forgetting
    """
    Xtr, ytr, Xte, yte = load_cifar10()
    task_data = {
        t: {'train': task_subset(Xtr, ytr, p),
            'test':  task_subset(Xte, yte, p)}
        for t, p in TASK_SPLITS.items()
    }

    all_pairs = [(tA, tB) for tA in range(5)
                          for tB in range(5) if tA != tB]
    total     = len(all_pairs)
    results   = []

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{total} pairs\n")
    print(f"{'#':>3}  {'Pair':<12}  {'ρ_pre':>7}  "
          f"{'R_AA':>6}  {'R_BA':>6}  {'forget':>8}")
    print("-" * 50)

    for idx, (tA, tB) in enumerate(all_pairs, 1):
        XAtr, yAtr = task_data[tA]['train']
        XAte, yAte = task_data[tA]['test']
        XBtr, yBtr = task_data[tB]['train']
        XBte, yBte = task_data[tB]['test']

        # 1. Train Task A
        mA   = Model().to(device)
        train(mA, tA, XAtr, yAtr, device)
        R_AA = 1 - error(mA, tA, XAte, yAte, device)

        # 2. Compute ρ_pre — PRE-HOC, no Task B training
        rho = rho_pre(mA, tA, XBte, yBte, device)

        # 3. Train Task B → measure actual forgetting
        train(mA, tB, XBtr, yBtr, device)
        R_BA       = 1 - error(mA, tA, XAte, yAte, device)
        forgetting = R_AA - R_BA

        results.append({
            'label':      f"T{tA}→T{tB}",
            'tA': tA,     'tB': tB,
            'rho_pre':    rho,
            'R_AA':       R_AA,
            'R_BA':       R_BA,
            'forgetting': forgetting,
        })

        print(f"[{idx:>2}/{total}]  "
              f"T{tA}→T{tB} ({TASK_NAMES[tA][:6]}→{TASK_NAMES[tB][:6]})  "
              f"ρ_pre={rho:+.3f}  "
              f"R_AA={R_AA:.3f}  R_BA={R_BA:.3f}  "
              f"forget={forgetting:.3f}")

        with open('e2_checkpoint.json', 'w') as f:
            json.dump(results, f)

    return results


# ── Leave-one-out prediction ──────────────────────────────────────────────────
def leave_one_out(rhos, fgts):
    """
    Fit linear model on 19 pairs, predict on held-out pair.
    Repeat for all 20 pairs.

    Returns predicted forgetting for each pair using only
    information available before that pair's Task B training.
    """
    n    = len(rhos)
    preds = np.zeros(n)
    for i in range(n):
        # Training set: all pairs except i
        X_tr = rhos[np.arange(n) != i].reshape(-1, 1)
        y_tr = fgts[np.arange(n) != i]
        reg  = LinearRegression().fit(X_tr, y_tr)
        preds[i] = reg.predict([[rhos[i]]])[0]
    return preds


# ── Analyze + Plot ────────────────────────────────────────────────────────────
def analyze(results):
    rhos = np.array([r['rho_pre']    for r in results])
    fgts = np.array([r['forgetting'] for r in results])
    lbls = [r['label'] for r in results]

    # Overall correlation
    r_val, p_val = pearsonr(rhos, fgts)
    sp_val, _    = spearmanr(rhos, fgts)

    # Leave-one-out predictions
    preds = leave_one_out(rhos, fgts)
    mae   = float(np.mean(np.abs(preds - fgts)))
    r_loo, p_loo = pearsonr(preds, fgts)

    print(f"\n{'='*55}")
    print(f"  ρ_pre vs actual forgetting:")
    print(f"    Pearson r  : {r_val:.3f}  (p={p_val:.4f})")
    print(f"    Spearman ρ : {sp_val:.3f}")
    print(f"    R²         : {r_val**2:.3f}  "
          f"({r_val**2*100:.1f}% variance explained)")
    print(f"\n  Leave-one-out prediction (pre-hoc):")
    print(f"    MAE        : {mae:.4f}  ({mae*100:.2f} pp)")
    print(f"    Pearson r  : {r_loo:.3f}  (p={p_loo:.4f})")

    target = 0.10
    if mae < target:
        verdict = f"✅  MAE < {target*100:.0f}pp — pre-hoc prediction HOLDS"
    else:
        verdict = f"⚠️  MAE = {mae*100:.2f}pp — above {target*100:.0f}pp target"
    print(f"    Verdict    : {verdict}")
    print(f"{'='*55}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ρ_pre vs actual forgetting
    ax = axes[0]
    ax.scatter(rhos, fgts, s=70, color='steelblue',
               edgecolors='white', linewidths=0.6, zorder=3)
    for lbl, rv, fv in zip(lbls, rhos, fgts):
        ax.annotate(lbl, (rv, fv), fontsize=7, alpha=0.8,
                    xytext=(4, 3), textcoords='offset points')
    xs   = np.linspace(rhos.min()*0.95, rhos.max()*1.05, 100)
    m, b = np.polyfit(rhos, fgts, 1)
    ax.plot(xs, m*xs+b, 'r--', lw=1.5,
            label=f'r={r_val:.3f} (p={p_val:.4f})\nR²={r_val**2:.3f}')
    ax.set_xlabel('ρ_pre  (pre-hoc similarity, higher = more similar)',
                  fontsize=11)
    ax.set_ylabel('Forgetting  =  R_AA − R_BA', fontsize=11)
    ax.set_title('ρ_pre vs actual forgetting\n'
                 '(negative slope: similar tasks forget less)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Right: LOO predicted vs actual forgetting
    ax2 = axes[1]
    ax2.scatter(preds, fgts, s=70, color='crimson',
                edgecolors='white', linewidths=0.6, zorder=3)
    for lbl, pv, fv in zip(lbls, preds, fgts):
        ax2.annotate(lbl, (pv, fv), fontsize=7, alpha=0.8,
                     xytext=(4, 3), textcoords='offset points')

    # Perfect prediction line
    lims = [min(preds.min(), fgts.min()) - 0.02,
            max(preds.max(), fgts.max()) + 0.02]
    ax2.plot(lims, lims, 'k--', lw=1, alpha=0.4, label='perfect prediction')

    # Regression line on LOO predictions
    m2, b2 = np.polyfit(preds, fgts, 1)
    xs2    = np.linspace(lims[0], lims[1], 100)
    ax2.plot(xs2, m2*xs2+b2, 'r--', lw=1.5,
             label=f'r={r_loo:.3f} (p={p_loo:.4f})\nMAE={mae*100:.2f}pp')

    # Error bands ± MAE
    ax2.fill_between(lims,
                     [l - mae for l in lims],
                     [l + mae for l in lims],
                     alpha=0.1, color='red', label=f'±MAE band')

    ax2.set_xlabel('Predicted forgetting  (leave-one-out, pre-hoc)',
                   fontsize=11)
    ax2.set_ylabel('Actual forgetting', fontsize=11)
    ax2.set_title('E2: Pre-hoc forgetting prediction\n'
                  '(leave-one-out cross-validation)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e2_results.png")

    return {'r_val': r_val, 'p_val': p_val, 'mae': mae, 'r_loo': r_loo}


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}\n")
    results = run(device)
    stats   = analyze(results)