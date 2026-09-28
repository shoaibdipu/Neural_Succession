# ============================================================
# E3 — Coexistence Under Niche Partitioning
# Theorem 3 of SLT (revised sign)
#
# Original design (incorrect direction):
#   Overlapping tasks → forgetting
#   Orthogonal tasks  → coexistence
#
# Revised prediction (consistent with E1/E2 findings):
#   Similar tasks   → coexistence  (low forgetting)
#   Dissimilar tasks → forgetting   (high forgetting)
#
# Experimental design:
#   Group A — SIMILAR pairs: tasks with shared visual structure
#     e.g. T0↔T4 (plane/auto ↔ ship/truck) — all vehicles
#          T1↔T2 (bird/cat  ↔ deer/dog)    — all animals
#
#   Group B — DISSIMILAR pairs: tasks with different visual structure
#     e.g. T0↔T3 (plane/auto ↔ frog/horse) — vehicles vs animals
#          T1↔T3 (bird/cat  ↔ frog/horse)  — different animal types
#          T2↔T3 (deer/dog  ↔ frog/horse)  — different animal types
#
#   ρ_pre defines similarity — pairs above median ρ = similar group,
#   pairs below median ρ = dissimilar group.
#
# Test:
#   Do similar pairs (high ρ_pre) coexist (low forgetting)?
#   Do dissimilar pairs (low ρ_pre) forget (high forgetting)?
#   Is the group difference statistically significant?
#
# Key metric:
#   Coexistence = forgetting < threshold (e.g. 0.20)
#   AUC of ρ_pre as binary classifier of coexistence vs forgetting
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
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import json

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS             = 20
BATCH              = 128
LR                 = 1e-3
SEED               = 42
COEXISTENCE_THRESH = 0.20   # forgetting below this = coexistence

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


def rho_pre(model_A, tA, XBte, yBte, device):
    """Pre-hoc similarity: Task A model on Task B data."""
    eB   = error(model_A, tA, XBte, yBte, device)
    eBsf = shuffled_error(model_A, tA, XBte, yBte, device)
    return float(1.0 - np.sqrt(eB / (eBsf + 1e-8)))


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
    total     = len(all_pairs)
    results   = []

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{total} pairs\n")
    print(f"{'#':>3}  {'Pair':<12}  {'ρ_pre':>7}  "
          f"{'forget':>8}  {'coexist':>9}")
    print("-" * 48)

    for idx, (tA, tB) in enumerate(all_pairs, 1):
        XAtr, yAtr = task_data[tA]['train']
        XAte, yAte = task_data[tA]['test']
        XBtr, yBtr = task_data[tB]['train']
        XBte, yBte = task_data[tB]['test']

        mA = Model().to(device)
        train(mA, tA, XAtr, yAtr, device)
        R_AA = 1 - error(mA, tA, XAte, yAte, device)

        rho = rho_pre(mA, tA, XBte, yBte, device)

        train(mA, tB, XBtr, yBtr, device)
        R_BA       = 1 - error(mA, tA, XAte, yAte, device)
        forgetting = R_AA - R_BA
        coexists   = forgetting < COEXISTENCE_THRESH

        results.append({
            'label':      f"T{tA}→T{tB}",
            'tA': tA,     'tB': tB,
            'rho_pre':    rho,
            'R_AA':       R_AA,
            'R_BA':       R_BA,
            'forgetting': forgetting,
            'coexists':   coexists,
        })

        print(f"[{idx:>2}/{total}]  "
              f"T{tA}→T{tB} ({TASK_NAMES[tA][:6]}→{TASK_NAMES[tB][:6]})  "
              f"ρ_pre={rho:+.3f}  forget={forgetting:.3f}  "
              f"{'✓ coexist' if coexists else '✗ forget'}")

        with open('e3_checkpoint.json', 'w') as f:
            json.dump(results, f)

    return results


# ── Analyze ───────────────────────────────────────────────────────────────────
def analyze(results):
    rhos    = np.array([r['rho_pre']    for r in results])
    fgts    = np.array([r['forgetting'] for r in results])
    coexist = np.array([r['coexists']   for r in results])

    n_coexist = coexist.sum()
    n_forget  = (~coexist).sum()

    # Split into similar / dissimilar by median ρ_pre
    median_rho     = np.median(rhos)
    similar_mask   = rhos >= median_rho
    dissimilar_mask = ~similar_mask

    fgt_similar    = fgts[similar_mask]
    fgt_dissimilar = fgts[dissimilar_mask]

    # Mann-Whitney U test: do similar pairs forget less?
    stat, p_mw = mannwhitneyu(fgt_similar, fgt_dissimilar,
                               alternative='less')

    # AUC: ρ_pre as binary classifier of coexistence
    # coexists=1 (low forgetting) should correspond to high ρ_pre
    if len(np.unique(coexist)) > 1:
        auc = roc_auc_score(coexist.astype(int), rhos)
    else:
        auc = float('nan')

    print(f"\n{'='*58}")
    print(f"  Coexistence threshold : {COEXISTENCE_THRESH:.2f}")
    print(f"  Coexisting pairs      : {n_coexist}/{len(results)}")
    print(f"  Forgetting pairs      : {n_forget}/{len(results)}")
    print()
    print(f"  Similar pairs (ρ ≥ {median_rho:.3f}):")
    print(f"    Mean forgetting : {fgt_similar.mean():.3f} ± "
          f"{fgt_similar.std():.3f}")
    print(f"    Coexistence rate: "
          f"{coexist[similar_mask].mean()*100:.0f}%")
    print()
    print(f"  Dissimilar pairs (ρ < {median_rho:.3f}):")
    print(f"    Mean forgetting : {fgt_dissimilar.mean():.3f} ± "
          f"{fgt_dissimilar.std():.3f}")
    print(f"    Coexistence rate: "
          f"{coexist[dissimilar_mask].mean()*100:.0f}%")
    print()
    print(f"  Mann-Whitney U (similar < dissimilar): p={p_mw:.4f}")
    print(f"  AUC (ρ_pre → coexistence): {auc:.3f}")

    if p_mw < 0.05 and auc > 0.7:
        verdict = "✅  Theory HOLDS — similar tasks coexist, dissimilar forget"
    elif p_mw < 0.05 or auc > 0.7:
        verdict = "⚠️  Partial support"
    else:
        verdict = "❌  Not supported"
    print(f"  Verdict : {verdict}")
    print(f"{'='*58}")

    return auc, p_mw, fgt_similar, fgt_dissimilar


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(results, auc, p_mw, fgt_similar, fgt_dissimilar):
    rhos    = np.array([r['rho_pre']    for r in results])
    fgts    = np.array([r['forgetting'] for r in results])
    coexist = np.array([r['coexists']   for r in results])
    lbls    = [r['label'] for r in results]

    median_rho = np.median(rhos)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: scatter ρ vs forgetting, coloured by coexistence outcome
    ax = axes[0]
    colors = ['green' if c else 'red' for c in coexist]
    ax.scatter(rhos, fgts, s=80, c=colors,
               edgecolors='white', linewidths=0.6, zorder=3)
    for lbl, rv, fv in zip(lbls, rhos, fgts):
        ax.annotate(lbl, (rv, fv), fontsize=6.5, alpha=0.75,
                    xytext=(4, 3), textcoords='offset points')
    ax.axhline(COEXISTENCE_THRESH, color='gray', lw=1.2, ls='--',
               label=f'coexistence threshold={COEXISTENCE_THRESH}')
    ax.axvline(median_rho, color='purple', lw=1, ls=':',
               label=f'median ρ={median_rho:.3f}')
    m, b = np.polyfit(rhos, fgts, 1)
    xs = np.linspace(rhos.min()*0.95, rhos.max()*1.05, 100)
    r_val, p_val = pearsonr(rhos, fgts)
    ax.plot(xs, m*xs+b, 'k--', lw=1.2, alpha=0.5,
            label=f'r={r_val:.3f} (p={p_val:.4f})')
    ax.set_xlabel('ρ_pre  (higher = more similar)', fontsize=11)
    ax.set_ylabel('Forgetting  =  R_AA − R_BA', fontsize=11)
    ax.set_title('E3: Similarity vs Forgetting\n'
                 'green=coexist  red=forgetting',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Middle: box plot similar vs dissimilar forgetting
    ax2 = axes[1]
    bp = ax2.boxplot([fgt_similar, fgt_dissimilar],
                     labels=['Similar\n(ρ ≥ median)', 'Dissimilar\n(ρ < median)'],
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='navy', lw=2))
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax2.axhline(COEXISTENCE_THRESH, color='gray', lw=1.2, ls='--',
                label=f'threshold={COEXISTENCE_THRESH}')
    ax2.set_ylabel('Forgetting', fontsize=11)
    ax2.set_title(f'Similar vs Dissimilar pairs\n'
                  f'Mann-Whitney p={p_mw:.4f}',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    # Right: bar chart sorted by ρ_pre, coloured by coexistence
    ax3 = axes[2]
    order  = np.argsort(rhos)
    cols   = ['green' if coexist[i] else 'red' for i in order]
    ax3.bar(range(len(results)), fgts[order],
            color=cols, edgecolor='white', linewidth=0.4, alpha=0.8)
    ax3.axhline(COEXISTENCE_THRESH, color='gray', lw=1.2, ls='--',
                label=f'threshold={COEXISTENCE_THRESH}')
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([lbls[i] for i in order],
                        rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Forgetting', fontsize=11)
    ax3.set_title(f'Pairs sorted by ρ_pre\n'
                  f'AUC={auc:.3f}  '
                  f'green=coexist  red=forget',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('e3_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e3_results.png")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    results = run(device)
    auc, p_mw, fgt_sim, fgt_dis = analyze(results)
    plot(results, auc, p_mw, fgt_sim, fgt_dis)