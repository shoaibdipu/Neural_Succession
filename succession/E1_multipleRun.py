# ============================================================
# Similarity vs Forgetting — v11
# Similarity: Li & Hiratani (2025) Eq. 16
# Forgetting: R_AA - R_BA  (positive, higher = worse)
#
# Supports: cifar10, mnist
# N_REPEATS: each pair run multiple times with different seeds
#            → per-pair mean ± std in plots
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
DATASET   = 'cifar10'   # 'cifar10' or 'mnist'
N_REPEATS = 3           # runs per pair  (1 = single run, no error bars)
EPOCHS    = 20          # fixed epochs per task — no early stopping
BATCH     = 128
LR        = 1e-3
SEED      = 42

# ── Dataset configs ───────────────────────────────────────────────────────────
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

CONFIGS = {
    'cifar10': {
        'task_splits': {
            0: (0, 1),   # airplane vs automobile
            1: (2, 3),   # bird vs cat
            2: (4, 5),   # deer vs dog
            3: (6, 7),   # frog vs horse
            4: (8, 9),   # ship vs truck
        },
        'task_names': {
            0: 'plane/auto', 1: 'bird/cat',   2: 'deer/dog',
            3: 'frog/horse', 4: 'ship/truck',
        },
        'head_dim': 512,
    },
    'mnist': {
        'task_splits': {
            0: (0, 1), 1: (2, 3), 2: (4, 5), 3: (6, 7), 4: (8, 9),
        },
        'task_names': {
            0: '0/1', 1: '2/3', 2: '4/5', 3: '6/7', 4: '8/9',
        },
        'head_dim': 400,
    },
}


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    if DATASET == 'cifar10':
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
    else:
        ds = load_dataset("ylecun/mnist")
        def extract(split):
            imgs = np.array(
                [np.array(x).flatten() / 255.0 for x in ds[split]['image']],
                dtype=np.float32)
            labs = np.array(ds[split]['label'], dtype=np.int64)
            return imgs, labs
        Xtr, ytr = extract('train')
        Xte, yte = extract('test')

    print(f"{DATASET.upper()} — Train {Xtr.shape}  Test {Xte.shape}")
    return Xtr, ytr, Xte, yte


def task_subset(X, y, pair):
    c0, c1 = pair
    m = (y == c0) | (y == c1)
    return X[m], (y[m] == c1).astype(np.int64)


# ── Models ────────────────────────────────────────────────────────────────────
class CIFARModel(nn.Module):
    """ResNet-18 backbone + per-task Linear(512, 2) heads."""
    def __init__(self, n_tasks=5):
        super().__init__()
        bb    = resnet18(weights=None)
        bb.fc = nn.Identity()
        self.bb    = bb
        self.heads = nn.ModuleList([nn.Linear(512, 2) for _ in range(n_tasks)])

    def forward(self, x, t):
        return self.heads[t](self.bb(x))


class MNISTModel(nn.Module):
    """MLP 784→400→400 backbone + per-task Linear(400, 2) heads."""
    def __init__(self, n_tasks=5):
        super().__init__()
        self.h1    = nn.Linear(784, 400)
        self.h2    = nn.Linear(400, 400)
        self.relu  = nn.ReLU()
        self.heads = nn.ModuleList([nn.Linear(400, 2) for _ in range(n_tasks)])

    def forward(self, x, t):
        return self.heads[t](self.relu(self.h2(self.relu(self.h1(x)))))


def make_model(n_tasks=5):
    return CIFARModel(n_tasks) if DATASET == 'cifar10' else MNISTModel(n_tasks)


# ── Train / Evaluate ──────────────────────────────────────────────────────────
def train(model, task_id, X, y, device):
    """Fixed EPOCHS training. Only current task head + backbone update."""
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i == task_id)
    backbone = (model.bb.parameters() if hasattr(model, 'bb')
                else list(model.h1.parameters()) + list(model.h2.parameters()))
    for p in backbone:
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
    """Classification error = 1 - accuracy."""
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
    """Chance-level error: average over n label-shuffled evaluations."""
    errs = []
    for _ in range(n):
        y_s = y.copy()
        np.random.shuffle(y_s)
        errs.append(error(model, task_id, X, y_s, device))
    return float(np.mean(errs))


# ── Li & Hiratani (2025) Eq. 16 ──────────────────────────────────────────────
def similarity(mA, mB, tA, tB, XAte, yAte, XBte, yBte, device):
    """
    ρ_AB = 1 - (1/2) * ( sqrt(ε_B[W_A] / ε_{B,sf}[W_A])
                        + sqrt(ε_A[W_B] / ε_{A,sf}[W_B]) )

    Symmetric bidirectional zero-shot transfer similarity.
    Higher ρ = more similar tasks = less expected forgetting.
    """
    eB_WA  = error(mA, tA, XBte, yBte, device)
    eBs_WA = shuffled_error(mA, tA, XBte, yBte, device)
    eA_WB  = error(mB, tB, XAte, yAte, device)
    eAs_WB = shuffled_error(mB, tB, XAte, yAte, device)

    t1  = np.sqrt(eB_WA  / (eBs_WA  + 1e-8))
    t2  = np.sqrt(eA_WB  / (eAs_WB  + 1e-8))
    rho = 1.0 - 0.5 * (t1 + t2)
    return float(rho)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(device):
    cfg        = CONFIGS[DATASET]
    task_splits = cfg['task_splits']
    task_names  = cfg['task_names']

    Xtr, ytr, Xte, yte = load_data()
    task_data = {
        t: {'train': task_subset(Xtr, ytr, p),
            'test':  task_subset(Xte, yte, p)}
        for t, p in task_splits.items()
    }

    all_pairs = [(tA, tB) for tA in range(5)
                          for tB in range(5) if tA != tB]
    total     = len(all_pairs) * N_REPEATS
    results   = []
    idx       = 0

    print(f"\n{len(all_pairs)} pairs × {N_REPEATS} repeats = {total} runs\n")
    print(f"{'#':>4}  {'rep':>3}  {'Pair':<12}  "
          f"{'ρ_AB':>6}  {'R_AA':>6}  {'R_BA':>6}  {'forget':>8}")
    print("-" * 55)

    for rep in range(N_REPEATS):
        torch.manual_seed(SEED + rep)
        np.random.seed(SEED + rep)

        for tA, tB in all_pairs:
            idx += 1
            XAtr, yAtr = task_data[tA]['train']
            XAte, yAte = task_data[tA]['test']
            XBtr, yBtr = task_data[tB]['train']
            XBte, yBte = task_data[tB]['test']

            # Train Task A
            mA = make_model().to(device)
            train(mA, tA, XAtr, yAtr, device)
            R_AA = 1 - error(mA, tA, XAte, yAte, device)

            # Train Task B separately (for similarity only)
            mB = make_model().to(device)
            train(mB, tB, XBtr, yBtr, device)

            # Li & Hiratani Eq. 16
            rho = similarity(mA, mB, tA, tB, XAte, yAte, XBte, yBte, device)

            # Train Task B on Task A model → measure forgetting
            train(mA, tB, XBtr, yBtr, device)
            R_BA       = 1 - error(mA, tA, XAte, yAte, device)
            forgetting = R_AA - R_BA

            results.append({
                'label':      f"T{tA}→T{tB}",
                'tA': tA,     'tB': tB,
                'rep':        rep,
                'rho':        rho,
                'R_AA':       R_AA,
                'R_BA':       R_BA,
                'forgetting': forgetting,
            })

            print(f"[{idx:>3}/{total}]  r{rep+1}  "
                  f"T{tA}→T{tB} "
                  f"({task_names[tA][:6]}→{task_names[tB][:6]})  "
                  f"ρ={rho:+.3f}  "
                  f"R_AA={R_AA:.3f}  R_BA={R_BA:.3f}  "
                  f"forget={forgetting:.3f}")

            # checkpoint after every run
            with open('e1_checkpoint.json', 'w') as f:
                json.dump(results, f)

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(results, dataset=None):
    ds_label = (dataset or DATASET).upper()

    # Per-pair means and stds
    labels = sorted(set(r['label'] for r in results))
    rho_m, fgt_m, rho_s, fgt_s = [], [], [], []
    for lbl in labels:
        sub = [r for r in results if r['label'] == lbl]
        rho_m.append(np.mean([r['rho']        for r in sub]))
        fgt_m.append(np.mean([r['forgetting'] for r in sub]))
        rho_s.append(np.std( [r['rho']        for r in sub]))
        fgt_s.append(np.std( [r['forgetting'] for r in sub]))

    rho_m = np.array(rho_m)
    fgt_m = np.array(fgt_m)
    rho_s = np.array(rho_s)
    fgt_s = np.array(fgt_s)

    r_val,  p_val  = pearsonr(rho_m, fgt_m)
    rho_sp, _      = spearmanr(rho_m, fgt_m)

    # Also report on all individual runs
    r_all, p_all = pearsonr([r['rho']        for r in results],
                            [r['forgetting'] for r in results])

    print(f"\n{'='*52}")
    print(f"  Dataset        : {ds_label}")
    print(f"  Pairs          : {len(labels)}  ×  {N_REPEATS} repeats")
    print(f"  Pearson r (means)  : {r_val:.3f}  (p={p_val:.4f})")
    print(f"  Spearman ρ (means) : {rho_sp:.3f}")
    print(f"  Pearson r (all)    : {r_all:.3f}  (p={p_all:.4f})")
    print(f"  R²                 : {r_val**2:.3f}  "
          f"({r_val**2*100:.1f}% variance explained)")

    if r_val < -0.5 and p_val < 0.05:
        verdict = "✅  HOLDS — similar tasks forget less"
    elif r_val < -0.3:
        verdict = "⚠️  Partial support"
    else:
        verdict = "❌  Not supported"
    print(f"  Verdict            : {verdict}")
    print(f"{'='*52}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter with error bars
    ax = axes[0]
    # individual runs (faint)
    if N_REPEATS > 1:
        ax.scatter([r['rho']        for r in results],
                   [r['forgetting'] for r in results],
                   s=15, color='steelblue', alpha=0.25, zorder=2)
    # per-pair means with error bars
    ax.errorbar(rho_m, fgt_m,
                xerr=rho_s if N_REPEATS > 1 else None,
                yerr=fgt_s if N_REPEATS > 1 else None,
                fmt='o', color='crimson', markersize=7,
                capsize=3, linewidth=1.2, zorder=4,
                label='pair mean' + (' ± std' if N_REPEATS > 1 else ''))
    for lbl, rv, fv in zip(labels, rho_m, fgt_m):
        ax.annotate(lbl, (rv, fv), fontsize=7, alpha=0.8,
                    xytext=(4, 3), textcoords='offset points')
    xs   = np.linspace(rho_m.min()*0.95, rho_m.max()*1.05, 100)
    m, b = np.polyfit(rho_m, fgt_m, 1)
    ax.plot(xs, m*xs+b, 'r--', lw=1.5,
            label=f'r={r_val:.3f} (p={p_val:.4f})\nρ_s={rho_sp:.3f}')
    ax.set_xlabel('ρ_AB  (Li & Hiratani Eq. 16, higher = more similar)',
                  fontsize=11)
    ax.set_ylabel('Forgetting  =  R_AA − R_BA', fontsize=11)
    ax.set_title(f'Task similarity vs forgetting\n'
                 f'({ds_label}, negative slope expected)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Right: bar chart sorted by ρ
    ax2   = axes[1]
    order = np.argsort(rho_m)
    norm  = (rho_m[order] - rho_m.min()) / (rho_m.max() - rho_m.min() + 1e-8)
    cols  = plt.cm.RdYlGn(norm)
    ax2.bar(range(len(labels)), fgt_m[order],
            color=cols, edgecolor='white', linewidth=0.4)
    if N_REPEATS > 1:
        ax2.errorbar(range(len(labels)), fgt_m[order], yerr=fgt_s[order],
                     fmt='none', color='black', capsize=3, linewidth=1)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([labels[i] for i in order],
                        rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Forgetting' + (' mean ± std' if N_REPEATS > 1 else ''),
                   fontsize=11)
    ax2.set_title(f'Pairs sorted by ρ_AB\n'
                  f'red=dissimilar  green=similar\n'
                  f'Theory: bars fall left→right',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f'e1_{DATASET}_r{N_REPEATS}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device   : {device}")
    print(f"Dataset  : {DATASET}")
    print(f"Repeats  : {N_REPEATS}")
    print(f"Epochs   : {EPOCHS}\n")
    results = run(device)
    plot(results)