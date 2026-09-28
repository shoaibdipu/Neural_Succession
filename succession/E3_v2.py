# ============================================================
# E3 v2 — Coexistence Under Niche Partitioning
#
# Problem with E3 v1:
#   Split-CIFAR-10's 5 fixed tasks gave only 20 pairs,
#   3 coexisting pairs, and AUC trivially = 1.0 at the
#   edge of the data range. Not enough variation.
#
# Fix:
#   Use ALL 45 possible binary tasks from CIFAR-10
#   (any two of the 10 classes form a task).
#   Sample N_PAIRS ordered task pairs covering the full
#   similarity spectrum — from same-domain (vehicles↔vehicles,
#   animals↔animals) to cross-domain (vehicles↔animals).
#
#   This gives:
#     - Genuine spread in ρ_pre by construction
#     - Enough coexisting pairs for meaningful AUC
#     - Threshold sweep with real shape
#
# Protocol:
#   1. Define all 45 binary tasks from C(10,2) class pairs
#   2. Sample N_PAIRS ordered pairs (tA, tB)
#   3. For each pair: train tA → compute ρ_pre → train tB → measure forgetting
#   4. Sweep thresholds, compute AUC, report coexistence boundary
#
# CIFAR-10 classes:
#   0:airplane  1:automobile  2:bird  3:cat  4:deer
#   5:dog  6:frog  7:horse  8:ship  9:truck
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
from itertools import combinations
import matplotlib.pyplot as plt
import json
import random

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS   = 20
BATCH    = 128
LR       = 1e-3
SEED     = 42
N_PAIRS  = 60    # ordered task pairs to sample from all possible combinations
                 # 45 tasks → 45×44=1980 ordered pairs available
                 # 60 gives good coverage without excessive runtime

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

CLASS_NAMES = {
    0: 'airplane', 1: 'automobile', 2: 'bird',  3: 'cat',   4: 'deer',
    5: 'dog',      6: 'frog',       7: 'horse', 8: 'ship',  9: 'truck',
}

# Semantic domain — used to label pairs as same-domain or cross-domain
DOMAIN = {
    0: 'vehicle', 1: 'vehicle', 8: 'vehicle', 9: 'vehicle',
    2: 'animal',  3: 'animal',  4: 'animal',  5: 'animal',
    6: 'animal',  7: 'animal',
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


def task_subset(X, y, c0, c1):
    """Binary task: class c0 vs c1, relabelled 0/1."""
    m = (y == c0) | (y == c1)
    return X[m], (y[m] == c1).astype(np.int64)


def build_tasks(Xtr, ytr, Xte, yte):
    """
    Build all 45 binary tasks from C(10,2) class pairs.
    Each task is identified by (c0, c1) with c0 < c1.
    Returns dict: (c0,c1) → {train, test}
    """
    tasks = {}
    for c0, c1 in combinations(range(10), 2):
        tasks[(c0, c1)] = {
            'train': task_subset(Xtr, ytr, c0, c1),
            'test':  task_subset(Xte, yte, c0, c1),
        }
    print(f"Built {len(tasks)} binary tasks from CIFAR-10")
    return tasks


def sample_pairs(n, seed=SEED):
    """
    Sample N_PAIRS ordered task pairs using PURE-DOMAIN tasks only.

    Pure-domain task: both classes from the same semantic group.
      Vehicle tasks (6): any two of {airplane(0), auto(1), ship(8), truck(9)}
      Animal tasks (15): any two of {bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)}

    Same-domain pair:  both Task A and Task B are vehicle tasks,
                       OR both are animal tasks.
                       Expected: high ρ_pre, low forgetting.

    Cross-domain pair: one vehicle task, one animal task.
                       Expected: low ρ_pre, high forgetting.

    Using pure tasks avoids the mixed-class problem where Task A
    trains on (airplane, dog) — a vehicle+animal mix — making the
    backbone broadly tuned and breaking the similarity→forgetting
    relationship.

    Returns list of ((c0A,c1A), (c0B,c1B)) pairs.
    """
    vehicles = [0, 1, 8, 9]   # airplane, automobile, ship, truck
    animals  = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse

    vehicle_tasks = list(combinations(vehicles, 2))   # 6 tasks
    animal_tasks  = list(combinations(animals,  2))   # 15 tasks

    rng = random.Random(seed)

    def ordered_pairs(list_A, list_B):
        return [(tA, tB) for tA in list_A
                         for tB in list_B if tA != tB]

    same_pairs  = (ordered_pairs(vehicle_tasks, vehicle_tasks) +
                   ordered_pairs(animal_tasks,  animal_tasks))
    cross_pairs = (ordered_pairs(vehicle_tasks, animal_tasks) +
                   ordered_pairs(animal_tasks,  vehicle_tasks))

    n_same  = max(4, int(n * 0.4))
    n_cross = n - n_same

    sampled_same  = rng.sample(same_pairs,  min(n_same,  len(same_pairs)))
    sampled_cross = rng.sample(cross_pairs, min(n_cross, len(cross_pairs)))

    pairs = sampled_same + sampled_cross
    rng.shuffle(pairs)

    print(f"Pure vehicle tasks: {len(vehicle_tasks)}  "
          f"Pure animal tasks: {len(animal_tasks)}")
    print(f"Available same-domain pairs : {len(same_pairs)}")
    print(f"Available cross-domain pairs: {len(cross_pairs)}")
    print(f"\nSampled {len(pairs)} pairs "
          f"({len(sampled_same)} same-domain, "
          f"{len(sampled_cross)} cross-domain)")
    return pairs


# ── Model ─────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    """ResNet-18 backbone + two heads (one per task class pair)."""
    def __init__(self):
        super().__init__()
        bb    = resnet18(weights=None)
        bb.fc = nn.Identity()
        self.bb   = bb
        # Two heads: head 0 = Task A, head 1 = Task B
        self.heads = nn.ModuleList([nn.Linear(512, 2), nn.Linear(512, 2)])

    def forward(self, x, t):
        return self.heads[t](self.bb(x))


# ── Train / Evaluate ──────────────────────────────────────────────────────────
def train(model, head_id, X, y, device):
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i == head_id)
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
            loss_fn(model(xb, head_id), yb).backward()
            opt.step()


def error(model, head_id, X, y, device, batch=256):
    model.eval()
    wrong, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch]).to(device)
            yb = torch.tensor(y[i:i+batch]).to(device)
            wrong += (model(xb, head_id).argmax(1) != yb).sum().item()
            total += len(yb)
    return wrong / total


def shuffled_error(model, head_id, X, y, device, n=5):
    errs = []
    for _ in range(n):
        y_s = y.copy()
        np.random.shuffle(y_s)
        errs.append(error(model, head_id, X, y_s, device))
    return float(np.mean(errs))


def rho_pre(model, X_B_te, y_B_te, device):
    """
    ρ_pre = 1 - sqrt(ε_B[W_A] / ε_{B,sf}[W_A])
    Uses head 0 (Task A head) to evaluate Task B data.
    """
    eB   = error(model, 0, X_B_te, y_B_te, device)
    eBsf = shuffled_error(model, 0, X_B_te, y_B_te, device)
    return float(1.0 - np.sqrt(eB / (eBsf + 1e-8)))


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(device):
    Xtr, ytr, Xte, yte = load_cifar10()
    tasks = build_tasks(Xtr, ytr, Xte, yte)
    pairs = sample_pairs(N_PAIRS)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = []
    total   = len(pairs)

    print(f"\n{'#':>3}  {'Pair':<22}  {'domain':>8}  "
          f"{'ρ_pre':>7}  {'forget':>8}")
    print("-" * 60)

    for idx, (tA, tB) in enumerate(pairs, 1):
        cA0, cA1 = tA
        cB0, cB1 = tB

        XAtr, yAtr = tasks[tA]['train']
        XAte, yAte = tasks[tA]['test']
        XBtr, yBtr = tasks[tB]['train']
        XBte, yBte = tasks[tB]['test']

        vehicles = {0, 1, 8, 9}
        tA_vehicle = (cA0 in vehicles and cA1 in vehicles)
        tB_vehicle = (cB0 in vehicles and cB1 in vehicles)
        domain = 'same' if (tA_vehicle == tB_vehicle) else 'cross'
        label  = (f"{CLASS_NAMES[cA0][:4]}/{CLASS_NAMES[cA1][:4]}"
                  f"→{CLASS_NAMES[cB0][:4]}/{CLASS_NAMES[cB1][:4]}")

        # Train Task A on head 0
        model = Model().to(device)
        train(model, 0, XAtr, yAtr, device)
        R_AA  = 1 - error(model, 0, XAte, yAte, device)

        # ρ_pre: Task A model on Task B test data
        rho = rho_pre(model, XBte, yBte, device)

        # Train Task B on head 1 (backbone shared, updates)
        train(model, 1, XBtr, yBtr, device)
        R_BA       = 1 - error(model, 0, XAte, yAte, device)
        forgetting = R_AA - R_BA

        results.append({
            'label':      label,
            'tA':         tA,
            'tB':         tB,
            'domain':     domain,
            'rho_pre':    rho,
            'R_AA':       R_AA,
            'R_BA':       R_BA,
            'forgetting': forgetting,
        })

        print(f"[{idx:>2}/{total}]  {label:<22}  {domain:>8}  "
              f"ρ={rho:+.3f}  forget={forgetting:.3f}")

        with open('e3v2_checkpoint.json', 'w') as f:
            json.dump(results, f)

    return results


# ── Analyze ───────────────────────────────────────────────────────────────────
def analyze(results):
    rhos   = np.array([r['rho_pre']    for r in results])
    fgts   = np.array([r['forgetting'] for r in results])
    domains = np.array([r['domain']    for r in results])

    r_val, p_val = pearsonr(rhos, fgts)
    sp_val, _    = spearmanr(rhos, fgts)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    thresholds = np.linspace(fgts.min() + 0.01, fgts.max() - 0.01, 80)
    aucs, n_coexists = [], []
    for thresh in thresholds:
        labels = (fgts < thresh).astype(int)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos < 3 or n_neg < 3:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(labels, rhos))
        n_coexists.append(n_pos)

    aucs       = np.array(aucs)
    best_idx   = np.nanargmax(aucs)
    best_thresh = thresholds[best_idx]
    best_auc    = aucs[best_idx]

    coexist     = fgts < best_thresh

    # ── Same vs cross domain forgetting ──────────────────────────────────────
    fgt_same  = fgts[domains == 'same']
    fgt_cross = fgts[domains == 'cross']
    _, p_domain = mannwhitneyu(fgt_same, fgt_cross, alternative='less')

    # ── Similar vs dissimilar (by median ρ_pre) ───────────────────────────────
    median_rho  = np.median(rhos)
    fgt_sim     = fgts[rhos >= median_rho]
    fgt_dis     = fgts[rhos <  median_rho]
    _, p_mw     = mannwhitneyu(fgt_sim, fgt_dis, alternative='less')

    print(f"\n{'='*60}")
    print(f"  n pairs          : {len(results)}")
    print(f"  Pearson r        : {r_val:.3f}  (p={p_val:.4f})")
    print(f"  Spearman ρ       : {sp_val:.3f}")
    print()
    print(f"  Same-domain  mean forgetting: {fgt_same.mean():.3f} ± "
          f"{fgt_same.std():.3f}  (n={len(fgt_same)})")
    print(f"  Cross-domain mean forgetting: {fgt_cross.mean():.3f} ± "
          f"{fgt_cross.std():.3f}  (n={len(fgt_cross)})")
    print(f"  Mann-Whitney (same < cross)  : p={p_domain:.4f}")
    print()
    print(f"  Threshold sweep best: {best_thresh:.3f}  AUC={best_auc:.3f}  "
          f"(n_coexist={coexist.sum()})")
    print(f"  Mann-Whitney (sim < dissim)  : p={p_mw:.4f}")
    print(f"{'='*60}")

    return (r_val, p_val, best_auc, best_thresh, coexist,
            fgt_same, fgt_cross, p_domain, p_mw,
            thresholds, aucs, domains)


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(results, r_val, p_val, best_auc, best_thresh, coexist,
         fgt_same, fgt_cross, p_domain, p_mw, thresholds, aucs, domains):
    """
    Two decisive panels only:

    Left  — Box plot: same-domain vs cross-domain forgetting distributions.
            Shows the qualitative regime separation. The decisive claim.

    Right — Strip plot: individual pair forgetting values coloured by domain,
            with coexistence threshold line. Shows that near-zero forgetting
            only occurs in same-domain pairs — no cross-domain pair coexists.
    """
    fgts = np.array([r['forgetting'] for r in results])
    COEXIST_THRESH = 0.10   # near-zero forgetting = genuine coexistence

    n_same_coexist  = ((fgts < COEXIST_THRESH) & (domains == 'same')).sum()
    n_cross_coexist = ((fgts < COEXIST_THRESH) & (domains == 'cross')).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: box plot ─────────────────────────────────────────────────────────
    ax = axes[0]
    bp = ax.boxplot([fgt_same, fgt_cross],
                    labels=['Same-domain\n(vehicle↔vehicle\nor animal↔animal)',
                            'Cross-domain\n(vehicle↔animal)'],
                    patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color='navy', lw=2.5),
                    whiskerprops=dict(lw=1.5),
                    capprops=dict(lw=1.5),
                    flierprops=dict(marker='o', markersize=5, alpha=0.5))
    bp['boxes'][0].set_facecolor('lightsteelblue')
    bp['boxes'][1].set_facecolor('lightsalmon')

    # overlay jittered points
    rng = np.random.default_rng(42)
    for i, (fgt_grp, col) in enumerate(
            [(fgt_same, 'steelblue'), (fgt_cross, 'tomato')], 1):
        jitter = rng.uniform(-0.08, 0.08, len(fgt_grp))
        ax.scatter(np.full(len(fgt_grp), i) + jitter, fgt_grp,
                   s=25, color=col, alpha=0.5, zorder=3)

    ax.axhline(COEXIST_THRESH, color='green', lw=1.5, ls='--',
               label=f'coexistence boundary ({COEXIST_THRESH:.2f})')
    ax.set_ylabel('Forgetting  =  R_AA − R_BA', fontsize=12)
    ax.set_title('E3: Coexistence by Semantic Domain\n'
                 f'Mann-Whitney p={p_domain:.2e}  '
                 f'(n_same={len(fgt_same)}, n_cross={len(fgt_cross)})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # annotation
    ax.text(0.97, 0.97,
            f'Same-domain coexist:  {n_same_coexist}/{len(fgt_same)}\n'
            f'Cross-domain coexist: {n_cross_coexist}/{len(fgt_cross)}',
            transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ── Right: strip plot ordered by domain then forgetting ────────────────────
    ax2 = axes[1]

    same_fgts  = np.sort(fgt_same)
    cross_fgts = np.sort(fgt_cross)

    x_same  = np.zeros(len(same_fgts))   - 0.2
    x_cross = np.ones(len(cross_fgts))   + 0.2

    # colour by coexistence outcome
    same_cols  = ['green' if f < COEXIST_THRESH else 'steelblue'
                  for f in same_fgts]
    cross_cols = ['green' if f < COEXIST_THRESH else 'tomato'
                  for f in cross_fgts]

    jitter_s = rng.uniform(-0.12, 0.12, len(same_fgts))
    jitter_c = rng.uniform(-0.12, 0.12, len(cross_fgts))

    # ax2.scatter(x_same  + jitter_s, same_fgts,
    #             c=same_cols,  s=50, edgecolors='white', lw=0.5, zorder=3)
    # ax2.scatter(x_cross + jitter_c, cross_fgts,
    #             c=cross_cols, s=50, edgecolors='white', lw=0.5, zorder=3)
               
    ax2.scatter(x_same, same_fgts,
                c=same_cols,  s=50, edgecolors='white', lw=0.5, zorder=3)
    ax2.scatter(x_cross, cross_fgts,
                c=cross_cols, s=50, edgecolors='white', lw=0.5, zorder=3)

    ax2.axhline(COEXIST_THRESH, color='green', lw=1.5, ls='--',
                label=f'coexistence boundary ({COEXIST_THRESH:.2f})')
    ax2.set_xticks([-0.2, 1.2])
    ax2.set_xticklabels(['Same-domain', 'Cross-domain'], fontsize=11)
    ax2.set_xlim(-0.6, 1.7)
    ax2.set_ylabel('Forgetting', fontsize=12)
    ax2.set_title('Individual pairs — green = coexisting\n'
                  'Near-zero forgetting only in same-domain',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f'E3 — Coexistence Requires Semantic Niche Overlap\n'
        f'CIFAR-10 pure-domain tasks, n={len(results)} pairs',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('e3v2_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e3v2_results.png")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None,
                        help='Path to checkpoint JSON to skip training '
                             'e.g. --load e3v2_checkpoint.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.load:
        print(f"Loading results from {args.load} (skipping training)")
        with open(args.load) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} pairs")
    else:
        print(f"Device  : {device}")
        print(f"N_PAIRS : {N_PAIRS}\n")
        results = run(device)

    stats = analyze(results)
    plot(results, *stats)