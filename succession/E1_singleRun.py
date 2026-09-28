# ============================================================
# Similarity vs Forgetting — CIFAR-10
# Similarity: Li & Hiratani (2025) Eq. 16
# Forgetting:  accuracy drop on Task A after training Task B
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

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS = 20
BATCH  = 128
LR     = 1e-3
SEED   = 42

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

# Split-CIFAR-10: 5 binary tasks, original CIFAR-10 splits
TASK_SPLITS = {
    0: (0, 1),   # airplane vs automobile
    1: (2, 3),   # bird vs cat
    2: (4, 5),   # deer vs dog
    3: (6, 7),   # frog vs horse
    4: (8, 9),   # ship vs truck
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


# ── Model: shared ResNet-18 backbone + per-task head ─────────────────────────
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
    # freeze all heads except current task
    for i, h in enumerate(model.heads):
        for p in h.parameters():
            p.requires_grad = (i == task_id)
    for p in model.bb.parameters():
        p.requires_grad = True

    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                        batch_size=BATCH, shuffle=True)
    opt  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
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
    """Classification error (1 - accuracy). Used in Li & Hiratani Eq. 16."""
    model.eval()
    wrong, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch]).to(device)
            yb = torch.tensor(y[i:i+batch]).to(device)
            wrong += (model(xb, task_id).argmax(1) != yb).sum().item()
            total += len(yb)
    return wrong / total


def shuffled_error(model, task_id, X, y, device, batch=256):
    """
    Error with label shuffling — chance-level error ε_{sf} in Eq. 16.
    Shuffle labels randomly to get chance performance.
    """
    y_shuf = y.copy()
    np.random.shuffle(y_shuf)
    return error(model, task_id, X, y_shuf, device, batch)


# ── Li & Hiratani (2025) Eq. 16 ──────────────────────────────────────────────
def rho_AB(model_A, model_B, tA, tB,
           X_A_te, y_A_te, X_B_te, y_B_te,
           device, n_shuffle=5):
    """
    ρ_AB = 1 - (1/2) * ( sqrt(ε_B[W_A] / ε_{B,sf}[W_A])
                        + sqrt(ε_A[W_B] / ε_{A,sf}[W_B]) )

    ε_B[W_A]    : error of Task A model on Task B test data
    ε_{B,sf}[W_A]: chance-level error (label shuffled) — averaged over
                   n_shuffle trials for stability
    Symmetric: uses both W_A→B and W_B→A directions.
    Square root because error scales as square of task correlation
    in linear model (Appendix C.4 of Li & Hiratani).
    """
    # ε_B[W_A] and ε_{B,sf}[W_A]
    eps_B_WA    = error(model_A, tA, X_B_te, y_B_te, device)
    eps_Bsf_WA  = np.mean([shuffled_error(model_A, tA, X_B_te, y_B_te, device)
                            for _ in range(n_shuffle)])

    # ε_A[W_B] and ε_{A,sf}[W_B]
    eps_A_WB    = error(model_B, tB, X_A_te, y_A_te, device)
    eps_Asf_WB  = np.mean([shuffled_error(model_B, tB, X_A_te, y_A_te, device)
                            for _ in range(n_shuffle)])

    term1 = np.sqrt(eps_B_WA  / (eps_Bsf_WA  + 1e-8))
    term2 = np.sqrt(eps_A_WB  / (eps_Asf_WB  + 1e-8))
    rho   = 1.0 - 0.5 * (term1 + term2)

    return float(rho), {
        'eps_B_WA': eps_B_WA,   'eps_Bsf_WA': eps_Bsf_WA,
        'eps_A_WB': eps_A_WB,   'eps_Asf_WB': eps_Asf_WB,
        'term1': term1,          'term2': term2,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(device):
    print("Loading CIFAR-10...")
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
    print(f"{'#':>3}  {'Pair':<12}  {'ρ_AB':>6}  "
          f"{'R_AA':>6}  {'R_BA':>6}  {'forget':>8}")
    print("-" * 48)

    for idx, (tA, tB) in enumerate(all_pairs, 1):
        XAtr, yAtr = task_data[tA]['train']
        XAte, yAte = task_data[tA]['test']
        XBtr, yBtr = task_data[tB]['train']
        XBte, yBte = task_data[tB]['test']

        # Train Task A model
        mA = Model().to(device)
        train(mA, tA, XAtr, yAtr, device)
        R_AA = 1 - error(mA, tA, XAte, yAte, device)

        # Train Task B model (separate — only for similarity computation)
        mB = Model().to(device)
        train(mB, tB, XBtr, yBtr, device)

        # Li & Hiratani Eq. 16
        rho, dbg = rho_AB(mA, mB, tA, tB,
                          XAte, yAte, XBte, yBte, device)

        # Train Task B on Task A's model → measure forgetting
        train(mA, tB, XBtr, yBtr, device)
        R_BA       = 1 - error(mA, tA, XAte, yAte, device)
        forgetting = R_AA - R_BA   # positive, higher = worse

        results.append({
            'label':      f"T{tA}→T{tB}",
            'tA': tA,     'tB': tB,
            'rho':        rho,
            'R_AA':       R_AA,
            'R_BA':       R_BA,
            'forgetting': forgetting,
        })

        print(f"[{idx:>2}/{total}]  "
              f"T{tA}→T{tB} ({TASK_NAMES[tA][:6]}→{TASK_NAMES[tB][:6]})  "
              f"ρ={rho:+.3f}  "
              f"R_AA={R_AA:.3f}  R_BA={R_BA:.3f}  "
              f"forget={forgetting:.3f}")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(results):
    rhos = np.array([r['rho']        for r in results])
    fgts = np.array([r['forgetting'] for r in results])
    lbls = [r['label'] for r in results]

    r_val, p_val = pearsonr(rhos, fgts)
    rho_sp, _    = spearmanr(rhos, fgts)

    print(f"\nPearson r  : {r_val:.3f}  (p={p_val:.4f})")
    print(f"Spearman ρ : {rho_sp:.3f}")
    print(f"Expected   : negative r  (similar → less forgetting)")

    if r_val < -0.5 and p_val < 0.05:
        verdict = "✅  Theory HOLDS — similar tasks forget less"
    elif r_val < -0.3:
        verdict = "⚠️  Partial support"
    else:
        verdict = "❌  Not supported"
    print(f"Verdict    : {verdict}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter ρ vs forgetting
    ax = axes[0]
    ax.scatter(rhos, fgts, s=70, color='steelblue',
               edgecolors='white', linewidths=0.6, zorder=3)
    for lbl, rv, fv in zip(lbls, rhos, fgts):
        ax.annotate(lbl, (rv, fv), fontsize=7, alpha=0.8,
                    xytext=(4, 3), textcoords='offset points')
    xs   = np.linspace(rhos.min()*0.95, rhos.max()*1.05, 100)
    m, b = np.polyfit(rhos, fgts, 1)
    ax.plot(xs, m*xs+b, 'r--', lw=1.5,
            label=f'r={r_val:.3f} (p={p_val:.4f})\nρ_s={rho_sp:.3f}')
    ax.set_xlabel('ρ_AB  (Li & Hiratani Eq. 16, higher = more similar)',
                  fontsize=11)
    ax.set_ylabel('Forgetting  =  R_AA − R_BA', fontsize=11)
    ax.set_title('Task similarity vs forgetting\n'
                 '(CIFAR-10, ResNet-18, negative slope expected)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Right: bars sorted by ρ — should fall left→right if theory holds
    ax2    = axes[1]
    order  = np.argsort(rhos)
    norm   = (rhos[order] - rhos.min()) / (rhos.max() - rhos.min() + 1e-8)
    cols   = plt.cm.RdYlGn(norm)
    ax2.bar(range(20), fgts[order], color=cols,
            edgecolor='white', linewidth=0.4)
    ax2.set_xticks(range(20))
    ax2.set_xticklabels([lbls[i] for i in order],
                        rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Forgetting', fontsize=11)
    ax2.set_title('Pairs sorted by ρ_AB\n'
                  'red=dissimilar, green=similar\n'
                  'Theory: bars fall left→right',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('e1_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: e1_results.png")


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    results = run(device)
    import json
    with open('e1_single_results.json', 'w') as f:
        json.dump(results, f, indent=1)
    plot(results)