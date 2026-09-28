# ============================================================================
# slt_common.py  —  Shared infrastructure for Successional Learning Theory (SLT)
#
# Import this from every experiment script. On Colab, write it to disk first:
#     %%writefile slt_common.py
#     <paste this file>
# then in later cells:  from slt_common import *
#
# Provides:
#   - Data loading (MNIST / CIFAR-10 / CIFAR-100 / Permuted-MNIST) -> numpy
#   - Models: MLP, ResNet-18, SmallViT  (each exposes .features(x) + forward(x,t))
#   - Continual-learning scenarios: task-IL (per-task heads) and class-IL (one head)
#   - Training / evaluation
#   - Task similarity rho  (Li & Hiratani 2025, Eq. 16):  bidirectional + pre-hoc
#   - FIVE overlap/similarity metrics that the paper argues about:
#       (1) rho_pre               behavioral zero-shot transfer  (your E1/E2 metric)
#       (2) act_cov_overlap       activation-covariance Omega    (paper's cheap proxy)
#       (3) jacobian_overlap      neuron-Jacobian Gram Omega      (the literal theory object)
#       (4) repr_overlap          centered representational similarity (CKA-style)
#       (5) gradient_overlap      NTK / task-gradient alignment   (Doan et al. baseline)
#   - Chesson decomposition: niche difference (ND) and fitness difference (FD)
#   - RRT penalty (activation-covariance niche regularizer)  and  EWC / rho-EWC helpers
#   - Metrics: ACC, BWT (sequential)  and  forgetting = R_AA - R_BA (pairwise)
#
# Design notes:
#   * Backbone is shared across tasks; per-task heads (task-IL) OR single head (class-IL).
#   * "Forgetting" (pairwise) follows your convention: R_AA - R_BA  (higher = worse).
#   * All overlap metrics return a scalar; higher = MORE similar UNLESS noted.
#   * Heavy diagnostics (jacobian_overlap, gradient_overlap) are subsampled and optional.
# ============================================================================

import os, copy, json, math, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------------------------------
# Normalization constants (match your existing CIFAR pipeline exactly)
# ----------------------------------------------------------------------------
CIFAR10_MEAN  = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD   = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# Standard CL task partitions
SPLITS = {
    'mnist':    [(0,1),(2,3),(4,5),(6,7),(8,9)],
    'cifar10':  [(0,1),(2,3),(4,5),(6,7),(8,9)],
    'cifar100': [list(range(i*10,(i+1)*10)) for i in range(10)],
}
IN_SHAPE = {'mnist': (1,28,28), 'cifar10': (3,32,32), 'cifar100': (3,32,32)}
FLAT_DIM = {'mnist': 784, 'cifar10': 3072, 'cifar100': 3072}

# ----------------------------------------------------------------------------
# Data loading via torchvision (reliable on Colab; downloads once, then cached)
#   Returns flat float32 arrays:
#     MNIST -> (N, 784);  CIFAR -> (N, 3, 32, 32)
# ----------------------------------------------------------------------------
def load_np(name, root=None):
    from torchvision import datasets
    root = root or os.environ.get('SLT_DATA', './data')
    name = name.lower()
    if name == 'mnist':
        tr = datasets.MNIST(root, train=True,  download=True)
        te = datasets.MNIST(root, train=False, download=True)
        def conv(ds):
            X = ds.data.numpy().astype(np.float32) / 255.0
            # ORIGINAL E1 MNIST protocol: scale to [0,1] only; no mean/std normalization.
            X = X.reshape(len(X), -1)
            y = ds.targets.numpy().astype(np.int64)
            return X, y
        Xtr, ytr = conv(tr); Xte, yte = conv(te)
    elif name in ('cifar10', 'cifar100'):
        D = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        mean = CIFAR10_MEAN if name == 'cifar10' else CIFAR100_MEAN
        std  = CIFAR10_STD  if name == 'cifar10' else CIFAR100_STD
        tr = D(root, train=True,  download=True)
        te = D(root, train=False, download=True)
        def conv(ds):
            X = np.asarray(ds.data, dtype=np.float32) / 255.0          # (N,32,32,3)
            X = (X - mean) / std
            X = X.transpose(0, 3, 1, 2).copy()                         # (N,3,32,32)
            y = np.asarray(ds.targets, dtype=np.int64)
            return X, y
        Xtr, ytr = conv(tr); Xte, yte = conv(te)
    else:
        raise ValueError(f"unknown dataset {name}")
    return Xtr, ytr, Xte, yte


def make_permuted_mnist(Xtr, Xte, n_tasks, seed=0):
    """Return list of (Xtr_perm, Xte_perm) under fixed random pixel permutations.
    Task 0 is identity; tasks 1.. apply distinct fixed permutations (near-orthogonal)."""
    rng = np.random.RandomState(seed)
    d = Xtr.shape[1]
    perms = [np.arange(d)] + [rng.permutation(d) for _ in range(n_tasks - 1)]
    return [(Xtr[:, p].copy(), Xte[:, p].copy()) for p in perms]


def binary_subset(X, y, c0, c1):
    """Extract a 2-class binary task; relabel to {0,1}."""
    m = (y == c0) | (y == c1)
    return X[m], (y[m] == c1).astype(np.int64)


def make_tasks(X, y, splits):
    """Slice into per-task (X, y) with WITHIN-TASK relabeled targets (0..c-1)."""
    tasks = []
    for classes in splits:
        classes = list(classes) if not isinstance(classes, list) else classes
        m = np.isin(y, classes)
        Xt = X[m]
        yt = np.array([classes.index(c) for c in y[m]], dtype=np.int64)
        tasks.append((Xt, yt))
    return tasks


def subsample(X, y, n, seed=0):
    if n is None or n >= len(X):
        return X, y
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], y[idx]

# ----------------------------------------------------------------------------
# Models.  Each model exposes:
#   .features(x)      -> penultimate representation (B, d)        [scenario-agnostic]
#   .forward(x, t)    -> logits.  task-IL uses head t; class-IL ignores t (one head)
#   .feat_dim         -> int
#   (ResNet/ViT) .features_at(x) optionally returns dict of intermediate reps for E8
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, n_tasks, n_cls, hidden=400, scenario='task'):
        super().__init__()
        self.scenario = scenario
        self.feat_dim = hidden
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        if scenario == 'task':
            self.heads = nn.ModuleList([nn.Linear(hidden, n_cls) for _ in range(n_tasks)])
        else:
            self.head = nn.Linear(hidden, n_tasks * n_cls)

    def features(self, x):
        h1 = self.relu(self.fc1(x))
        return self.relu(self.fc2(h1))

    def features_at(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return {'h1': h1, 'h2': h2}

    def forward(self, x, t=0):
        f = self.features(x)
        return self.heads[t](f) if self.scenario == 'task' else self.head(f)


class ResNet18CL(nn.Module):
    """Stock torchvision ResNet-18 backbone (matches your E1/E5) + heads."""
    def __init__(self, n_tasks, n_cls, scenario='task'):
        super().__init__()
        from torchvision.models import resnet18
        self.scenario = scenario
        self.feat_dim = 512
        bb = resnet18(weights=None)
        bb.fc = nn.Identity()
        self.bb = bb
        if scenario == 'task':
            self.heads = nn.ModuleList([nn.Linear(512, n_cls) for _ in range(n_tasks)])
        else:
            self.head = nn.Linear(512, n_tasks * n_cls)

    def features(self, x):
        return self.bb(x)

    def features_at(self, x):
        # Tap each residual stage for layer-wise displacement analysis (E8).
        bb = self.bb
        x = bb.relu(bb.bn1(bb.conv1(x))); x = bb.maxpool(x)
        l1 = bb.layer1(x); l2 = bb.layer2(l1)
        l3 = bb.layer3(l2); l4 = bb.layer4(l3)
        pooled = torch.flatten(bb.avgpool(l4), 1)
        gap = lambda z: z.mean(dim=(2, 3))
        return {'layer1': gap(l1), 'layer2': gap(l2),
                'layer3': gap(l3), 'layer4': gap(l4), 'feat': pooled}

    def forward(self, x, t=0):
        f = self.bb(x)
        return self.heads[t](f) if self.scenario == 'task' else self.head(f)


class SmallViT(nn.Module):
    """Compact ViT for 32x32 (architecture-diversity check). Uses torch's built-in
    TransformerEncoderLayer to minimize bug surface. ~2.5M params."""
    def __init__(self, n_tasks, n_cls, scenario='task',
                 img=32, patch=4, dim=192, depth=6, heads=3, mlp=384, in_ch=3):
        super().__init__()
        self.scenario = scenario
        self.feat_dim = dim
        self.np = (img // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.np + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, mlp, dropout=0.0,
                                           batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        if scenario == 'task':
            self.heads = nn.ModuleList([nn.Linear(dim, n_cls) for _ in range(n_tasks)])
        else:
            self.head = nn.Linear(dim, n_tasks * n_cls)

    def features(self, x):
        B = x.size(0)
        z = self.proj(x).flatten(2).transpose(1, 2)              # (B, np, dim)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1) + self.pos
        z = self.enc(z)
        return self.norm(z[:, 0])

    def features_at(self, x):
        return {'feat': self.features(x)}

    def forward(self, x, t=0):
        f = self.features(x)
        return self.heads[t](f) if self.scenario == 'task' else self.head(f)


def make_model(arch, n_tasks, n_cls, scenario='task', dataset='cifar10'):
    arch = arch.lower()
    if arch == 'mlp':
        return MLP(FLAT_DIM[dataset], n_tasks, n_cls, scenario=scenario)
    if arch in ('resnet18', 'resnet'):
        return ResNet18CL(n_tasks, n_cls, scenario=scenario)
    if arch == 'vit':
        in_ch = 1 if dataset == 'mnist' else 3
        img = 28 if dataset == 'mnist' else 32
        patch = 4 if dataset == 'mnist' else 4
        return SmallViT(n_tasks, n_cls, scenario=scenario, img=img, patch=patch, in_ch=in_ch)
    raise ValueError(arch)


def _reshape_for(arch, dataset, X):
    """MLP wants flat; conv/vit want NCHW. X is stored flat (mnist) or NCHW (cifar)."""
    arch = arch.lower()
    if arch == 'mlp':
        return X if X.ndim == 2 else X.reshape(len(X), -1)
    # conv/vit need image tensors
    if X.ndim == 2:                                   # mnist flat -> (N,1,28,28)
        return X.reshape(len(X), *IN_SHAPE[dataset])
    return X

# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def _set_trainable(model, task_id):
    """Train current head (task-IL) or the single head (class-IL) + full backbone."""
    if model.scenario == 'task':
        for i, h in enumerate(model.heads):
            for p in h.parameters():
                p.requires_grad = (i == task_id)
    else:
        for p in model.head.parameters():
            p.requires_grad = True
    for n, p in model.named_parameters():
        if 'heads' not in n and 'head' not in n:
            p.requires_grad = True


def train_task(model, task_id, X, y, *, arch, dataset, epochs=20, lr=1e-3,
               batch=128, wd=1e-4, extra_loss_fn=None, device=DEVICE,
               global_labels=False, class_offset=0, verbose=False, drop_last=False):
    """Fixed-epoch training on one task. No early stopping (matches your protocol).
    extra_loss_fn(model, features, logits) -> scalar tensor for regularizers (RRT/EWC).
    For class-IL pass global_labels=True and class_offset = task_id*n_cls."""
    model.to(device); _set_trainable(model, task_id)
    Xr = _reshape_for(arch, dataset, X)
    yy = (y + class_offset) if global_labels else y
    loader = DataLoader(TensorDataset(torch.tensor(Xr), torch.tensor(yy)),
                        batch_size=batch, shuffle=True, drop_last=drop_last)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            f = model.features(xb)
            logits = (model.heads[task_id](f) if model.scenario == 'task'
                      else model.head(f))
            loss = ce(logits, yb)
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(model, f, logits)
            loss.backward()
            opt.step()
        if verbose:
            print(f"    epoch {ep+1}/{epochs}  loss={loss.item():.4f}")
    return model


@torch.no_grad()
def accuracy(model, task_id, X, y, *, arch, dataset, device=DEVICE, batch=256,
             global_labels=False, class_offset=0, classil_classes=None):
    """task-IL: argmax within task head. class-IL: argmax over all classes (global)."""
    model.eval().to(device)
    Xr = _reshape_for(arch, dataset, X)
    yy = (y + class_offset) if global_labels else y
    correct = total = 0
    for i in range(0, len(Xr), batch):
        xb = torch.tensor(Xr[i:i+batch]).to(device)
        yb = torch.tensor(yy[i:i+batch]).to(device)
        f = model.features(xb)
        if model.scenario == 'task':
            pred = model.heads[task_id](f).argmax(1)
        else:
            logits = model.head(f)
            if classil_classes is not None:      # restrict to classes seen so far
                mask = torch.full_like(logits, float('-inf'))
                mask[:, classil_classes] = 0.0
                logits = logits + mask
            pred = logits.argmax(1)
        correct += (pred == yb).sum().item(); total += len(yb)
    return correct / total


def error(model, task_id, X, y, **kw):
    return 1.0 - accuracy(model, task_id, X, y, **kw)


def shuffled_error(model, task_id, X, y, *, n=5, **kw):
    errs = []
    for _ in range(n):
        ys = y.copy(); np.random.shuffle(ys)
        errs.append(error(model, task_id, X, ys, **kw))
    return float(np.mean(errs))

# ----------------------------------------------------------------------------
# Task similarity  rho  (Li & Hiratani 2025, Eq. 16)
# ----------------------------------------------------------------------------
def rho_pre(modelA, taskA_id, X_B, y_B, *, arch, dataset, n_samples=None,
            n_shuffle=5, device=DEVICE, seed=0):
    """Pre-hoc, unidirectional: rho = 1 - sqrt(e_B[W_A]/e_{B,shuf}[W_A]).

    By default this uses the FULL Task-B evaluation set, matching the original
    E1.5/E2/E4 protocol. Pass n_samples explicitly only where the original
    experiment did so (E5 uses 100).
    """
    Xs, ys = subsample(X_B, y_B, n_samples, seed=seed)
    # Preserve global NumPy RNG semantics used by the original shuffled-error code.
    eB  = error(modelA, taskA_id, Xs, ys, arch=arch, dataset=dataset, device=device)
    eBs = shuffled_error(modelA, taskA_id, Xs, ys, n=n_shuffle, arch=arch,
                         dataset=dataset, device=device)
    return float(1.0 - np.sqrt(eB / (eBs + 1e-8)))


def rho_pre_classil(model, previous_task_id, X_B, y_B, *, arch, dataset, n_cls,
                     n_samples=100, n_shuffle=5, device=DEVICE, seed=0):
    """Class-IL analogue of the original pre-hoc rho.

    The single class-IL head is sliced to the *previous task's* class block,
    yielding the same local n_cls-way decision rule as a task-specific head.
    Incoming Task-B local labels (0..n_cls-1) are then evaluated against that
    slice. This avoids using untrained future-class logits and keeps the
    behavioral definition faithful to the task-IL protocol.
    """
    Xs, ys = subsample(X_B, y_B, n_samples, seed=seed)
    Xr = _reshape_for(arch, dataset, Xs)
    lo = previous_task_id * n_cls
    hi = lo + n_cls
    model.eval().to(device)

    def _err(y_use):
        wrong = total = 0
        with torch.no_grad():
            for i in range(0, len(Xr), 256):
                xb = torch.tensor(Xr[i:i+256]).to(device)
                yb = torch.tensor(y_use[i:i+256]).to(device)
                logits = model.head(model.features(xb))[:, lo:hi]
                wrong += (logits.argmax(1) != yb).sum().item()
                total += len(yb)
        return wrong / total

    eB = _err(ys)
    sh = []
    for _ in range(n_shuffle):
        yp = ys.copy(); np.random.shuffle(yp); sh.append(_err(yp))
    eBs = float(np.mean(sh))
    return float(1.0 - np.sqrt(eB / (eBs + 1e-8)))


def rho_bidir(modelA, modelB, tA, tB, XA, yA, XB, yB, *, arch, dataset, device=DEVICE):
    """Symmetric bidirectional (post-hoc): average of both transfer directions."""
    eB_WA  = error(modelA, tA, XB, yB, arch=arch, dataset=dataset, device=device)
    eBs_WA = shuffled_error(modelA, tA, XB, yB, arch=arch, dataset=dataset, device=device)
    eA_WB  = error(modelB, tB, XA, yA, arch=arch, dataset=dataset, device=device)
    eAs_WB = shuffled_error(modelB, tB, XA, yA, arch=arch, dataset=dataset, device=device)
    t1 = np.sqrt(eB_WA / (eBs_WA + 1e-8)); t2 = np.sqrt(eA_WB / (eAs_WB + 1e-8))
    return float(1.0 - 0.5 * (t1 + t2))

# ----------------------------------------------------------------------------
# Feature extraction + overlap / similarity metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def feature_matrix(model, X, *, arch, dataset, max_n=2000, device=DEVICE, batch=256):
    model.eval().to(device)
    Xr = _reshape_for(arch, dataset, X)
    Xr, _ = subsample(Xr, np.zeros(len(Xr)), max_n)
    outs = []
    for i in range(0, len(Xr), batch):
        xb = torch.tensor(Xr[i:i+batch]).to(device)
        outs.append(model.features(xb).cpu())
    return torch.cat(outs).float()                  # (n, d)


def _cov(Phi, centered):
    if centered:
        Phi = Phi - Phi.mean(0, keepdim=True)
    return (Phi.t() @ Phi) / Phi.shape[0]            # (d, d)


def _cov_cosine(CA, CB):
    num = (CA * CB).sum()
    den = CA.norm() * CB.norm() + 1e-8
    return float((num / den).clamp(-1, 1))


def act_cov_overlap(model, X_A, X_B, *, arch, dataset, max_n=2000, device=DEVICE):
    """(2) Activation-covariance Omega proxy (paper's cheap metric).
    Cosine between UNCENTERED 2nd-moment matrices on Task-A vs Task-B inputs.
    Higher = features occupy the same directions for both tasks = more similar."""
    PA = feature_matrix(model, X_A, arch=arch, dataset=dataset, max_n=max_n, device=device)
    PB = feature_matrix(model, X_B, arch=arch, dataset=dataset, max_n=max_n, device=device)
    return _cov_cosine(_cov(PA, False), _cov(PB, False))


def repr_overlap(model, X_A, X_B, *, arch, dataset, max_n=2000, device=DEVICE):
    """(4) Centered representational similarity (CKA-style cosine of covariances)."""
    PA = feature_matrix(model, X_A, arch=arch, dataset=dataset, max_n=max_n, device=device)
    PB = feature_matrix(model, X_B, arch=arch, dataset=dataset, max_n=max_n, device=device)
    return _cov_cosine(_cov(PA, True), _cov(PB, True))


def jacobian_overlap(model, X_A, X_B, *, arch, dataset, k=32, n_probe=24,
                     device=DEVICE, seed=0):
    """(3) The literal theory object: Gram alignment of feature input-Jacobians.
    We project features to k random dims, average the per-sample input-Jacobian over
    n_probe inputs, form the kxk Gram G, and return cosine(G_A, G_B).
    EXPENSIVE -> heavily subsampled. Higher = aligned input-sensitivity geometry."""
    model.eval().to(device)
    d = model.feat_dim
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(d, k, generator=g).to(device)

    def gram(X):
        Xr = _reshape_for(arch, dataset, X)
        Xr, _ = subsample(Xr, np.zeros(len(Xr)), n_probe, seed=seed)
        xs = torch.tensor(Xr, dtype=torch.float32, device=device)
        Js = []
        for i in range(len(xs)):
            xi = xs[i:i+1].clone().requires_grad_(True)
            def f(z):
                return (model.features(z) @ P).reshape(-1)        # (k,)
            J = torch.autograd.functional.jacobian(f, xi, vectorize=True)
            Js.append(J.reshape(k, -1))
        Jbar = torch.stack(Js).mean(0)                            # (k, in)
        return Jbar @ Jbar.t()                                    # (k, k)

    return _cov_cosine(gram(X_A), gram(X_B))


def gradient_overlap(model, X_A, y_A, X_B, y_B, *, arch, dataset, n_cls,
                     taskA_id=0, device=DEVICE, n_samples=512, probe_steps=60):
    """(5) NTK / task-gradient alignment (Doan et al. 2021 baseline), pre-hoc.
    Fit a linear probe head for Task B on FROZEN Task-A features (cheap), then return
    cosine between the backbone gradients of Task-A loss and Task-B loss.
    NOTE: sign w.r.t. forgetting is reported empirically, not assumed."""
    model = copy.deepcopy(model).to(device)
    XAr = _reshape_for(arch, dataset, X_A); XBr = _reshape_for(arch, dataset, X_B)
    XAr, yA = subsample(XAr, y_A, n_samples); XBr, yB = subsample(XBr, y_B, n_samples)

    # 1) train a fresh linear probe head_B on frozen features for Task B
    for p in model.parameters():
        p.requires_grad_(False)
    probe = nn.Linear(model.feat_dim, n_cls).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        fB = model.features(torch.tensor(XBr).to(device))
    for _ in range(probe_steps):
        opt.zero_grad(); ce(probe(fB), torch.tensor(yB).to(device)).backward(); opt.step()

    # 2) backbone-gradient for each task; cosine of the flattened vectors
    bb_params = [p for n, p in model.named_parameters()
                 if ('heads' not in n and 'head' not in n)]
    for p in bb_params:
        p.requires_grad_(True)

    def bb_grad(Xr, yv, head):
        model.zero_grad()
        out = head(model.features(torch.tensor(Xr).to(device)))
        ce(out, torch.tensor(yv).to(device)).backward()
        return torch.cat([p.grad.reshape(-1).clone() for p in bb_params])

    headA = (model.heads[taskA_id] if model.scenario == 'task' else model.head)
    gA = bb_grad(XAr, yA, headA)
    gB = bb_grad(XBr, yB, probe)
    cos = (gA @ gB) / (gA.norm() * gB.norm() + 1e-8)
    return float(cos.clamp(-1, 1))

# ----------------------------------------------------------------------------
# Chesson Modern-Coexistence-Theory decomposition: ND and FD
# ----------------------------------------------------------------------------
def chesson_nd_fd(model, X_A, y_A, X_B, y_B, *, arch, dataset, n_cls,
                  device=DEVICE, max_n=2000):
    """Niche Difference (ND): within-task feature coherence MINUS cross-task overlap.
       Higher ND -> tasks use complementary feature subspaces -> safer coexistence.
    Fitness Difference (FD): how much stronger Task-B's gradient signal is than A's.
       Higher FD -> Task B dominates -> more forgetting.
    Coexistence predicted when (ND - FD) > 0 after z-scoring across pairs."""
    PA = feature_matrix(model, X_A, arch=arch, dataset=dataset, max_n=max_n, device=device)
    PB = feature_matrix(model, X_B, arch=arch, dataset=dataset, max_n=max_n, device=device)
    CA, CB = _cov(PA, True), _cov(PB, True)
    # split A in half to estimate within-task self-similarity (the "intra" baseline)
    h = len(PA) // 2
    intra = _cov_cosine(_cov(PA[:h], True), _cov(PA[h:], True))
    cross = _cov_cosine(CA, CB)
    ND = intra - cross                                  # >0 when cross-task overlap is low

    # FD: ratio of task-B vs task-A gradient magnitude on a shared linear probe
    gnA = _grad_norm(model, X_A, y_A, arch=arch, dataset=dataset, n_cls=n_cls, device=device)
    gnB = _grad_norm(model, X_B, y_B, arch=arch, dataset=dataset, n_cls=n_cls, device=device)
    FD = float((gnB - gnA) / (gnB + gnA + 1e-8))        # in (-1, 1); >0 means B stronger
    return ND, FD


def _grad_norm(model, X, y, *, arch, dataset, n_cls, device=DEVICE, n=512, steps=40):
    m = copy.deepcopy(model).to(device)
    Xr = _reshape_for(arch, dataset, X); Xr, y = subsample(Xr, y, n)
    for p in m.parameters():
        p.requires_grad_(False)
    probe = nn.Linear(m.feat_dim, n_cls).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-2); ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        f = m.features(torch.tensor(Xr).to(device))
    for _ in range(steps):
        opt.zero_grad(); ce(probe(f), torch.tensor(y).to(device)).backward(); opt.step()
    bb = [p for n_, p in m.named_parameters() if ('heads' not in n_ and 'head' not in n_)]
    for p in bb:
        p.requires_grad_(True)
    m.zero_grad()
    ce(probe(m.features(torch.tensor(Xr).to(device))), torch.tensor(y).to(device)).backward()
    return float(torch.cat([p.grad.reshape(-1) for p in bb]).norm())

# ----------------------------------------------------------------------------
# Regularizers / CL algorithm helpers
# ----------------------------------------------------------------------------
def rrt_penalty(features, lam):
    """RRT niche regularizer (activation-covariance proxy): penalize off-diagonal
    feature covariance on the current batch -> drives features toward disjoint niches.
    R = || C - diag(C) ||_F^2 ,  C = Phi^T Phi / B."""
    Phi = features - features.mean(0, keepdim=True)
    C = (Phi.t() @ Phi) / Phi.shape[0]
    off = C - torch.diag(torch.diag(C))
    return lam * (off ** 2).sum()


def compute_fisher(model, task_id, X, y, *, arch, dataset, device=DEVICE,
                   n_samples=500, global_labels=False, class_offset=0):
    """Diagonal empirical Fisher over trainable params (backbone + active head)."""
    Xr = _reshape_for(arch, dataset, X); Xr, y = subsample(Xr, y, n_samples)
    yy = (y + class_offset) if global_labels else y
    loader = DataLoader(TensorDataset(torch.tensor(Xr), torch.tensor(yy)),
                        batch_size=64, shuffle=False)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        f = model.features(xb)
        logits = model.heads[task_id](f) if model.scenario == 'task' else model.head(f)
        F.cross_entropy(logits, yb).backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += (p.grad ** 2) * len(xb)
    for n in fisher:
        fisher[n] /= len(Xr)
    return fisher


def ewc_penalty(model, fisher_list, param_list, lam):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for fisher, params in zip(fisher_list, param_list):
        for n, p in model.named_parameters():
            if n in fisher and n in params:
                loss = loss + (fisher[n] * (p - params[n]) ** 2).sum()
    return (lam / 2) * loss

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def acc_bwt(acc_matrix, n_tasks):
    """ACC = mean over tasks of final accuracy. BWT = mean (final - at-learning).
    BWT is negative under forgetting (more negative = worse)."""
    T = n_tasks
    ACC = float(np.mean([acc_matrix[T-1][j] for j in range(T)]))
    BWT = float(np.mean([acc_matrix[T-1][j] - acc_matrix[j][j] for j in range(T-1)]))
    return ACC, BWT


def pairwise_forgetting(R_AA, R_BA):
    """Your pairwise convention: drop in Task-A accuracy. Higher = worse."""
    return float(R_AA - R_BA)




def json_dump(obj, path):
    """Write JSON safely when NumPy scalars/arrays are present."""
    def conv(x):
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=conv)

# Export internal protocol helpers as well. Several experiment scripts intentionally
# use low-level helpers such as _reshape_for/_set_trainable/_cov so that the
# implementation stays in one audited place. Python's default `import *` omits
# underscore-prefixed names, therefore enumerate the complete module namespace.
__all__ = [name for name in globals() if not name.startswith('__')]

# ----------------------------------------------------------------------------
# Tiny self-test (no GPU/data needed): checks shapes + that everything imports.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("slt_common self-test on", DEVICE)
    set_seed(0)
    # synthetic 'cifar10'-shaped data
    X = np.random.randn(200, 3, 32, 32).astype(np.float32)
    y = np.random.randint(0, 2, size=200).astype(np.int64)
    for arch in ['resnet18', 'vit']:
        m = make_model(arch, n_tasks=5, n_cls=2, scenario='task', dataset='cifar10')
        train_task(m, 0, X, y, arch=arch, dataset='cifar10', epochs=1, batch=64)
        a = accuracy(m, 0, X, y, arch=arch, dataset='cifar10')
        ov = act_cov_overlap(m, X[:100], X[100:], arch=arch, dataset='cifar10', max_n=100)
        print(f"  {arch:9s} acc={a:.3f}  act_cov_overlap={ov:.3f}  feat_dim={m.feat_dim}")
    # MLP on synthetic mnist
    Xm = np.random.randn(200, 784).astype(np.float32)
    m = make_model('mlp', 5, 2, scenario='task', dataset='mnist')
    train_task(m, 0, Xm, y, arch='mlp', dataset='mnist', epochs=1, batch=64)
    print("  mlp acc=", accuracy(m, 0, Xm, y, arch='mlp', dataset='mnist'))
    print("OK")
