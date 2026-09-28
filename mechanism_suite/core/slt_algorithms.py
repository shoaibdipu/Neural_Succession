#!/usr/bin/env python3
"""slt_algorithms.py — the pieces E4, E5 and M0 need.

Three groups, all frozen per Phase 0:

  A9   four RRT penalties. Historical campaigns evaluated covariance variants;
       the exact Jacobian RRT-J branch did not complete.  The revised R8 E5
       benchmark therefore keeps literal/centered covariance as separate
       ablations and treats RRT-J as the theory-faithful algorithmic branch.

  E4   q as the replay REINFORCEMENT coefficient in L = L_B + q*L_A, which
       implements Theorem 4's g_i^(A)(q) = q*g_i^(A,full) exactly. A fixed
       minibatch holding fraction q of A data changes B pressure too and does
       not match the theorem. The old E4 varied buffer size/diversity, which
       answers a different question.

  M0   fit Eq (1) to observed trajectories. Definition 5 claims Eq (1) "is not
       an approximation -- it is an exact reformulation of gradient descent
       dynamics", and NO confirmatory experiment tests that. All four theorems
       rest on it, so this is promoted from diagnostic to a test in its own
       right. It uses post-B information, so it can never stand in for E1.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# A9 -- RRT penalties. Definition 7 sums over ALL neuron pairs, no task-A term.
# ---------------------------------------------------------------------------


def rrt_j(jacobians, lam):
    """RRT-J: exact Definition 7.  Omega = sum_{i!=j} alpha_ij^2  with
    alpha_ij = <phi_i,phi_j> / (||phi_i|| ||phi_j||)  -- COSINE normalised.

    `jacobians` is a [K, d] torch tensor of mean feature Jacobians, so this
    must be differentiable w.r.t. theta (build it inside the graph).
    """
    n = jacobians.norm(dim=1, keepdim=True).clamp_min(1e-12)
    A = (jacobians / n) @ (jacobians / n).t()
    off = A - torch.diag(torch.diag(A))
    return lam * (off ** 2).sum()


def rrt_cov_literal(features, lam):
    """RRT-Cov-literal: the draft's stated proxy, 5.2 approximation #2.
    Sigma_ij = E[n_i(x) n_j(x)] -- UNCENTERED second moment. This is the
    primary cheap variant; keep separate from the centered-covariance ablation."""
    S = (features.t() @ features) / features.shape[0]
    off = S - torch.diag(torch.diag(S))
    return lam * (off ** 2).sum()


def rrt_cov_centered(features, lam):
    """RRT-Cov-centered: what the previous run actually tested. Kept ONLY as an
    ablation so the 0.589/0.201 result remains attributable."""
    Phi = features - features.mean(0, keepdim=True)
    C = (Phi.t() @ Phi) / Phi.shape[0]
    off = C - torch.diag(torch.diag(C))
    return lam * (off ** 2).sum()


def rrt_corr(features, lam):
    """RRT-Corr: normalised activation analogue, Sigma_ij/sqrt(Sigma_ii Sigma_jj).
    Optional ablation, closest activation-space analogue of Definition 7."""
    S = (features.t() @ features) / features.shape[0]
    d = torch.diag(S).clamp_min(1e-12).sqrt()
    A = S / (d[:, None] * d[None, :])
    off = A - torch.diag(torch.diag(A))
    return lam * (off ** 2).sum()


RRT_VARIANTS = {"rrt_j": rrt_j, "rrt_cov_literal": rrt_cov_literal,
                "rrt_cov_centered": rrt_cov_centered, "rrt_corr": rrt_corr}


def mean_feature_jacobian_differentiable(model, xb, max_features=None):
    """[K, d] mean feature Jacobian kept inside the autograd graph, for RRT-J.
    Expensive: one create_graph backward per feature. Subsample features if
    needed and say so in the config."""
    xb = xb.requires_grad_(True)
    feats = model.features(xb)
    K = feats.shape[1]
    idx = range(K) if max_features is None else range(min(K, max_features))
    rows = []
    for i in idx:
        g, = torch.autograd.grad(feats[:, i].sum(), xb, create_graph=True, retain_graph=True)
        rows.append(g.mean(0).reshape(-1))
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# E4 -- q as the reinforcement coefficient
# ---------------------------------------------------------------------------


def q_replay_loss(model, task_b, xb_b, yb_b, task_a, xb_a, yb_a, q, *,
                  global_labels=False, offset_b=0, offset_a=0):
    """L = L_B + q * L_A^replay.

    Theorem 4 assumes g_i^(A)(q) = q * g_i^(A,full), so q must scale the
    task-A reinforcement while leaving task-B competitive pressure untouched.
    Batch construction is FIXED across q: same B batch size, same replay batch
    size, same optimizer-step count. Only the coefficient moves.
    """
    import torch.nn.functional as F
    fb = model.features(xb_b)
    lb = F.cross_entropy(model.heads[task_b](fb) if not global_labels
                         else model.heads[0](fb)[:, offset_b:offset_b + 2], yb_b)
    if q <= 0 or xb_a is None:
        return lb, float(lb), 0.0
    fa = model.features(xb_a)
    la = F.cross_entropy(model.heads[task_a](fa) if not global_labels
                         else model.heads[0](fa)[:, offset_a:offset_a + 2], yb_a)
    return lb + q * la, float(lb), float(la)


def empirical_q_star(qs, retentions, threshold, min_seeds_frac=2 / 3):
    """Pre-registered stabilisation rule: the smallest q meeting the retention
    criterion in at least min_seeds_frac of seeds. Declared before any run so
    the transition point is not chosen after seeing the curve.

    retentions: {q: [per-seed retention]}
    """
    for q in sorted(qs):
        vals = np.asarray(retentions[q], dtype=float)
        if len(vals) and (vals >= threshold).mean() >= min_seeds_frac:
            return float(q)
    return None                                     # never stabilises in range


# ---------------------------------------------------------------------------
# M0 -- does Eq (1) actually describe the trajectory?
# ---------------------------------------------------------------------------


def fit_lv_trajectory(x_A_traj, x_B_traj, taus, ridge=1e-6, cond_max=1e8):
    """Fit  dx_i^A/dtau = x_i^A (g_i - sum_j alpha_ij x_j^B)  per A feature.

    Regress  (dx_i/dtau)/x_i  on  [1, -x^B]  to recover g_i and alpha_i..
    Returns held-out R^2 (second half of the trajectory), the recovered alpha
    row-matrix, and the fraction of recovered alpha that is negative.

    IDENTIFIABILITY LIMIT, found by synthetic recovery test: with collinear
    task-B trajectories the LV form fits almost perfectly (held-out R2 = 0.98)
    while individual alpha_ij are NOT recoverable (pearson 0.35 against the true
    alpha, sign agreement 0.59 ~ chance). Only the aggregate sum_j alpha_ij x_j
    is identified. So M0 can test whether Eq (1) DESCRIBES the dynamics, but can
    only recover alpha when the x^B trajectories are sufficiently diverse.
    `cond_design` is returned for exactly this reason. `alpha_identifiable`
    is true only when the frozen conditioning criterion is met. A high held-out
    R2 with an ill-conditioned design is evidence that the aggregate LV pressure
    can describe the trajectory, NOT that individual alpha_ij are identifiable.
    The default threshold (cond(X^T X) <= 1e8) is fixed before real-data M0 runs
    and may be overridden only explicitly via the driver configuration.

    Compare alpha_hat against Definition-4 alpha only when the design is well
    conditioned.  A good LV fit with poor alpha agreement can leave the LV
    dynamical backbone viable, but replacing Definition 4 would require every
    affected theorem to be re-derived/rechecked; it does NOT automatically
    preserve Theorems 1-4.
    """
    x_A_traj = np.asarray(x_A_traj, float)          # [T, K_A]
    x_B_traj = np.asarray(x_B_traj, float)          # [T, K_B]
    taus = np.asarray(taus, float)
    T_, K_A = x_A_traj.shape
    if T_ < 6:
        return {"error": "need >= 6 trajectory points"}

    dt = np.gradient(taus)
    half = T_ // 2
    alphas, gs, r2s = [], [], []
    for i in range(K_A):
        xi = x_A_traj[:, i]
        if xi.max() <= 1e-12:
            continue
        y = np.gradient(xi, taus) / np.clip(xi, 1e-10, None)     # d log x_i / dtau
        X = np.concatenate([np.ones((T_, 1)), -x_B_traj], 1)
        Xtr, ytr = X[:half], y[:half]
        w = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(X.shape[1]), Xtr.T @ ytr)
        pred = X[half:] @ w
        ss = float(((y[half:] - pred) ** 2).sum())
        tot = float(((y[half:] - y[half:].mean()) ** 2).sum()) + 1e-12
        r2s.append(1.0 - ss / tot)
        gs.append(float(w[0]))
        alphas.append(w[1:])
    if not alphas:
        return {"error": "no eligible A features"}
    Ah = np.asarray(alphas)
    Xfull = np.concatenate([np.ones((T_, 1)), -x_B_traj], 1)
    cond_design = float(np.linalg.cond(Xfull.T @ Xfull))
    alpha_identifiable = bool(np.isfinite(cond_design) and cond_design <= float(cond_max))
    return {"n_features_fit": len(alphas),
            "cond_design": cond_design,
            "cond_max": float(cond_max),
            "alpha_identifiable": alpha_identifiable,
            "alpha_identifiability_reason": ("well_conditioned" if alpha_identifiable else "ill_conditioned_design"),
            "heldout_R2_median": float(np.median(r2s)),
            "heldout_R2_mean": float(np.mean(r2s)),
            "frac_features_R2_gt_0.5": float((np.asarray(r2s) > 0.5).mean()),
            "g_hat_median": float(np.median(gs)),
            "frac_alpha_hat_negative": float((Ah < 0).mean()),
            "alpha_hat": Ah}


def compare_alpha(alpha_def4, alpha_hat):
    """Does the fitted alpha agree with Definition 4? Decides whether a good LV
    fit rescues the theorems with a different alpha."""
    from scipy import stats
    n = min(alpha_def4.shape[0], alpha_hat.shape[0])
    m = min(alpha_def4.shape[1], alpha_hat.shape[1])
    a = alpha_def4[:n, :m].ravel()
    b = alpha_hat[:n, :m].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return {"error": "too few comparable entries"}
    return {"pearson": float(stats.pearsonr(a[ok], b[ok])[0]),
            "spearman": float(stats.spearmanr(a[ok], b[ok])[0]),
            "sign_agreement": float((np.sign(a[ok]) == np.sign(b[ok])).mean()),
            "n_compared": int(ok.sum())}
