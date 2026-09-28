#!/usr/bin/env python3
"""slt_theory.py — the shared Jacobian / Lotka-Volterra engine.

ONE implementation of the objects Theorems 1-4 refer to. E1, E2, E3, E4, E6 all
import from here. Previously each experiment used a different proxy for the same
quantity, which is why their results were not comparable.

Implements, with the frozen Phase-0 decisions:

  Definition 1   feature_jacobians()        mean input Jacobian, feature identity kept
  Definition 2   feature_strength()
  Definition 3   resource_consumption()     PCA resources
  Definition 4   competitive_alpha_AB()     <phi_i^A, phi_j^B> / ||phi_i^A||^2   (ASYMMETRIC)
  Definition 7   rrt_overlap_matrix()       <phi_i, phi_j> / (||phi_i|| ||phi_j||)  (COSINE)
  Corollary 1.1  lambdas()                  lambda_i, lambda_bar, lambda_min, lambda_eff
  A5/A6/A17      estimate_g_and_equilibrium()

NOT implemented here, deliberately: Neural R* (Definition 6) and Chesson ND/FD
(Corollary 3.1). Both are blocked on Phase-0 items A15 and A16 -- Definition 6
uses one index j for both the task-B feature and the PCA resource, and
Corollary 3.1's intra-task term alpha_ii^(AA) is identically 1 under
Definition 4. Coding either now would mean inventing a resource weighting and
calling it the original theorem. See PROTOCOL.md.
"""
from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Definition 1 -- mean input Jacobian, per feature
# ---------------------------------------------------------------------------


def feature_jacobians(model, X, *, arch, dataset, reshape_fn, n_probe=100,
                      proj_dim=None, seed=0, device="cpu", chunk=None):
    """phi_i^(t) = E_{x~D_t}[ grad_x n_i(x) ]   for every feature i.

    One backward pass per feature: d/dX sum_n n_i(x_n) gives grad_{x_n} n_i(x_n)
    for each n (each term depends only on its own x_n), so the row mean is the
    mean Jacobian exactly. K backward passes total, not K*N.

    Feature identity is preserved. The optional JL projection is applied AFTER
    the exact per-feature Jacobian, so row i still corresponds to feature i --
    but note this does NOT deliver 5.2 approximation #1's speedup, since the
    d-dimensional Jacobian has already been materialised. It only shrinks
    storage and downstream cost. A genuine speedup needs projected JVPs (P phi_i
    computed directly). For the confirmatory E1, prefer EXACT Jacobians so that
    projection error cannot be blamed for a weak result: 400x784 and 512x3072
    are small once averaged over probes, and the cost is the repeated backward
    passes, not the storage.

    Correctness requires model.eval() with frozen BatchNorm: the one-backward-
    per-feature identity holds only when samples do not interact in the forward
    pass. Verified on MLP (max|batched - per-sample| = 4.1e-08); re-verify on
    ResNet-18 before the CIFAR run. This is the step E0 got wrong: it
    collapsed each task into one Gram matrix and took a single cosine, which
    destroys the per-feature indexing Definition 4 requires.

    Returns J of shape [K, d] (or [K, proj_dim]), float64.
    """
    model.eval()
    Xs = X[:n_probe]
    Xr = reshape_fn(arch, dataset, Xs)
    xb = torch.tensor(Xr, dtype=torch.float32, device=device).requires_grad_(True)

    feats = model.features(xb)                      # [N, K]
    K = feats.shape[1]
    rows = []
    idx = range(K) if chunk is None else range(K)
    for i in idx:
        g, = torch.autograd.grad(feats[:, i].sum(), xb, retain_graph=(i < K - 1))
        rows.append(g.detach().mean(0).reshape(-1).cpu().numpy())
    J = np.asarray(rows, dtype=np.float64)          # [K, d]

    if proj_dim is not None and proj_dim < J.shape[1]:
        rng = np.random.RandomState(seed)
        P = rng.normal(scale=1.0 / np.sqrt(proj_dim), size=(J.shape[1], proj_dim))
        J = J @ P
    return J


# ---------------------------------------------------------------------------
# Definition 4 vs Definition 7 -- DIFFERENT normalisations, not interchangeable
# ---------------------------------------------------------------------------


def competitive_alpha_AB(JA, JB, eps=1e-12):
    """Definition 4: alpha_ij^(AB) = <phi_i^A, phi_j^B> / ||phi_i^A||^2.

    Rows whose Task-A mean Jacobian norm is numerically zero are mathematically
    undefined under Definition 4.  They are returned as NaN rather than silently
    assigning zero; callers must use eligible_features() and exclude only the
    pre-specified ineligible rows.
    """
    JA = np.asarray(JA, dtype=np.float64)
    JB = np.asarray(JB, dtype=np.float64)
    num = JA @ JB.T
    den = (JA ** 2).sum(1, keepdims=True)
    out = np.full_like(num, np.nan, dtype=np.float64)
    valid = den[:, 0] > eps
    out[valid] = num[valid] / den[valid]
    return out


def rrt_overlap_matrix(J, eps=1e-12):
    """Definition 7:  alpha_ij = <phi_i, phi_j> / (||phi_i|| ||phi_j||).

    Cosine-normalised, symmetric, within one network, NO task-A term -- RRT is
    task-agnostic by design. Returns [K, K].
    """
    n = np.linalg.norm(J, axis=1, keepdims=True) + eps
    Jn = J / n
    return Jn @ Jn.T


def eligible_features(J, x_strength, eps_phi=1e-10, eps_x=1e-6):
    """A19. Definition 4 is undefined when ||phi_i|| = 0, so freeze eligibility
    on BOTH Definition-1 and Definition-2 quantities and report the four cases.

    A zero MEAN Jacobian does not by itself imply a dead unit: per-sample
    gradients can cancel, leaving a strongly active feature with ||E[grad]||=0.
    A zero mean Jacobian does not itself imply a dead unit because per-sample
    gradients may cancel.  In one MNIST/MLP diagnostic the zero-mean rows also
    had zero per-sample input gradients and very low strength, but the code does
    not assume this generalizes.  All four categories are reported per dataset
    and architecture.

    Category 2 (x > 0 but ||phi|| ~ 0) is the one to watch: if it is large,
    Definition 1/4 cannot describe those features at all.
    """
    pn = np.linalg.norm(J, axis=1)
    live_phi = pn > eps_phi
    live_x = x_strength > eps_x
    return (live_phi & live_x), {
        "n_features": int(len(pn)),
        "cat_both_live": int((live_phi & live_x).sum()),
        "cat_x_live_phi_zero": int((~live_phi & live_x).sum()),
        "cat_x_zero_phi_live": int((live_phi & ~live_x).sum()),
        "cat_both_zero": int((~live_phi & ~live_x).sum()),
        "frac_excluded": float(1.0 - (live_phi & live_x).mean()),
        "eps_phi": eps_phi, "eps_x": eps_x,
    }


def alpha_diagnostics(alpha_AB, alpha_AA=None, active=None, active_B=None):
    """A2: the proof of Theorem 1 assumes alpha >= 0, which Definition 4 does
    not deliver. Report the sign structure so that assumption is checked rather
    than presumed."""
    if active is not None:
        alpha_AB = alpha_AB[active][:, :]
        if alpha_AA is not None:
            alpha_AA = alpha_AA[np.ix_(active, active)]
    if active_B is not None:
        alpha_AB = alpha_AB[:, active_B]
    out = {"frac_alpha_negative": float((alpha_AB < 0).mean()),
           "alpha_min": float(alpha_AB.min()),
           "alpha_max": float(alpha_AB.max()),
           "alpha_mean": float(alpha_AB.mean())}
    if alpha_AA is not None:
        d = np.diag(alpha_AA)
        # A16: under Definition 4 the self term is identically 1, so it carries
        # no information about intra-task competition.
        out["alpha_ii_AA_mean"] = float(d.mean())
        # Confirmed on real data: alpha_ii == 1 to 2.2e-07 on live features.
        # This is the A16 defect -- Corollary 3.1 uses alpha_ii^(AA) as its
        # intra-task competition term, and it carries no information.
        out["alpha_ii_AA_is_unity"] = bool(np.allclose(d, 1.0, atol=1e-5))
        off = alpha_AA[~np.eye(len(alpha_AA), dtype=bool)]
        out["alpha_ik_AA_offdiag_mean"] = float(off.mean())
    return out


# ---------------------------------------------------------------------------
# Definition 2 / Definition 3
# ---------------------------------------------------------------------------


@torch.no_grad()
def feature_strength(model, X, *, arch, dataset, reshape_fn, max_n=2000,
                     batch=256, device="cpu"):
    """Definition 2 (verified from the rendered page 5):

        x_i^(t)(tau) = ( E_{x~D_t}[ n_i(x; theta(tau)) ] )^2

    The SQUARE OF THE MEAN, not the mean of the square. These differ by the
    variance, and (E[n])^2 vanishes when the mean activation cancels even if the
    feature is highly variable -- the same mean-cancellation structure as
    Definition 1. For the post-ReLU features used here (ResNet avgpool output,
    MLP after ReLU) n_i >= 0, so no cancellation occurs and the two orderings
    coincide in sign; the distinction would bite on pre-activation or tanh units.
    """
    model.eval()
    Xr = reshape_fn(arch, dataset, X[:max_n])
    acc = []
    for i in range(0, len(Xr), batch):
        xb = torch.tensor(Xr[i:i + batch], dtype=torch.float32, device=device)
        acc.append(model.features(xb).cpu().numpy())   # no abs: Def 2 squares the mean
    return (np.concatenate(acc, 0).mean(0).astype(np.float64)) ** 2


def resource_consumption(J, X_flat, n_resources=32):
    """Definition 3: c_ij = (phi_i . v_j)^2 over PCA directions v_j of the data.

    Provided for completeness. Note it is NOT enough to specify Neural R*:
    Definition 6 writes R*_ij = alpha_ij^(AB) xbar_j^(B) / c_ij, using j as the
    task-B feature index AND the resource index simultaneously. That collision
    (A15) must be resolved in the manuscript before E3 can be coded.
    """
    Xc = X_flat - X_flat.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:n_resources]                                 # [R, d]
    return (J @ V.T) ** 2                                # [K, R]


# ---------------------------------------------------------------------------
# A5 / A6 / A17 -- growth rates and the pre-B equilibrium
# ---------------------------------------------------------------------------


def estimate_g_and_equilibrium(x0, xdot0, alpha_self, *, ridge=1e-6, eps=1e-8,
                               method="complementarity"):
    """Estimate intrinsic reinforcement g and an LV equilibrium.

    Eq. (1) gives g = xdot/x + A x at the pre-intervention state.  A true
    non-negative LV equilibrium is a complementarity problem, not A x = g for
    every feature: extinct features may have x_i=0 with (A x-g)_i >= 0.

    Primary solver: bounded least-squares on the Fischer-Burmeister residual
        FB(x, A x-g) = sqrt(x^2 + y^2) - x - y,
    which is zero iff x>=0, y>=0 and x*y=0 componentwise.
    A ridge-NNLS equality solution is returned as a sensitivity diagnostic.
    Residuals are ALWAYS computed against the unregularized original equation.
    High residuals are reported; they are never used to silently drop pairs.
    """
    x0 = np.asarray(x0, float)
    xdot0 = np.asarray(xdot0, float)
    A0 = np.asarray(alpha_self, float)
    if len(x0) == 0:
        return np.asarray([], float), {"error": "empty active set"}
    if not np.all(np.isfinite(A0)):
        return np.zeros_like(x0), {"error": "non-finite alpha_self on active set"}
    g_hat = xdot0 / np.clip(x0, eps, None) + A0 @ x0

    def _eq_resid(x):
        return float(np.linalg.norm(A0 @ x - g_hat) / (np.linalg.norm(g_hat) + eps))
    def _comp_metrics(x):
        y = A0 @ x - g_hat
        neg_x = np.minimum(x, 0.0)
        neg_y = np.minimum(y, 0.0)
        comp = x * y
        return {
            "equation_residual_unregularized": _eq_resid(x),
            "feasibility_x_negative_l2": float(np.linalg.norm(neg_x)),
            "feasibility_pressure_negative_l2": float(np.linalg.norm(neg_y)),
            "complementarity_l2": float(np.linalg.norm(comp) / (np.linalg.norm(g_hat) + eps)),
            "max_complementarity_abs": float(np.max(np.abs(comp))) if len(comp) else 0.0,
        }

    # Sensitivity: the earlier equality NNLS estimator.
    try:
        from scipy.optimize import nnls
        Areg = A0 + ridge * np.eye(len(x0))
        x_nnls, _ = nnls(Areg, g_hat)
        nnls_diag = _comp_metrics(x_nnls)
    except Exception as e:  # pragma: no cover
        x_nnls = np.zeros_like(x0)
        nnls_diag = {"error": str(e)}

    solver = "fischer_burmeister_least_squares"
    xbar = x_nnls.copy()
    try:
        from scipy.optimize import least_squares
        def fb_residual(z):
            y = A0 @ z - g_hat
            return np.sqrt(z*z + y*y + 1e-18) - z - y
        init = np.maximum(x_nnls, np.maximum(x0, 0.0))
        fit = least_squares(fb_residual, init, bounds=(0.0, np.inf),
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=5000)
        xbar = fit.x
        fb_norm = float(np.linalg.norm(fb_residual(xbar)) / (np.linalg.norm(g_hat) + eps))
        solver_ok = bool(fit.success)
        solver_msg = str(fit.message)
    except Exception as e:  # pragma: no cover
        fb_norm = float("nan")
        solver_ok = False
        solver_msg = f"fallback_to_nnls: {e}"
        solver = "nnls_fallback"

    diag = {
        "g_hat_min": float(np.min(g_hat)), "g_hat_max": float(np.max(g_hat)),
        "g_hat_median": float(np.median(g_hat)),
        "g_hat_min_positive": float(g_hat[g_hat > 0].min()) if (g_hat > 0).any() else float("nan"),
        "xbar_frac_zero": float((xbar <= 1e-12).mean()),
        "solver": solver, "solver_success": solver_ok, "solver_message": solver_msg,
        "fb_residual_normalised": fb_norm,
        "cond_alpha": float(np.linalg.cond(A0)) if len(A0) else float("nan"),
        "ridge_sensitivity": ridge,
        "nnls_sensitivity": nnls_diag,
    }
    diag.update(_comp_metrics(xbar))
    # Backward-compatible field name used by old logging.
    diag["residual_normalised"] = diag["equation_residual_unregularized"]
    return xbar, diag


def feature_velocity(model, task_id, X, y, *, arch, dataset, reshape_fn,
                     set_trainable_fn, lr, n_batches=1, batch=128, device="cpu",
                     n_cls=2, freeze_bn_stats=True):
    """Estimate dx/dtau at the current state using virtual *vanilla SGD* steps.

    A18 defines LV/gradient-flow time as tau = sum_s eta_s.  The previous
    implementation used Adam and divided only by the number of batches, which
    made g and xbar scale incorrectly.  This implementation clones the model,
    applies plain SGD with zero momentum, and returns
        (x_after - x_before) / (n_actual_steps * lr).
    BatchNorm running statistics are frozen by default so the transition is due
    to gradient-updated parameters only, matching Eq. (1).
    """
    import copy
    m = copy.deepcopy(model).to(device)
    set_trainable_fn(m, task_id)
    kw = dict(arch=arch, dataset=dataset, reshape_fn=reshape_fn, device=device)
    x_before = feature_strength(m, X, **kw)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)
    Xr = reshape_fn(arch, dataset, X[:batch * n_batches])
    yt = torch.tensor(y[:batch * n_batches], dtype=torch.long, device=device)
    m.train()
    if freeze_bn_stats:
        for mod in m.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()
    steps = 0
    for b in range(n_batches):
        sl = slice(b * batch, min((b + 1) * batch, len(Xr)))
        if sl.start >= sl.stop:
            break
        xb = torch.tensor(Xr[sl], dtype=torch.float32, device=device)
        loss = torch.nn.functional.cross_entropy(m.heads[task_id](m.features(xb)), yt[sl])
        opt.zero_grad(); loss.backward(); opt.step(); steps += 1
    x_after = feature_strength(m, X, **kw)
    tau = max(steps * float(lr), 1e-12)
    return (x_after - x_before) / tau, x_before


# ---------------------------------------------------------------------------
# Corollary 1.1 -- the aggregates, with the A4 repairs
# ---------------------------------------------------------------------------


def lambdas(alpha_AB, xbar_B, x0_A=None, tau=2000.0):
    """lambda_i = sum_j alpha_ij^(AB) xbar_j^(B), plus the three aggregates.

    lambda_bar  arithmetic mean, as the draft writes it.
    lambda_min  the VALID lower-bound rate for Theorem 2 (A4). The draft's
                per-feature step needs lambda_i >= lambda_bar, true for only
                about half the features.
    lambda_eff  exact fixed-horizon aggregate (A4). With w_i = x_i(0)/sum x(0),
                    lambda_eff = -(1/tau) log( sum_i w_i exp(-lambda_i tau) )
                satisfies  sum_i x_i(0)(1-e^{-lambda_i tau})
                         = (sum_i x_i(0))(1-e^{-lambda_eff tau})   EXACTLY.
                Since exp(-lambda tau) is convex, lambda_eff <= lambda_bar, so
                the draft's lambda_bar OVERSTATES the bound and a correct
                dataset can violate it.
    """
    lam = alpha_AB @ xbar_B
    out = {"lambda_i": lam,
           "lambda_bar": float(lam.mean()),
           "lambda_min": float(lam.min()),
           "frac_lambda_positive": float((lam > 0).mean())}
    if x0_A is not None and x0_A.sum() > 0:
        w = x0_A / x0_A.sum()
        # Stable log-sum-exp: the naive form overflows to inf (hence -inf for
        # lambda_eff) whenever any lambda_i is strongly negative, which happens
        # routinely once alpha is signed.
        try:
            from scipy.special import logsumexp
            out["lambda_eff"] = float(-logsumexp(-lam * tau, b=w) / tau)
        except Exception:                                  # pragma: no cover
            z = -lam * tau
            zmax = z.max()
            out["lambda_eff"] = float(-(zmax + np.log((w * np.exp(z - zmax)).sum())) / tau)
        out["lambda_eff_le_lambda_bar"] = bool(out["lambda_eff"] <= out["lambda_bar"] + 1e-12)
    return out


def lv_time(n_steps, lr):
    """A18. Eq (1) contains no learning rate, so a raw gradient-step count is
    dimensionally inconsistent inside exp(-lambda*tau). Continuous gradient-flow
    time for N vanilla-SGD steps is tau = sum_s eta_s, i.e. N*eta at constant LR.

    Only valid for vanilla SGD. Under Adam or momentum the update is not
    eta*grad, so theorem-validation runs (E1/E2/E4) must use plain SGD; E5 may
    keep conventional optimizers.
    """
    lr = np.asarray(lr, dtype=np.float64)
    return float(lr.sum()) if lr.ndim else float(n_steps * lr)


def theorem2_bound(C, lam_rate, tau=2000.0):
    """ACC_A(0) - ACC_A(tau) >= C (1 - exp(-lam_rate tau)).

    Pass lam_rate=lambda_min for the repaired lower bound; lambda_eff for the
    fixed-horizon calibration. A3: C = c_A * sum_i x_i^(A)(0) with c_A a local
    margin-sensitivity constant, NOT a Lipschitz constant -- Lipschitz gives an
    upper bound and cannot yield this inequality.
    """
    # If lam_rate <= 0 the bound is negative, hence trivially satisfied. With
    # signed alpha this can occur. That is a property of the repaired theorem,
    # NOT a bug, and negative-lambda features must not be filtered out
    # to rescue it unless the theorem is formally restricted to a task-critical
    # subset declared in advance.
    if not np.isfinite(C) or not np.isfinite(lam_rate) or not np.isfinite(tau):
        return float("nan"), {"status": "invalid_nonfinite", "lambda_tau": float("nan")}
    val = float(C * (1.0 - np.exp(-np.clip(lam_rate * tau, -700, 700))))
    # A18 (new): the draft fixes tau = 2000 gradient steps but never states the
    # time-unit correspondence between tau and the LV time variable. Measured
    # lambda on MNIST/MLP is O(1), and 1-exp(-lambda*2000) = 1.000000 for any
    # lambda >~ 5e-3, so the bound collapses to the constant C and carries no
    # information about lambda. It is only informative for lambda*tau ~ O(1).
    lt = float(lam_rate * tau)
    if C <= 0:
        status = "vacuous_zero_C"
    elif lam_rate <= 0:
        status = "vacuous_facilitation"
    elif lt > 20:
        status = "saturated"
    else:
        status = "informative"
    return val, {"status": status, "lambda_tau": lt}


def theorem4_q_star(lambda_bar, g_min_A_full, eps=1e-12):
    """q* = lambda_bar / g_min^(A,full), with the frozen edge cases.

    lambda_bar <= 0  -> q* = 0     (net facilitation: no replay needed)
    q* > 1           -> reported as-is, NEVER clipped: it means the theorem
                       predicts no feasible replay fraction guarantees
                       coexistence.
    """
    if lambda_bar <= 0:
        return 0.0, "net_facilitation"
    if not np.isfinite(g_min_A_full) or g_min_A_full <= eps:
        return float("nan"), "invalid_g_min"
    q = float(lambda_bar / g_min_A_full)
    return q, ("infeasible_gt_1" if q > 1 else "ok")


def theorem4_q_star_featurewise(lambda_i, g_i_full, eps=1e-10):
    """Repaired Theorem-4 threshold: q* = max_i [lambda_i]_+ / g_i^(A,full).

    The condition for every resident feature to have non-negative invasion
    growth under L_B + q L_A is
        q * g_i^(A,full) >= lambda_i   for every task-critical feature i.
    Hence the exact threshold for the supplied feature set is the maximum ratio.

    No positively pressured feature is silently discarded.  A positive lambda
    paired with non-finite/non-positive g makes the theorem unable to certify a
    finite q and is reported explicitly.
    """
    lam=np.asarray(lambda_i,float); g=np.asarray(g_i_full,float)
    if lam.shape!=g.shape:
        return float('nan'),'invalid_shape',{'n_lambda':int(lam.size),'n_g':int(g.size)}
    if np.any(~np.isfinite(lam)):
        return float('nan'),'invalid_nonfinite_lambda',{'n_total':int(len(lam)),'n_nonfinite_lambda':int((~np.isfinite(lam)).sum())}
    need=lam>0
    if not need.any():
        return 0.0,'net_facilitation',{'n_total':int(len(lam)),'n_positive_pressure':0}
    bad=need&((~np.isfinite(g))|(g<=eps))
    if bad.any():
        return float('inf'),'infeasible_nonpositive_g',{'n_total':int(len(lam)),'n_positive_pressure':int(need.sum()),
            'n_bad_g_for_positive_pressure':int(bad.sum()),'bad_indices':np.where(bad)[0].astype(int).tolist()}
    ratios=lam[need]/g[need]
    q=float(np.max(ratios))
    return q,('infeasible_gt_1' if q>1 else 'ok'),{'n_total':int(len(lam)),'n_positive_pressure':int(need.sum()),
        'ratio_median':float(np.median(ratios)),'ratio_max':q,'argmax_local_index':int(np.where(need)[0][np.argmax(ratios)])}


# ---------------------------------------------------------------------------
# Displacement / mediation (supporting diagnostic, not a confirmatory test)
# ---------------------------------------------------------------------------


def param_displacement(theta_before, model):
    num = 0.0
    den = 0.0
    for n, p in model.named_parameters():
        if "head" in n or n not in theta_before:
            continue
        num += float(((p.detach() - theta_before[n]) ** 2).sum())
        den += float((theta_before[n] ** 2).sum())
    return float(np.sqrt(num) / (np.sqrt(den) + 1e-12))


def linear_cka(Xa, Xb):
    Xa = Xa - Xa.mean(0, keepdims=True)
    Xb = Xb - Xb.mean(0, keepdims=True)
    num = np.linalg.norm(Xa.T @ Xb, "fro") ** 2
    den = np.linalg.norm(Xa.T @ Xa, "fro") * np.linalg.norm(Xb.T @ Xb, "fro")
    return float(num / (den + 1e-12))


def snapshot_backbone(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if "head" not in n}
