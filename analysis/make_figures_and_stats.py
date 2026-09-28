#!/usr/bin/env python3
"""Regenerate every figure and every derived statistic used in the revised manuscript.

Inputs : ../records/**  (JSON records written by the experiment scripts; nothing is re-trained here)
Outputs: ../figures/*.pdf and ../records/derived/derived_revision_stats.json

Run:  python analysis/make_figures_and_stats.py
"""
from __future__ import annotations
import collections, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D, FIG = ROOT / "records", ROOT / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20260926)
B = 10000

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", grey="#8C8C8C", dark="#333333")
W = 5.5  # ICLR text width (in)
OUT: dict = {}


def load(p):
    return json.load(open(D / p))


def summ(obj):
    return obj.get("summary") or obj.get("analysis") or obj


def tag(ax, s):
    t = ax.get_title()
    ax.set_title(f"({s}) " + t if t else f"({s})")


def cluster_boot_r(x, y, g, n=B):
    x, y, g = map(np.asarray, (x, y, g))
    ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}; vals = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in RNG.choice(ug, len(ug), replace=True)])
        if x[ii].std() > 0 and y[ii].std() > 0:
            vals.append(np.corrcoef(x[ii], y[ii])[0, 1])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ols_band(ax, x, y, color, n=200):
    xs = np.linspace(x.min(), x.max(), n)
    b1, b0 = np.polyfit(x, y, 1); yhat = b0 + b1 * xs
    res = y - (b0 + b1 * x); s = np.sqrt(np.sum(res ** 2) / (len(x) - 2))
    se = s * np.sqrt(1 / len(x) + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t = stats.t.ppf(0.975, len(x) - 2)
    ax.plot(xs, yhat, color=color, lw=1.1, zorder=3)
    ax.fill_between(xs, yhat - t * se, yhat + t * se, color=color, alpha=0.15, lw=0, zorder=1)


# ---------------------------------------------------------------- E1 / E1.5 / E2
e1 = load("succession/e1_checkpoint.json")
pm = collections.defaultdict(list)
for r in e1:
    pm[(r["tA"], r["tB"])].append((r["rho"], r["forgetting"]))
keys = sorted(pm)
x1 = np.array([np.mean([a for a, _ in pm[k]]) for k in keys])
y1 = np.array([np.mean([b for _, b in pm[k]]) for k in keys])
y1sd = np.array([np.std([b for _, b in pm[k]], ddof=1) for k in keys])
g_in = np.array([k[1] for k in keys])
r1 = stats.pearsonr(x1, y1)[0]
# exact task-label (QAP) permutation over the 5! relabelings of task identities
T = 5; Rm = np.full((T, T), np.nan); Fm = np.full((T, T), np.nan)
for (a, b), xv, yv in zip(keys, x1, y1):
    Rm[a, b], Fm[a, b] = xv, yv
off = [(a, b) for a in range(T) for b in range(T) if a != b]
perm_r = []
for p in itertools.permutations(range(T)):
    perm_r.append(np.corrcoef([Rm[p[a], p[b]] for a, b in off], [Fm[a, b] for a, b in off])[0, 1])
perm_r = np.array(perm_r)
OUT["E1"] = dict(n_pairs=len(keys), repeats=3, pearson_r=float(r1),
                 spearman=float(stats.spearmanr(x1, y1).statistic), R2=float(r1 ** 2),
                 cluster_boot_ci_incoming=cluster_boot_r(x1, y1, g_in),
                 qap_exact_p_one_sided=float(np.mean(perm_r <= r1 + 1e-12)), qap_n_perm=len(perm_r),
                 sd_rho=float(x1.std()), sd_F=float(y1.std()),
                 mean_F=float(y1.mean()), range_F=[float(y1.min()), float(y1.max())],
                 per_incoming_r={int(t): float(np.corrcoef(x1[g_in == t], y1[g_in == t])[0, 1]) for t in range(T)})

e2 = load("succession/e2_checkpoint.json")
x2 = np.array([r["rho_pre"] for r in e2]); y2 = np.array([r["forgetting"] for r in e2]); n2 = len(x2)
pred, null = [], []
for i in range(n2):
    m = np.ones(n2, bool); m[i] = False
    pred.append(np.polyval(np.polyfit(x2[m], y2[m], 1), x2[i])); null.append(y2[m].mean())
pred, null = np.array(pred), np.array(null)
mae, mae0 = np.mean(np.abs(pred - y2)), np.mean(np.abs(null - y2))
OUT["E2"] = dict(n=int(n2), lopo_r=float(stats.pearsonr(pred, y2)[0]), lopo_mae_pp=float(100 * mae),
                 null_mae_pp=float(100 * mae0), mae_skill=float(1 - mae / mae0),
                 in_sample_r=float(stats.pearsonr(x2, y2)[0]))

e15 = load("succession/e15_checkpoint.json")
a15 = np.array([r["rho_pre"] for r in e15]); f15 = np.array([r["rho_full"] for r in e15])
v15 = np.array([r["rho_rev"] for r in e15]); F15 = np.array([r["forgetting"] for r in e15])
OUT["E15"] = dict(r_pre_full=float(stats.pearsonr(a15, f15)[0]), r_pre_rev=float(stats.pearsonr(a15, v15)[0]),
                  r_pre_F=float(stats.pearsonr(a15, F15)[0]), r_full_F=float(stats.pearsonr(f15, F15)[0]))
e1s = load("succession/e1_single_results.json")
xs1 = np.array([r["rho"] for r in e1s]); ys1 = np.array([r["forgetting"] for r in e1s])
allx = np.array([r["rho"] for r in e1]); ally = np.array([r["forgetting"] for r in e1])
reps = sorted(set(r["rep"] for r in e1))
Fr = np.array([[next(r["forgetting"] for r in e1 if r["rep"] == k and (r["tA"], r["tB"]) == kk) for kk in keys] for k in reps])
Xr = np.array([[next(r["rho"] for r in e1 if r["rep"] == k and (r["tA"], r["tB"]) == kk) for kk in keys] for k in reps])
OUT["E1"].update(single_run_r=float(stats.pearsonr(xs1, ys1)[0]), single_run_spearman=float(stats.spearmanr(xs1, ys1).statistic),
                 single_run_p=float(stats.pearsonr(xs1, ys1)[1]),
                 all_runs_r=float(stats.pearsonr(allx, ally)[0]), n_runs=int(len(allx)),
                 per_repeat_r={int(k): float(stats.pearsonr(Xr[i], Fr[i])[0]) for i, k in enumerate(reps)},
                 repeat_reliability_F=float(np.mean([np.corrcoef(Fr[i], Fr[j])[0, 1] for i in range(len(reps)) for j in range(i + 1, len(reps))])),
                 repeat_reliability_rho=float(np.mean([np.corrcoef(Xr[i], Xr[j])[0, 1] for i in range(len(reps)) for j in range(i + 1, len(reps))])))
tb2 = np.array([r["tB"] for r in e2]); predT = np.zeros(n2)
for t in np.unique(tb2):
    m = tb2 != t
    predT[~m] = np.polyval(np.polyfit(x2[m], y2[m], 1), x2[~m])
OUT["E2"].update(loto_incoming_r=float(stats.pearsonr(predT, y2)[0]), loto_incoming_mae_pp=float(100 * np.mean(np.abs(predT - y2))),
                 in_sample_spearman=float(stats.spearmanr(x2, y2).statistic))

fig, ax = plt.subplots(1, 3, figsize=(W, 1.72))
a = ax[0]
for k, (xv, yv, sv) in enumerate(zip(x1, y1, y1sd)):
    a.errorbar(xv, yv, yerr=sv, fmt="o", ms=3.2, color=COL["blue"], ecolor=COL["sky"], elinewidth=0.7, capsize=0, zorder=4)
ols_band(a, x1, y1, COL["red"])
a.set_xlabel(r"pre-invasion compatibility $\rho_{A\to B}$"); a.set_ylabel(r"forgetting $F_{A\to B}$")
a.set_title(f"E1: natural transitions ($r={r1:.3f}$)"); tag(a, "a")
a = ax[1]
lim = [min(pred.min(), y2.min()) - 0.02, max(pred.max(), y2.max()) + 0.02]
a.plot(lim, lim, ls="--", color=COL["grey"], lw=0.8)
a.fill_between(lim, [l - mae for l in lim], [l + mae for l in lim], color=COL["grey"], alpha=0.12, lw=0)
a.scatter(pred, y2, s=11, color=COL["green"], zorder=3)
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("held-out prediction (LOPO)"); a.set_ylabel("observed forgetting")
a.set_title(f"E2: prospective forecast (MAE {100*mae:.2f} pp)"); tag(a, "b")
a = ax[2]
a.scatter(a15, f15, s=11, color=COL["purple"], zorder=3)
lo, hi = min(a15.min(), f15.min()) - 0.02, max(a15.max(), f15.max()) + 0.02
a.plot([lo, hi], [lo, hi], ls="--", color=COL["grey"], lw=0.8)
a.set_xlabel(r"$\rho_{\mathrm{pre}}$ (resident only)"); a.set_ylabel(r"$\rho_{\mathrm{full}}$ (both directions)")
a.set_title(f"Pre-hoc sufficiency ($r={OUT['E15']['r_pre_full']:.3f}$)"); tag(a, "c")
fig.tight_layout(w_pad=1.2); fig.savefig(FIG / "fig_e1_e2.pdf"); plt.close(fig)

# ---------------------------------------------------------------- E3 coexistence
import re
# (a) natural-transition coexistence (E3 v1)
v1 = load("succession/e3v1_results.json")["rows"]
assert len(v1) == 20, len(v1)
rv = np.array([r["rho_pre"] for r in v1]); fv = np.array([r["forgetting"] for r in v1]); cv = np.array([r["coexist"] for r in v1])
thr_split = 0.067  # similar/dissimilar split used by the reproduction script (printed in the log)
sim = rv >= thr_split
OUT["E3_natural"] = dict(n=20, n_coexist=int(cv.sum()), auc_rho=float(roc_auc_score(cv, rv)), r_rho_F=float(stats.pearsonr(rv, fv)[0]),
                         split_rho=thr_split, n_similar=int(sim.sum()), mean_F_similar=float(fv[sim].mean()), mean_F_dissimilar=float(fv[~sim].mean()),
                         coexist_rate_similar=float(cv[sim].mean()), coexist_rate_dissimilar=float(cv[~sim].mean()),
                         mannwhitney_p_one_sided=float(stats.mannwhitneyu(fv[sim], fv[~sim], alternative="less").pvalue),
                         coexisting_are_top_rho=bool(set(np.argsort(-rv)[:int(cv.sum())]) == set(np.where(cv == 1)[0])))
# (b) reproduced pure-domain construction (E3 v2)
e3 = load("succession/e3v2_checkpoint.json")
same = np.array([r["forgetting"] for r in e3 if r["domain"] == "same"])
cross = np.array([r["forgetting"] for r in e3 if r["domain"] == "cross"])
rho3 = np.array([r["rho_pre"] for r in e3]); F3 = np.array([r["forgetting"] for r in e3])
OUT["E3_domain"] = dict(n_same=len(same), n_cross=len(cross), mean_same=float(same.mean()), mean_cross=float(cross.mean()),
                        coexist_same=int((same < 0.10).sum()), coexist_cross=int((cross < 0.10).sum()),
                        mannwhitney_p_two_sided=float(stats.mannwhitneyu(same, cross, alternative="two-sided").pvalue),
                        r_rho_F_within=float(stats.pearsonr(rho3, F3)[0]))
# (c) 84-transition MNIST suite (R1): mutual coexistence at four retention thresholds
r1 = pd.DataFrame(load("mechanism_suite/r1_mutual_coexistence_results.json")["rows"])
r1["cluster"] = r1["class_pair"].astype(str) + "|" + r1["transform"].astype(str)
thrs = ["0.80", "0.90", "0.95", "0.99"]
cl = r1["cluster"].unique(); idx = {k: np.where(r1["cluster"] == k)[0] for k in cl}
auc_t = {}
for t in thrs:
    yv = r1[f"mutual_{t}"].values.astype(int)
    a_r, a_s = roc_auc_score(yv, r1["rho_pre_pi"]), roc_auc_score(yv, r1["S"])
    br, bs, bd = [], [], []
    for _ in range(B):
        ii = np.concatenate([idx[k] for k in RNG.choice(cl, len(cl), replace=True)])
        if yv[ii].min() == yv[ii].max():
            continue
        x_r, x_s = roc_auc_score(yv[ii], r1["rho_pre_pi"].values[ii]), roc_auc_score(yv[ii], r1["S"].values[ii])
        br.append(x_r); bs.append(x_s); bd.append(x_r - x_s)
    auc_t[t] = dict(rate=float(yv.mean()), auc_rho=float(a_r), auc_S=float(a_s),
                    ci_rho=[float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))],
                    ci_S=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    diff=float(a_r - a_s), diff_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))])
nid = r1[~r1["identity"].astype(bool)]
OUT["E3_suite"] = dict(n=len(r1), n_clusters=len(cl), by_threshold=auc_t, r_rho_F=float(stats.pearsonr(r1["rho_pre_pi"], r1["forgetting"])[0]),
                       auc_rho_excl_identity_090=float(roc_auc_score(nid["mutual_0.90"].astype(int), nid["rho_pre_pi"])),
                       auc_S_excl_identity_090=float(roc_auc_score(nid["mutual_0.90"].astype(int), nid["S"])), n_excl_identity=len(nid))

fig, ax = plt.subplots(1, 3, figsize=(W, 1.78), gridspec_kw=dict(width_ratios=[1.15, 0.95, 1.05]))
a = ax[0]
a.scatter(rv[cv == 0], fv[cv == 0], s=12, color=COL["red"], label="displacement", zorder=3)
a.scatter(rv[cv == 1], fv[cv == 1], s=16, color=COL["green"], marker="D", label="coexistence", zorder=4)
a.axhline(0.20, ls="--", color=COL["grey"], lw=0.8); a.axvline(thr_split, ls=":", color=COL["grey"], lw=0.8)
a.set_xlabel(r"$\rho_{\mathrm{pre}}$"); a.set_ylabel(r"forgetting $F_{A\to B}$"); a.legend(frameon=False, loc="upper right")
a.set_title(f"E3a: natural transitions (AUC {OUT['E3_natural']['auc_rho']:.2f})"); tag(a, "a")
a = ax[1]
for i, (vals, col) in enumerate([(same, COL["green"]), (cross, COL["red"])]):
    jit = RNG.uniform(-0.12, 0.12, len(vals))
    a.scatter(np.full(len(vals), i) + jit, vals, s=6, color=col, alpha=0.8, zorder=3)
    a.boxplot(vals, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor="none", edgecolor=COL["dark"], lw=0.7), medianprops=dict(color=COL["dark"], lw=1),
              whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
a.axhline(0.10, ls="--", color=COL["grey"], lw=0.8)
a.set_xticks([0, 1]); a.set_xticklabels([f"same domain\n{OUT['E3_domain']['coexist_same']}/{len(same)} coexist",
                                         f"cross domain\n{OUT['E3_domain']['coexist_cross']}/{len(cross)} coexist"], fontsize=6)
a.set_ylabel("forgetting"); a.set_title("E3b: 60 pure-domain transitions"); tag(a, "b")
a = ax[2]
xt = [float(t) for t in thrs]
a.plot(xt, [auc_t[t]["auc_rho"] for t in thrs], "o-", color=COL["blue"], ms=3.2, label=r"$\rho_{\mathrm{pre}}$ (SLT)")
a.fill_between(xt, [auc_t[t]["ci_rho"][0] for t in thrs], [auc_t[t]["ci_rho"][1] for t in thrs], color=COL["blue"], alpha=0.15, lw=0)
a.plot(xt, [auc_t[t]["auc_S"] for t in thrs], "s--", color=COL["orange"], ms=3, label="representation compatibility $S$")
a.set_ylim(0.7, 1.02); a.set_xticks(xt); a.set_xticklabels(["0.80", "0.90", "0.95", "0.99"])
a.set_xlabel("mutual-retention threshold"); a.set_ylabel("coexistence AUC"); a.legend(frameon=False, loc="lower right", fontsize=5.6)
a.set_title("E3c: 84-transition suite"); tag(a, "c")
fig.tight_layout(w_pad=1.1); fig.savefig(FIG / "fig_coexistence.pdf"); plt.close(fig)

# ---------------------------------------------------------------- E8 controlled
ctrl = {}
fig, ax = plt.subplots(1, 2, figsize=(W * 0.62, 1.62))
for a, (name, fn, col, t) in zip(ax, [("mnist", "transition_suite/E8_mnist", COL["blue"], "E8a: MNIST permutations"),
                                      ("cifar", "transition_suite/E8_cifar", COL["orange"], "E8b: CIFAR-10 rotations")]):
    rows = load(fn + "/e8_results.json"); sm = load(fn + "/e8_summary.json")
    xx = np.array([r["rho_pre"] for r in rows]); yy = np.array([r["forgetting"] for r in rows])
    a.scatter(xx, yy, s=6, color=col, alpha=0.75, zorder=3)
    ols_band(a, xx, yy, COL["dark"])
    if sm["quadratic"]["vertex_in_range"]:
        xs = np.linspace(xx.min(), xx.max(), 200)
        a.plot(xs, np.polyval(sm["quadratic"]["coef"], xs), color=COL["red"], lw=0.9, ls="--", label="quadratic")
        a.legend(frameon=False, loc="upper right")
    a.set_title(f"{t} ($r={sm['linear']['pearson_r']:.3f}$)", fontsize=7.3)
    a.set_xlabel(r"$\rho_{\mathrm{pre}}$"); a.set_ylabel("forgetting")
    ctrl[name] = dict(n=int(sm["n"]), r=float(sm["linear"]["pearson_r"]), cv_R2_linear=float(sm["linear"]["group_CV_R2"]),
                      cv_R2_quadratic=float(sm["quadratic"]["group_CV_R2"]), vertex=float(sm["quadratic"]["vertex_rho"]),
                      vertex_in_range=bool(sm["quadratic"]["vertex_in_range"]), sd_rho=float(xx.std()), sd_F=float(yy.std()))
tag(ax[0], "a"); tag(ax[1], "b")
fig.tight_layout(w_pad=1.0); fig.savefig(FIG / "fig_controlled.pdf"); plt.close(fig)
OUT["E8"] = ctrl

# ---------------------------------------------------------------- E6 scale selection
nat = load("transition_suite/E0/e0_summary.json")["pair_means"]
dfn = pd.DataFrame(nat).T.astype(float)
pred_cols = [("rho_pre", "transition\ncompatibility"), ("gradient", "one-step\ngradient align."),
             ("act_cov", "activation\ncovariance"), ("jacobian", "Jacobian\noverlap"), ("repr", "represent.\nsimilarity")]
scale = {c: float(stats.pearsonr(dfn[c], dfn["forgetting"])[0]) for c, _ in pred_cols}
OUT["E6"] = dict(n_pairs=len(dfn), pearson_with_forgetting=scale)
s2 = summ(load("mechanism_suite/S2_jacobian_probe_sweep.json"))
a1 = load("mechanism_suite/A1_summary.json")["criteria"]
a2 = summ(load("mechanism_suite/A2_local_lv_validity.json"))
a3 = summ(load("mechanism_suite/A3_rstar_task_resources.json"))
OUT["E6_micro"] = dict(S2={k: {"median_partial_r": v["median_partial_r"], "frac_sig": v["fraction_p_shuffle_lt_0p05"]} for k, v in s2.items()},
                       A1=dict(forgetting_sd=a1["forgetting_sd"], task_r=a1["task_permutation"]["r_observed"],
                               task_p=a1["task_permutation"]["p_task_permutation"], fixed_alpha_wins=a1["fixed_alpha_wins"],
                               fixed_alpha_n=a1["fixed_alpha_n"], heldout_R2_median=a1["fixed_alpha_median_heldout_R2"]),
                       A2=dict(n_windows=a2["n_windows"], n_stable=a2["n_stable"], unstable_R2=a2["unstable_median_R2"]),
                       A3=dict(identified_fraction=a3["mean_identified_resource_fraction"], precision=a3["partition_precision"],
                               base_rate=a3["behavior_base_rate"]))
fig, ax = plt.subplots(1, 2, figsize=(W * 0.80, 1.72), gridspec_kw=dict(width_ratios=[1.45, 1]))
a = ax[0]
vals = [abs(scale[c]) for c, _ in pred_cols]
cols = [COL["blue"]] + [COL["grey"]] * 4
a.bar(range(len(vals)), vals, color=cols, width=0.65)
for i, v in enumerate(vals):
    a.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6)
a.set_xticks(range(len(vals))); a.set_xticklabels([l for _, l in pred_cols], fontsize=5.8)
a.set_ylabel(r"$|r|$ with forgetting"); a.set_ylim(0, 0.85)
a.set_title("E6a: which scale predicts displacement?"); tag(a, "a")
a = ax[1]
probes = [64, 256, 1024]
for lay, col, lab in [("feat", COL["red"], "penultimate"), ("layer3", COL["orange"], "layer 3")]:
    a.plot(probes, [s2[f"{lay}_n{p}"]["median_partial_r"] for p in probes], "o-", color=col, ms=3, label=lab)
a.axhline(0.10, ls="--", color=COL["grey"], lw=0.8); a.text(64, 0.108, "pre-registered pass", fontsize=5.6, color=COL["grey"])
a.set_xscale("log", base=2); a.set_xticks(probes); a.set_xticklabels(probes)
a.set_ylim(0, 0.2); a.set_xlabel("Jacobian probe size"); a.set_ylabel("median partial $r$")
a.set_title("E6b: microscopic coefficient"); a.legend(frameon=False, loc="upper left"); tag(a, "b")
fig.tight_layout(w_pad=1.2); fig.savefig(FIG / "fig_scale.pdf"); plt.close(fig)

# ---------------------------------------------------------------- E7 stabilization
a7 = load("mechanism_suite/A7_succession_stages.json")
con = a7["contrasts"]
OUT["E7"] = {k: dict(mean=v["mean_diff"], ci=v["ci95"], frac_pos=v["fraction_positive"]) for k, v in con.items()}
OUT["E7"]["n_runs"] = len(a7["rows"])
fig, ax = plt.subplots(1, 3, figsize=(W, 1.62), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
for a, metric, col, lab in [(ax[0], "plasticity", COL["blue"], r"plasticity $\overline{|\Delta\log x_i|}$"),
                            (ax[1], "turnover", COL["green"], "turnover (1$-$Jaccard)")]:
    for phase, eps, off_, ls in [("A", [1, 5, 10, 20], 0, "-"), ("B", [1, 5, 10], 24, "--")]:
        M = np.array([[next(e[metric] for e in r[phase] if e["epoch"] == ep) for ep in eps] for r in a7["rows"]], float)
        med, q1, q3 = np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
        xs = np.array(eps) + off_
        a.plot(xs, med, ls, color=col, marker="o", ms=2.8)
        a.fill_between(xs, q1, q3, color=col, alpha=0.15, lw=0)
    a.axvline(22, color=COL["grey"], lw=0.7, ls=":")
    a.text(10, a.get_ylim()[1] * 0.92, "establish A", fontsize=5.8, ha="center", color=COL["grey"])
    a.text(29, a.get_ylim()[1] * 0.92, "invade B", fontsize=5.8, ha="center", color=COL["grey"])
    a.set_xticks([1, 5, 10, 20, 25, 29, 34]); a.set_xticklabels(["1", "5", "10", "20", "1", "5", "10"])
    a.set_xlabel("epoch within phase"); a.set_ylabel(lab)
tag(ax[0], "a"); tag(ax[1], "b")
ax[0].set_title("E7a: plasticity (median, IQR)"); ax[1].set_title("E7b: feature turnover")
a = ax[2]
names = [("early_vs_late_plasticity", "early$-$late plasticity"), ("early_vs_late_turnover", "early$-$late turnover"),
         ("invasion_vs_preB_plasticity", "invasion$-$pre-B plasticity"), ("invasion_vs_preB_turnover", "invasion$-$pre-B turnover")]
for i, (k, lab) in enumerate(names):
    v = con[k]; sig = v["ci95"][0] > 0
    sc = 1.0 if "plasticity" not in k else 0.1  # plasticity scaled for a shared axis
    a.errorbar(v["mean_diff"] * sc, i, xerr=[[(v["mean_diff"] - v["ci95"][0]) * sc], [(v["ci95"][1] - v["mean_diff"]) * sc]],
               fmt="o", ms=3.2, color=COL["blue"] if sig else COL["grey"], capsize=1.5, elinewidth=0.9)
    a.text(0.62, i, f"{int(round(v['fraction_positive']*15))}/15", fontsize=6, va="center")
a.axvline(0, color=COL["dark"], lw=0.6)
a.set_yticks(range(len(names))); a.set_yticklabels([n for _, n in names], fontsize=6); a.invert_yaxis()
a.set_xlim(-0.35, 0.72); a.set_xlabel("contrast (plasticity $\\times 0.1$)")
a.set_title("E7c: stage contrasts, 95% CI"); tag(a, "c")
fig.tight_layout(w_pad=1.0); fig.savefig(FIG / "fig_stabilization.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Identifiability (Thm 3 / Cor 3.1)
r5 = {}
for lab, fn in [("CIFAR-100 contiguous", "mechanism_suite/r5_cifar100_standard.json"),
                ("CIFAR-100 random", "mechanism_suite/r5_cifar100_fresh.json"),
                ("CIFAR-100 semantic", "mechanism_suite/R5_sem_cifar100.json")]:
    d = load(fn); pmx = d.get("pair_means") or d.get("rows")
    xx = np.array([p.get("rho_pre_pi", p.get("rho_pre", np.nan)) for p in pmx], float)
    yy = np.array([p["forgetting"] for p in pmx], float); ok = np.isfinite(xx) & np.isfinite(yy)
    r5[lab] = dict(n=int(ok.sum()), sd_rho=float(xx[ok].std()), sd_F=float(yy[ok].std()), r=float(stats.pearsonr(xx[ok], yy[ok])[0]))
r_ref, s_ref = OUT["E1"]["pearson_r"], OUT["E1"]["sd_rho"]
def thorndike(s):
    return abs(r_ref) * s / np.sqrt(1 - r_ref ** 2 + r_ref ** 2 * s ** 2)
for lab in r5:
    r5[lab]["predicted_abs_r_range_restriction"] = float(thorndike(r5[lab]["sd_rho"] / s_ref))
s3 = summ(load("mechanism_suite/S3_omega_regime_sweep.json"))
reg_lab = {"adam_lr1e-3_bs128": "Adam b128", "sgd_lr0.03_bs32": "SGD b32", "sgd_lr0.03_bs128": "SGD b128",
           "sgdm_lr0.03_bs32": "SGDM b32", "sgdm_lr0.1_cosine_bs32": "SGDM+cos b32"}
OUT["Identifiability"] = dict(reference=dict(r=r_ref, sd_rho=s_ref), cifar100=r5,
                              optimizer_regimes={reg_lab[k]: dict(r=v["pearson_r"], sd_F=v["forgetting_sd"]) for k, v in s3.items()
                                                 if isinstance(v, dict) and "pearson_r" in v})
fig, ax = plt.subplots(1, 2, figsize=(W * 0.80, 1.72))
a = ax[0]
ss = np.linspace(0.005, 0.1, 200)
a.plot(ss, thorndike(ss / s_ref), color=COL["grey"], lw=1, ls="--", label="Cor. 1 (range restriction)")
a.scatter([s_ref], [abs(r_ref)], color=COL["blue"], s=18, zorder=4, label="CIFAR-10 (E1)")
mk = {"CIFAR-100 contiguous": "s", "CIFAR-100 random": "^", "CIFAR-100 semantic": "D"}
for lab, v in r5.items():
    a.scatter([v["sd_rho"]], [-v["r"]], marker=mk[lab], color=COL["red"], s=14, zorder=4, label=lab)
a.axhline(0, color=COL["dark"], lw=0.5)
a.set_xlabel(r"between-transition s.d. of $\rho$"); a.set_ylabel(r"observed $-r(\rho,F)$")
a.set_title("Coordinate variance"); a.legend(frameon=False, fontsize=5.2, loc="center right"); tag(a, "a")
a = ax[1]
for k, v in OUT["Identifiability"]["optimizer_regimes"].items():
    a.scatter(v["sd_F"], -v["r"], color=COL["red"] if "b128" in k and "SGD" in k else COL["blue"], s=14, zorder=3)
    off = {"Adam b128": (4, -2), "SGD b32": (-26, -9), "SGD b128": (4, -2), "SGDM b32": (-30, 3), "SGDM+cos b32": (-12, 5)}[k]
    a.annotate(k, (v["sd_F"], -v["r"]), fontsize=5.5, xytext=off, textcoords="offset points")
a.set_xlabel(r"s.d. of forgetting across transitions"); a.set_ylabel(r"observed $-r(\rho,F)$")
a.set_xlim(0.044, 0.088); a.set_ylim(0, 0.85)
a.set_title("Outcome variance (optimizer regimes)"); tag(a, "b")
fig.tight_layout(w_pad=1.3); fig.savefig(FIG / "fig_identifiability.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Robustness (S1, S6, S3, S5)
s1 = summ(load("mechanism_suite/S1_rho_sensitivity.json"))
probe_n = [32, 64, 128, 256, 512, 1000]; shuf = [1, 5, 20, 100]
Hm = np.array([[s1[f"n{n}_sh{k}"]["pearson_r"] for k in shuf] for n in probe_n])
s6 = summ(load("mechanism_suite/S6_label_light_omega.json"))
s5 = summ(load("mechanism_suite/S5_pretrained_backbone.json"))["resnet50_imagenet"]
OUT["Robustness"] = dict(S1=dict(grid_min=float(Hm.min()), grid_max=float(Hm.max()), n_settings=int(Hm.size),
                                 linear_probe=float(s1["linear_probe_compatibility"]["pearson_r"]), cka=float(s1["cka"]["pearson_r"]),
                                 historical_full_test=float(s1["historical_full_test_rho"]["pearson_r"])),
                         S6={k: dict(agree=v["corr_with_full_rho"], r_F=v["corr_with_forgetting"]) for k, v in s6.items()
                             if isinstance(v, dict) and "corr_with_full_rho" in v},
                         S3={reg_lab[k]: dict(r=v["pearson_r"], p=v["pearson_p"], r_ci=v["rho_ci"]["ci95"], slope_ci=v["slope_ci"]["ci95"],
                                              loto_neg=v["loto"]["n_negative_slope_folds"], sd_F=v["forgetting_sd"])
                             for k, v in s3.items() if isinstance(v, dict) and "pearson_r" in v},
                         S5=dict(r=s5["pearson_r"], p=s5["pearson_p"], r_ci=s5["rho_ci"]["ci95"],
                                 loto_slopes=[f["slope"] for f in s5["loto"]["folds"]]))
fig, ax = plt.subplots(1, 4, figsize=(W, 1.62), gridspec_kw=dict(width_ratios=[1.35, 1.1, 1.15, 0.95]))
a = ax[0]
im = a.imshow(Hm, cmap="Blues_r", vmin=-0.85, vmax=-0.6, aspect="auto")
for i in range(len(probe_n)):
    for j in range(len(shuf)):
        a.text(j, i, f"{Hm[i, j]:.2f}", ha="center", va="center", fontsize=4.9, color="white" if Hm[i, j] < -0.76 else COL["dark"])
a.set_xticks(range(len(shuf))); a.set_xticklabels(shuf); a.set_yticks(range(len(probe_n))); a.set_yticklabels(probe_n)
a.set_xlabel("shuffled references"); a.set_ylabel("probe examples"); a.set_title("Estimator grid, $r(\\rho,F)$"); tag(a, "a")
a = ax[1]
labs = [2, 5, 10, 25, 50, 100]
a.plot(labs, [s6[f"label_{k}"]["corr_with_full_rho"] for k in labs], "o-", color=COL["blue"], ms=3, label="agreement with full $\\rho$")
a.plot(labs, [-s6[f"label_{k}"]["corr_with_forgetting"] for k in labs], "s-", color=COL["green"], ms=3, label="$-r$ with forgetting")
a.axhline(-s6["rho_kmeans_label_free"]["corr_with_forgetting"], color=COL["red"], ls=":", lw=0.9)
a.text(2.1, 0.0, "label-free", fontsize=5.6, color=COL["red"], va="bottom")
a.set_xscale("log"); a.set_xticks([2, 10, 100]); a.set_xticklabels([2, 10, 100]); a.set_ylim(-0.1, 1.25)
a.set_xlabel("labels per class"); a.set_title("Label budget"); a.legend(frameon=False, fontsize=5.2, loc="upper left", borderaxespad=0.1); tag(a, "b")
a = ax[2]
for i, (k, v) in enumerate(OUT["Robustness"]["S3"].items()):
    lo_, hi_ = v["r_ci"]
    a.errorbar(v["r"], i, xerr=[[v["r"] - lo_], [hi_ - v["r"]]], fmt="o", ms=3, capsize=1.5, elinewidth=0.8,
               color=COL["grey"] if hi_ > 0 else COL["blue"])
a.axvline(0, color=COL["dark"], lw=0.6)
a.set_yticks(range(len(OUT["Robustness"]["S3"]))); a.set_yticklabels(list(OUT["Robustness"]["S3"].keys()), fontsize=6)
a.invert_yaxis(); a.set_xlabel(r"$r(\rho,F)$, task-cluster 95% CI"); a.set_title("Optimizer regime"); tag(a, "c")
a = ax[3]
sl = OUT["Robustness"]["S5"]["loto_slopes"]
a.scatter(sl, range(len(sl)), s=10, color=COL["purple"])
a.axvline(0, color=COL["dark"], lw=0.6)
a.set_yticks(range(len(sl))); a.set_yticklabels([f"held-out {i}" for i in range(len(sl))], fontsize=5.2)
a.invert_yaxis(); a.set_xlabel("LOTO slope"); a.set_title("Pretrained ResNet-50"); tag(a, "d")
fig.tight_layout(w_pad=0.9); fig.savefig(FIG / "fig_robustness.pdf"); plt.close(fig)

# ---------------------------------------------------------------- E4 replay
e4 = load("succession/e4_summary.json")
r4 = pd.DataFrame(e4["rows"])
OUT["E4"] = dict(n=len(r4), setting=e4["setting"], threshold=e4["threshold"], raw_Mstar=e4["raw_Mstar"],
                 initial_efficiency=e4["initial_efficiency"],
                 r_eta_forgettingM0=float(stats.pearsonr(r4["eta_M0_to_M25"], r4["forgetting_M0"])[0]),
                 mean_forgetting_M0=float(r4["forgetting_M0"].mean()), n_repaired=int((r4["M_star_fix"] <= 1000).sum()),
                 max_Mstar=int(r4["M_star_fix"].max()), median_Mstar=float(r4["M_star_fix"].median()))
fig, ax = plt.subplots(1, 3, figsize=(W, 1.6))
for a, ycol, col, t, yl in [(ax[0], "M_star", COL["grey"], f"global threshold ($r={e4['raw_Mstar']['pearson_r']:.3f}$)", r"minimum buffer $M^\ast$"),
                            (ax[1], "eta_M0_to_M25", COL["blue"], f"initial efficiency ($r={e4['initial_efficiency']['pearson_r']:.3f}$)", r"efficiency $\eta$ (per sample)")]:
    a.scatter(r4["rho_pre"], r4[ycol], s=11, color=col, zorder=3)
    ols_band(a, r4["rho_pre"].values, r4[ycol].values, col)
    a.set_xlabel(r"$\rho_{\mathrm{pre}}$"); a.set_ylabel(yl); a.set_title(t)
a = ax[2]
a.scatter(r4["forgetting_M0"], r4["eta_M0_to_M25"], s=11, color=COL["green"], zorder=3)
ols_band(a, r4["forgetting_M0"].values, r4["eta_M0_to_M25"].values, COL["green"])
a.set_xlabel("forgetting without replay"); a.set_ylabel(r"efficiency $\eta$")
a.set_title(f"Prop. 2: displaced $\\Rightarrow$ efficient ($r={OUT['E4']['r_eta_forgettingM0']:.3f}$)")
for i, t in enumerate("abc"):
    tag(ax[i], t)
fig.tight_layout(w_pad=1.0); fig.savefig(FIG / "fig_replay.pdf"); plt.close(fig)

# ---------------------------------------------------------------- E5 intervention sign test
e5 = load("succession/e5_summary.json")["results"]
b7 = summ(load("mechanism_suite/B7_gated_intervention.json"))
OUT["E5"] = dict(benchmark={m: {ds: {k: v for k, v in d.items()} for ds, d in dd.items()} for m, dd in e5.items()},
                 gated={k: v for k, v in b7.items() if isinstance(v, dict) and "ACC" in v},
                 paired=b7["paired_uncertainty"])
meth = [("finetune", "Fine-tune"), ("ewc", "EWC"), ("agem", "A-GEM"), ("rho_ewc", r"$\rho$-EWC"),
        ("rho_ewc_replay", r"$\rho$-EWC+rep."), ("derpp", "DER++")]
fig, ax = plt.subplots(1, 3, figsize=(W, 1.62), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1.0]))
for a, ds, t in [(ax[0], "cifar10", "Split-CIFAR-10 (task-IL)"), (ax[1], "cifar100", "Split-CIFAR-100 (task-IL)")]:
    accs = [e5[m][ds]["ACC_mean"] for m, _ in meth]; sds = [e5[m][ds]["ACC_sd"] for m, _ in meth]
    cols = [COL["grey"], COL["grey"], COL["grey"], COL["sky"], COL["blue"], COL["dark"]]
    a.bar(range(len(meth)), accs, yerr=sds, color=cols, width=0.65, error_kw=dict(lw=0.7, capsize=1.2))
    a.set_xticks(range(len(meth))); a.set_xticklabels([l for _, l in meth], rotation=35, ha="right", fontsize=5.8)
    a.set_ylim(0, 1.0); a.set_ylabel("ACC"); a.set_title(t)
a = ax[2]
pu = b7["paired_uncertainty"]
items = [("omega_vs_ewc_ACC", r"$\Omega$-gated EWC-DR vs EWC-DR: $\Delta$ACC"), ("omega_vs_ewc_BWT", r"$\Omega$-gated vs EWC-DR: $\Delta$BWT"),
         ("kappa_vs_mean_replay_ACC", r"$\kappa$-gated vs matched replay: $\Delta$ACC")]
for i, (k, lab) in enumerate(items):
    v = pu[k]
    a.errorbar(100 * v["mean_diff"], i, xerr=[[100 * (v["mean_diff"] - v["ci95"][0])], [100 * (v["ci95"][1] - v["mean_diff"])]],
               fmt="o", ms=3, capsize=1.5, elinewidth=0.8, color=COL["blue"] if "omega" in k else COL["red"])
a.axvline(0, color=COL["dark"], lw=0.6)
a.set_yticks(range(len(items))); a.set_yticklabels([l for _, l in items], fontsize=5.4); a.invert_yaxis()
a.set_xlabel("paired difference (pp), 95% CI, 3 seeds"); a.set_title("Gated sign tests")
for i, t in enumerate("abc"):
    tag(ax[i], t)
fig.tight_layout(w_pad=0.8); fig.savefig(FIG / "fig_intervention.pdf"); plt.close(fig)

# ---------------------------------------------------------------- remaining frozen numbers
s1r = pd.DataFrame([{"seed": r["seed"], "task_a": r["task_a"], "task_b": r["task_b"], "omega": 1 - r["rho_grid"]["n256_sh20"]}
                     for r in load("mechanism_suite/S1_rho_sensitivity.json")["rows"]])
b1r = pd.DataFrame([r for r in load("mechanism_suite/B1_habitat_interaction.json")["rows"] if r["regime"] == "sgd"])
brg = s1r.merge(b1r[["seed", "task_a", "task_b", "H_hab", "probe_loss_B"]], on=["seed", "task_a", "task_b"])
brg = brg.groupby(["task_a", "task_b"], as_index=False)[["omega", "H_hab", "probe_loss_B"]].mean()
def within_incoming(df, xcol, ycol, gcol="task_b"):
    dx = df[xcol] - df.groupby(gcol)[xcol].transform("mean"); dy = df[ycol] - df.groupby(gcol)[ycol].transform("mean")
    return dx.values, dy.values
bridge = {}
for tgt in ["probe_loss_B", "H_hab"]:
    dx, dy = within_incoming(brg, "omega", tgt)
    bridge[tgt] = dict(r=float(np.corrcoef(dx, dy)[0, 1]), ci=cluster_boot_r(dx, dy, brg["task_b"].values), n=len(brg))
sem = pd.DataFrame(load("mechanism_suite/R5_sem_cifar100.json")["rows"])
semm = sem.groupby(["task_a", "task_b"], as_index=False)[["omega", "probe_loss_B"]].mean()
dxs, dys = within_incoming(semm, "omega", "probe_loss_B")
bridge["semantic_cifar100_probe_loss"] = dict(r=float(np.corrcoef(dxs, dys)[0, 1]), n=len(semm))
OUT["Bridge"] = bridge
fig, ax = plt.subplots(1, 2, figsize=(W * 0.62, 1.62))
for a, (dx, dy, t, col) in zip(ax, [(*within_incoming(brg, "omega", "probe_loss_B"), f"Split-CIFAR-10 ($r={bridge['probe_loss_B']['r']:.2f}$)", COL["blue"]),
                                    (dxs, dys, f"semantic CIFAR-100 ($r={bridge['semantic_cifar100_probe_loss']['r']:.2f}$)", COL["grey"])]):
    a.scatter(dx, dy, s=8, color=col, zorder=3); ols_band(a, dx, dy, col)
    a.axhline(0, color=COL["dark"], lw=0.4); a.axvline(0, color=COL["dark"], lw=0.4)
    a.set_xlabel(r"$\Omega$ (demeaned by incoming task)"); a.set_ylabel("probe loss (demeaned)"); a.set_title(t, fontsize=7)
tag(ax[0], "a"); tag(ax[1], "b")
fig.tight_layout(w_pad=1.0); fig.savefig(FIG / "fig_bridge.pdf"); plt.close(fig)
pr3 = pd.concat([pd.read_csv(D / f"protection_suite/seed{s}_p3_protection.csv") for s in range(3)], ignore_index=True)
prot = {}
for mth in ["derpp", "ewc_dr"]:
    g = pr3[pr3["method"] == mth].groupby("pair", as_index=False).agg(omega=("omega", "mean"), gain=("protection_gain", "mean"))
    prot[mth] = dict(r=float(stats.pearsonr(g["omega"], g["gain"])[0]), p=float(stats.pearsonr(g["omega"], g["gain"])[1]),
                     spearman=float(stats.spearmanr(g["omega"], g["gain"]).statistic), mean_gain=float(pr3[pr3["method"] == mth]["protection_gain"].mean()),
                     positive_pairs=int((g["gain"] > 0).sum()), n_pairs=len(g))
OUT["Protection"] = dict(derpp_r=prot["derpp"]["r"], derpp_mean_gain=prot["derpp"]["mean_gain"],
                         ewcdr_r=prot["ewc_dr"]["r"], ewcdr_mean_gain=prot["ewc_dr"]["mean_gain"],
                         positive_pairs=prot["derpp"]["positive_pairs"], n_pairs=prot["derpp"]["n_pairs"], detail=prot)
ch = load("transition_suite/E6/e6_summary.json")
OUT["Chesson"] = dict(n_pairs=ch["n_pairs"], ND_FD_R2=ch["ND_FD"]["R2"], ND_FD_LOO_R2=ch["ND_FD"]["LOO_R2"])
s4b = summ(load("mechanism_suite/S4b_classil_omega.json"))
OUT["ClassIL_S4b"] = {k: v for k, v in s4b.items() if not isinstance(v, (list,))} if isinstance(s4b, dict) else s4b

(D / "derived").mkdir(exist_ok=True)
json.dump(OUT, open(D / "derived" / "derived_revision_stats.json", "w"), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ["E1", "E2", "E3_natural", "E3_domain", "E3_suite", "E4", "Bridge"]}, indent=1, default=float)[:7000])
