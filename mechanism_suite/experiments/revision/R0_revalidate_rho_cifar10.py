#!/usr/bin/env python3
"""R0 — Revalidate the pre-hoc compatibility law with task-aware inference.

Addresses three concrete concerns without changing the historical E1 result:
  1) arbitrary local binary-label alignment in rho_pre;
  2) pseudo-replication across directed task pairs sharing the same task identities;
  3) E2 pair-wise CV leakage of task identities.

Outputs raw historical rho_pre, permutation-invariant rho_PI, a frozen-linear-probe
control, forgetting, exact 5-task QAP/randomisation inference, and leave-one-task-out
forecasting.  The network/training protocol remains Split-CIFAR-10, ResNet-18,
20 Adam epochs/task, 3 seeds by default.
"""
from __future__ import annotations
import copy, itertools, os, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *  # noqa
from slt_confirmatory import json_dump_safe, run_metadata
from slt_revision import (rho_pre_permutation_invariant, frozen_probe_compatibility,
                          pair_means, qap_task_label_permutation,
                          leave_one_task_out_forecast, pearson_safe, spearman_safe)

SMOKE=os.environ.get('SLT_SMOKE','0')=='1'
SEEDS=int(os.environ.get('SLT_R0_SEEDS','1' if SMOKE else '3'))
BASE_SEED=int(os.environ.get('SLT_BASE_SEED','42'))
EPOCHS=int(os.environ.get('SLT_R0_EPOCHS','2' if SMOKE else '20'))
MAX_PAIRS=int(os.environ.get('SLT_MAX_PAIRS','3')) if SMOKE else None
PROBE_STEPS=int(os.environ.get('SLT_R0_PROBE_STEPS','30' if SMOKE else '200'))


def run():
    Xtr,ytr,Xte,yte=load_np('cifar10')
    tr=make_tasks(Xtr,ytr,SPLITS['cifar10']); te=make_tasks(Xte,yte,SPLITS['cifar10'])
    pairs=list(itertools.permutations(range(5),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs
    rows=[]; t0=time.time()
    for si in range(SEEDS):
        seed=BASE_SEED+si; set_seed(seed)
        # Train each resident once; clone it for each invader.
        residents={}
        for a in sorted(set(a for a,_ in pairs)):
            m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE)
            train_task(m,a,*tr[a],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
            residents[a]=m
        for a,b in pairs:
            mA=residents[a]; RAA=accuracy(mA,a,*te[a],arch='resnet18',dataset='cifar10')
            raw=rho_pre(mA,a,*te[b],arch='resnet18',dataset='cifar10',n_samples=None,seed=seed+100*a+b)
            pi=rho_pre_permutation_invariant(mA,a,*te[b],arch='resnet18',dataset='cifar10',n_classes=2,
                                             n_samples=None,n_shuffle=20,seed=seed+100*a+b)
            # E1.5 consistency under the corrected label-permutation-invariant score.
            mB=residents[b]
            pi_rev=rho_pre_permutation_invariant(mB,b,*te[a],arch='resnet18',dataset='cifar10',n_classes=2,
                                                 n_samples=None,n_shuffle=20,seed=seed+700+b*10+a)
            pi_full=0.5*(pi+pi_rev)
            probe,probe_acc=frozen_probe_compatibility(mA,*tr[b],*te[b],arch='resnet18',dataset='cifar10',
                                                       n_classes=2,steps=PROBE_STEPS,seed=seed+1000+a*10+b)
            m=copy.deepcopy(mA)
            train_task(m,b,*tr[b],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
            RBA=accuracy(m,a,*te[a],arch='resnet18',dataset='cifar10'); fgt=pairwise_forgetting(RAA,RBA)
            row={'seed':seed,'task_a':a,'task_b':b,'rho_pre_raw':raw,'rho_pre_pi':pi,'rho_rev_pi':pi_rev,'rho_full_pi':pi_full,
                 'probe_compatibility':probe,'probe_acc_B':probe_acc,'R_AA':RAA,'R_BA':RBA,'forgetting':fgt}
            rows.append(row)
            print(f"s{seed} T{a}->T{b}: rho={raw:+.3f} rho_PI={pi:+.3f} full_PI={pi_full:+.3f} probe={probe:+.3f} F={fgt:+.3f}",flush=True)
    keys=['rho_pre_raw','rho_pre_pi','rho_rev_pi','rho_full_pi','probe_compatibility','R_AA','R_BA','forgetting']
    means=pair_means(rows,keys)
    analysis={}
    for k in ['rho_pre_raw','rho_pre_pi','probe_compatibility']:
        x=[r[k] for r in means]; y=[r['forgetting'] for r in means]
        rp,pp=pearson_safe(x,y); rs,ps=spearman_safe(x,y)
        analysis[k]={'pearson_r':rp,'pearson_pair_p':pp,'spearman_r':rs,'spearman_pair_p':ps,
                     'task_label_permutation':qap_task_label_permutation(means,k,n_tasks=5,seed=BASE_SEED),
                     'LOTO_incoming':leave_one_task_out_forecast(means,k,n_tasks=5,incoming_only=True),
                     'LOTO_all_incident':leave_one_task_out_forecast(means,k,n_tasks=5,incoming_only=False)}
    prepi=[r['rho_pre_pi'] for r in means]; fullpi=[r['rho_full_pi'] for r in means]
    rr,pv=pearson_safe(prepi,fullpi); analysis['E1p5_PI_consistency']={'pearson_r_pre_vs_full':rr,'pair_level_p':pv,'mae_pre_vs_full':float(np.mean(np.abs(np.asarray(prepi)-np.asarray(fullpi))))}
    out={'rows':rows,'pair_means':means,'analysis':analysis,
         '_meta':run_metadata(ROOT,{'experiment':'R0','seeds':SEEDS,'epochs':EPOCHS,'smoke':SMOKE,
                                    'purpose':'label-invariant rho + PI E1.5 consistency + task-aware inference + LOTO forecasting',
                                    'elapsed_s':time.time()-t0})}
    json_dump_safe(out,'r0_revalidate_rho_results.json')

    fig,ax=plt.subplots(1,3,figsize=(14,4.2))
    for a,k,title in zip(ax,['rho_pre_raw','rho_pre_pi','probe_compatibility'],
                         ['Historical pre-hoc score','Permutation-invariant pre-hoc score','Frozen-probe compatibility']):
        x=np.array([r[k] for r in means]); y=np.array([r['forgetting'] for r in means]); rr,_=pearson_safe(x,y)
        a.scatter(x,y,s=50,edgecolor='k',linewidth=.4)
        if len(x)>=2:
            c=np.polyfit(x,y,1); xx=np.linspace(x.min(),x.max(),100); a.plot(xx,np.polyval(c,xx),lw=2)
        a.set_xlabel(title); a.set_ylabel('Forgetting' if a is ax[0] else ''); a.set_title(f'r = {rr:.3f}'); a.grid(alpha=.25)
    plt.tight_layout(); plt.savefig('r0_compatibility_revalidation.png',dpi=220,bbox_inches='tight'); plt.close()
    print(json.dumps(analysis,indent=2) if False else 'saved r0_revalidate_rho_results.json')

if __name__=='__main__': run()
