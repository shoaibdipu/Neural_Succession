#!/usr/bin/env python3
"""R5 — Split-CIFAR-100 directional pre-hoc compatibility at scale.

Runs the exact task-level question on 10 task identities / 90 ordered transitions.
The permutation-invariant score uses Hungarian label matching for the 10 local
classes, eliminating arbitrary class-index alignment.  Task-label permutation
inference and leave-one-task-out forecasting are computed on pair means.
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
                          leave_one_task_out_forecast, pearson_safe)

SMOKE=os.environ.get('SLT_SMOKE','0')=='1'
SEEDS=int(os.environ.get('SLT_R5_SEEDS','1'))
BASE_SEED=int(os.environ.get('SLT_BASE_SEED','42'))
EPOCHS=int(os.environ.get('SLT_R5_EPOCHS','2' if SMOKE else '20'))
PROBE_STEPS=int(os.environ.get('SLT_R5_PROBE_STEPS','30' if SMOKE else '250'))
MAX_PAIRS=int(os.environ.get('SLT_MAX_PAIRS','3')) if SMOKE else None
N_PERM=int(os.environ.get('SLT_R5_PERMUTATIONS','2000' if SMOKE else '20000'))


FRESH=os.environ.get('SLT_R5_FRESH_PARTITION','0')=='1'
FRESH_SEED=int(os.environ.get('SLT_R5_PARTITION_SEED','20260916'))


def fresh_partition(n_classes=100, n_tasks=10, seed=FRESH_SEED):
    """A CIFAR-100 grouping that no prior analysis has seen.

    Every result in this project -- slide runs, first campaign, this revision --
    used SPLITS['cifar100'], the standard contiguous 10x10 grouping. Those
    results GENERATED the hypotheses being tested, so re-measuring on the same
    task identities is discovery, not confirmation.

    This regroups all 100 classes under one fixed seed recorded in the output.
    Reproducible, but not available when any hypothesis was formed. The seed
    must be frozen BEFORE the run -- changing it after seeing a result forfeits
    the point.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_classes)
    per = n_classes // n_tasks
    return [sorted(perm[i*per:(i+1)*per].tolist()) for i in range(n_tasks)]


def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('cifar100')
    splits = fresh_partition() if FRESH else SPLITS['cifar100']
    if FRESH:
        print(f"FRESH PARTITION (seed={FRESH_SEED}) -- confirmation set, not discovery",flush=True)
        for i,g in enumerate(splits): print(f"  task {i}: {g}",flush=True)
    tr=make_tasks(Xtr,ytr,splits); te=make_tasks(Xte,yte,splits)
    pairs=list(itertools.permutations(range(10),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs; rows=[]
    for si in range(SEEDS):
        seed=BASE_SEED+si; set_seed(seed); residents={}
        for a in sorted(set(a for a,_ in pairs)):
            m=make_model('resnet18',10,10,scenario='task',dataset='cifar100').to(DEVICE)
            train_task(m,a,*tr[a],arch='resnet18',dataset='cifar100',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
            residents[a]=m
        for a,b in pairs:
            mA=residents[a]; RAA=accuracy(mA,a,*te[a],arch='resnet18',dataset='cifar100')
            raw=rho_pre(mA,a,*te[b],arch='resnet18',dataset='cifar100',n_samples=None,seed=seed+100*a+b)
            pi=rho_pre_permutation_invariant(mA,a,*te[b],arch='resnet18',dataset='cifar100',n_classes=10,n_samples=None,n_shuffle=20,seed=seed+100*a+b)
            probe,probe_acc=frozen_probe_compatibility(mA,*tr[b],*te[b],arch='resnet18',dataset='cifar100',n_classes=10,
                                                       max_train=3000,max_test=1000,steps=PROBE_STEPS,seed=seed+1000+a*10+b)
            m=copy.deepcopy(mA)
            train_task(m,b,*tr[b],arch='resnet18',dataset='cifar100',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
            RBA=accuracy(m,a,*te[a],arch='resnet18',dataset='cifar100'); fgt=RAA-RBA
            rows.append({'seed':seed,'task_a':a,'task_b':b,'rho_pre_raw':raw,'rho_pre_pi':pi,'probe_compatibility':probe,
                         'probe_acc_B':probe_acc,'R_AA':RAA,'R_BA':RBA,'forgetting':fgt})
            print(f"s{seed} T{a}->T{b}: rhoPI={pi:+.3f} probe={probe:+.3f} F={fgt:+.3f}",flush=True)
    means=pair_means(rows,['rho_pre_raw','rho_pre_pi','probe_compatibility','R_AA','R_BA','forgetting']); analysis={}
    # Full task-aware analysis requires all 90 directed transitions.
    full=len(means)==90
    for k in ['rho_pre_raw','rho_pre_pi','probe_compatibility']:
        rr,pp=pearson_safe([r[k] for r in means],[r['forgetting'] for r in means]); z={'pearson_r':rr,'pair_level_p':pp}
        if full:
            z['task_label_permutation']=qap_task_label_permutation(means,k,n_tasks=10,n_perm=N_PERM,seed=BASE_SEED,exact_max_tasks=8)
            z['LOTO_incoming']=leave_one_task_out_forecast(means,k,n_tasks=10,incoming_only=True)
        analysis[k]=z
    out={'rows':rows,'pair_means':means,'analysis':analysis,
         '_meta':run_metadata(ROOT,{'experiment':'R5','seeds':SEEDS,'epochs':EPOCHS,'n_ordered_pairs':len(means),
                                    'full_90_pairs':full,'smoke':SMOKE,'elapsed_s':time.time()-t0,'fresh_partition':FRESH,'partition_seed':(FRESH_SEED if FRESH else None),'partition_classes':(splits if FRESH else 'SPLITS[cifar100] standard'),'preregistered_criterion':'CONFIRMATORY (fresh partition only): r(rho_pre,F) <= -0.50 on pair means, sign consistent under leave-one-task-out in >=8/10 folds, cluster-bootstrap CI over task identities excluding zero. Frozen before the run; not to be revised after.'})}
    json_dump_safe(out,'r5_cifar100_directional_rho_results.json')
    if means:
        plt.figure(figsize=(6.2,4.8)); x=np.array([r['rho_pre_pi'] for r in means]); y=np.array([r['forgetting'] for r in means]); rr,_=pearson_safe(x,y)
        plt.scatter(x,y,s=34,alpha=.8,edgecolor='k',linewidth=.3); c=np.polyfit(x,y,1); xx=np.linspace(x.min(),x.max(),100); plt.plot(xx,np.polyval(c,xx),lw=2)
        plt.xlabel('Permutation-invariant pre-hoc compatibility'); plt.ylabel('Forgetting'); plt.title(f'Split-CIFAR-100 directed transitions | r={rr:.3f}'); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig('r5_cifar100_rho_forgetting.png',dpi=220,bbox_inches='tight'); plt.close()
    print('saved r5_cifar100_directional_rho_results.json')

if __name__=='__main__': main()
