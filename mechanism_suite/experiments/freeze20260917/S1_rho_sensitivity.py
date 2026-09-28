#!/usr/bin/env python3
"""S1 — train-only compatibility sensitivity + uncertainty.

Reviewer coverage:
  * exact supervision/data budget for rho;
  * shuffled-label normalization sensitivity;
  * train-only linear-probe and label-free CKA alternatives;
  * task-cluster CIs;
  * partial correlations controlling resident and incoming-task difficulty.
Benchmark test labels are never used to construct a pre-hoc metric.
"""
from __future__ import annotations
import copy,itertools,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_revision import rho_pre_permutation_invariant, frozen_probe_compatibility, partial_corr_residualized
from slt_freeze import *
from slt_confirmatory import run_metadata, stratified_train_val

EXP='S1_rho_sensitivity'; SMOKE=env_bool('SLT_SMOKE')
SEEDS=int(os.environ.get('SLT_S1_SEEDS','1' if SMOKE else '3')); EPOCHS=int(os.environ.get('SLT_S1_EPOCHS','2' if SMOKE else '20'))
PROBES=[int(x) for x in os.environ.get('SLT_S1_PROBES','32,128' if SMOKE else '32,64,128,256,512,1000').split(',')]
SHUFFLES=[int(x) for x in os.environ.get('SLT_S1_SHUFFLES','1,5' if SMOKE else '1,5,20,100').split(',')]
BASE=int(os.environ.get('SLT_BASE_SEED','42')); MAX_PAIRS=2 if SMOKE else None
NBOOT=200 if SMOKE else 5000


def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('cifar10'); tr=make_tasks(Xtr,ytr,SPLITS['cifar10']); te=make_tasks(Xte,yte,SPLITS['cifar10'])
    pairs=list(itertools.permutations(range(5),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs; rows=[]
    for si in range(SEEDS):
        seed=BASE+si; set_seed(seed); residents={}
        for a in range(5):
            m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE)
            train_task(m,a,*tr[a],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4); residents[a]=m
        solo={t:accuracy(residents[t],t,*te[t],arch='resnet18',dataset='cifar10') for t in range(5)}
        for a,b in pairs:
            mA=residents[a]; RAA=solo[a]; RBB=solo[b]; sens={}
            # rho uses a class-balanced subset of B TRAIN only. Test labels are never seen.
            for n in PROBES:
                Xp,yp,_=train_probe_pool(*tr[b],n=n,seed=seed+10000+a*100+b+n)
                for sh in SHUFFLES:
                    k=f'n{n}_sh{sh}'
                    sens[k]=rho_pre_permutation_invariant(mA,a,Xp,yp,arch='resnet18',dataset='cifar10',n_classes=2,n_samples=None,n_shuffle=sh,seed=seed+20000+a*100+b+sh)
            # Historical bridge only: reproduces the earlier paper-style full B-test-set rho.
            # It is NEVER used as a pre-hoc training/selection quantity in this campaign.
            historical_test_rho=rho_pre_permutation_invariant(mA,a,*te[b],arch='resnet18',dataset='cifar10',n_classes=2,n_samples=None,n_shuffle=20,seed=seed+25000+a*100+b)
            # Linear probe: fit and evaluate on disjoint subsets of B TRAIN.
            (Bfit,yfit),(Bev,yev),_,_=stratified_train_val(*tr[b],val_frac=.50,seed=seed+3000+a*10+b)
            probe_score,probe_acc=frozen_probe_compatibility(mA,Bfit,yfit,Bev,yev,arch='resnet18',dataset='cifar10',n_classes=2,steps=50 if SMOKE else 200,seed=seed+4000+a*10+b,max_train=min(2000,len(Bfit)),max_test=min(2000,len(Bev)))
            # Label-free CKA on TRAIN probes only.
            XA,ya,_=train_probe_pool(*tr[a],n=min(512,len(tr[a][0])),seed=seed+5000+a)
            XB,yb,_=train_probe_pool(*tr[b],n=min(512,len(tr[b][0])),seed=seed+6000+b)
            FA=feature_matrix(mA,XA,arch='resnet18',dataset='cifar10',max_n=len(XA)).cpu().numpy(); FB=feature_matrix(mA,XB,arch='resnet18',dataset='cifar10',max_n=len(XB)).cpu().numpy()
            cka=linear_cka_np(FA,FB)
            m=copy.deepcopy(mA); train_task(m,b,*tr[b],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
            RBA=accuracy(m,a,*te[a],arch='resnet18',dataset='cifar10'); Fgt=RAA-RBA
            rows.append({'seed':seed,'task_a':a,'task_b':b,'R_AA':RAA,'R_BB':RBB,'R_BA':RBA,'forgetting':Fgt,'rho_grid':sens,'historical_full_test_rho':historical_test_rho,'linear_probe_compatibility':probe_score,'linear_probe_acc':probe_acc,'cka':cka})
            print(seed,a,b,'F',round(Fgt,3),flush=True)
    summary={}
    metric_names=[f'n{n}_sh{sh}' for n in PROBES for sh in SHUFFLES]+['historical_full_test_rho','linear_probe_compatibility','cka']
    for key in metric_names:
        tmp=[]
        for a,b in pairs:
            rs=[r for r in rows if r['task_a']==a and r['task_b']==b]
            if not rs: continue
            vals=[r['rho_grid'][key] for r in rs] if key.startswith('n') else [r[key] for r in rs]
            tmp.append({'task_a':a,'task_b':b,'x':float(np.mean(vals)),'forgetting':float(np.mean([r['forgetting'] for r in rs])),'R_AA':float(np.mean([r['R_AA'] for r in rs])),'R_BB':float(np.mean([r['R_BB'] for r in rs]))})
        r,p=pearson_safe([z['x'] for z in tmp],[z['forgetting'] for z in tmp])
        pc=partial_corr_residualized([z['x'] for z in tmp],[z['forgetting'] for z in tmp],np.asarray([[z['R_AA'],z['R_BB']] for z in tmp]))
        summary[key]={'pearson_r':r,'p':p,'partial_r_controlling_R_AA_R_BB':pc.get('partial_r'),'n_pairs':len(tmp),
                      'pearson_task_cluster_ci':bootstrap_pearson_ci(tmp,'x','forgetting',n_boot=NBOOT,seed=BASE+11),
                      'partial_task_cluster_ci':bootstrap_partial_corr_ci(tmp,'x','forgetting',['R_AA','R_BB'],n_boot=NBOOT,seed=BASE+12)}
    out={'rows':rows,'summary':summary,'protocol':{'rho_supervision':'class-balanced labeled subset of incoming-task TRAIN only','benchmark_test_labels_used_for_primary_prehoc_metrics':False,'historical_bridge_uses_full_test_labels':True,'historical_bridge_role':'descriptive bridge to legacy headline only; excluded from model selection and confirmation','probes':PROBES,'shuffle_draws':SHUFFLES,'linear_probe':'fit/eval split within B train','cka':'label-free, train probes only','difficulty_controls':['R_AA','R_BB'],'class_balance':'fixed by task construction','primary_purpose':'sensitivity only; no best variant selected'},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}
    p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
