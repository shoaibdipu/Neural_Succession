#!/usr/bin/env python3
"""S6 — label-light / label-free pre-hoc compatibility pilot.

This is explicitly exploratory reviewer coverage, not a redefinition of Omega.
It compares the frozen full-label train-probe rho with smaller labeled budgets,
a label-free KMeans pseudo-label analogue, and label-free CKA.
"""
from __future__ import annotations
import copy,itertools,os,sys,time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_revision import (rho_pre_permutation_invariant, task_predictions,
                          permutation_invariant_error_from_predictions)
from slt_freeze import *
from slt_confirmatory import run_metadata

EXP='S6_label_light_omega'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_S6_SEEDS','1' if SMOKE else '3')); EPOCHS=int(os.environ.get('SLT_S6_EPOCHS','2' if SMOKE else '20')); NBOOT=200 if SMOKE else 5000
BUDGET_PER_CLASS=[2,5] if SMOKE else [2,5,10,25,50,100]
MAX_PAIRS=2 if SMOKE else None


def rho_from_surrogate_labels(pred,labels,K,seed,n_shuffle=20):
    e=permutation_invariant_error_from_predictions(pred,labels,K); rng=np.random.RandomState(seed); sh=[]
    for _ in range(n_shuffle):
        yp=np.asarray(labels).copy(); rng.shuffle(yp); sh.append(permutation_invariant_error_from_predictions(pred,yp,K))
    return float(1-np.sqrt(e/(float(np.mean(sh))+1e-8)))


def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('cifar10'); tr=make_tasks(Xtr,ytr,SPLITS['cifar10']); te=make_tasks(Xte,yte,SPLITS['cifar10']); pairs=list(itertools.permutations(range(5),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs; rows=[]
    for si in range(SEEDS):
        seed=BASE+si; set_seed(seed); residents={}
        for a in range(5):
            m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); train_task(m,a,*tr[a],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4); residents[a]=m
        for a,b in pairs:
            mA=residents[a]; RAA=accuracy(mA,a,*te[a],arch='resnet18',dataset='cifar10')
            Xfull,yfull,_=train_probe_pool(*tr[b],n=min(512,len(tr[b][0])),seed=seed+1000+a*10+b)
            rho_full=rho_pre_permutation_invariant(mA,a,Xfull,yfull,arch='resnet18',dataset='cifar10',n_classes=2,n_shuffle=20,seed=seed+2000+a*10+b)
            label_light={}
            for q in BUDGET_PER_CLASS:
                Xp,yp,_=train_probe_pool(*tr[b],n=min(2*q,len(tr[b][0])),seed=seed+3000+a*100+b*10+q)
                label_light[str(q)]=rho_pre_permutation_invariant(mA,a,Xp,yp,arch='resnet18',dataset='cifar10',n_classes=2,n_shuffle=20,seed=seed+4000+a*100+b*10+q)
            # Label-free pseudo-rho: cluster frozen B features; compare resident-head predictions to cluster IDs.
            F_B=feature_matrix(mA,Xfull,arch='resnet18',dataset='cifar10',max_n=len(Xfull)).cpu().numpy(); pseudo=KMeans(n_clusters=2,n_init=20,random_state=seed+5000+a*10+b).fit_predict(F_B)
            pred=task_predictions(mA,a,Xfull,arch='resnet18',dataset='cifar10'); rho_kmeans=rho_from_surrogate_labels(pred,pseudo,2,seed+6000+a*10+b)
            XA,ya,_=train_probe_pool(*tr[a],n=min(512,len(tr[a][0])),seed=seed+7000+a)
            F_A=feature_matrix(mA,XA,arch='resnet18',dataset='cifar10',max_n=len(XA)).cpu().numpy(); cka=linear_cka_np(F_A,F_B)
            m=copy.deepcopy(mA); train_task(m,b,*tr[b],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4); RBA=accuracy(m,a,*te[a],arch='resnet18',dataset='cifar10')
            rows.append({'seed':seed,'task_a':a,'task_b':b,'rho_full_trainprobe':rho_full,'rho_label_light':label_light,'rho_kmeans_label_free':rho_kmeans,'cka_label_free':cka,'forgetting':RAA-RBA})
            print(seed,a,b,'full',round(rho_full,3),'kmeans',round(rho_kmeans,3),'F',round(RAA-RBA,3),flush=True)
    # Aggregate pairs and compare each approximation with full rho and with forgetting.
    methods=['rho_kmeans_label_free','cka_label_free']+[f'label_{q}' for q in BUDGET_PER_CLASS]; summary={}
    pair=[]
    for a,b in pairs:
        z=[r for r in rows if r['task_a']==a and r['task_b']==b]
        if not z: continue
        d={'task_a':a,'task_b':b,'rho_full':float(np.mean([r['rho_full_trainprobe'] for r in z])),'forgetting':float(np.mean([r['forgetting'] for r in z])),'rho_kmeans_label_free':float(np.mean([r['rho_kmeans_label_free'] for r in z])),'cka_label_free':float(np.mean([r['cka_label_free'] for r in z]))}
        for q in BUDGET_PER_CLASS: d[f'label_{q}']=float(np.mean([r['rho_label_light'][str(q)] for r in z]))
        pair.append(d)
    for k in methods:
        rf,_=pearson_safe([z[k] for z in pair],[z['rho_full'] for z in pair]); rr,rp=pearson_safe([z[k] for z in pair],[z['forgetting'] for z in pair]); summary[k]={'corr_with_full_rho':rf,'corr_with_forgetting':rr,'p_forgetting':rp,'forgetting_ci':bootstrap_pearson_ci(pair,k,'forgetting',n_boot=NBOOT,seed=BASE+51)}
    out={'rows':rows,'pair_means':pair,'summary':summary,'protocol':{'full_rho_reference':'512 balanced labels from B train only','label_budgets_per_class':BUDGET_PER_CLASS,'label_free_kmeans':'KMeans on frozen resident-backbone B features; cluster IDs used only as pseudo-label diagnostic','cka':'label-free representation control','status':'exploratory reviewer coverage; no replacement definition selected post hoc'},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}; p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
