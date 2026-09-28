#!/usr/bin/env python3
"""R1 — Mutual coexistence and threshold robustness.

Extends the completed 84-transition E3 design by requiring *both* populations to
persist: resident Task A retains performance and invader Task B establishes relative
to a B-only reference.  This directly repairs the one-sided "90% A retention"
definition while preserving the original controlled transform spectrum.
"""
from __future__ import annotations
import copy, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *  # noqa
from slt_confirmatory import json_dump_safe, run_metadata
from slt_revision import rho_pre_permutation_invariant, pearson_safe

SMOKE=os.environ.get('SLT_SMOKE','0')=='1'
EPOCHS=int(os.environ.get('SLT_R1_EPOCHS','2' if SMOKE else '20'))
N_SEEDS=int(os.environ.get('SLT_R1_SEEDS','1' if SMOKE else '3'))
BASE_SEED=int(os.environ.get('SLT_BASE_SEED','42'))
CLASS_PAIRS=[(0,1),(3,5),(4,9),(2,7)]
TRANSFORMS=[('identity',0),('rot',15),('rot',45),('rot',90),('perm',0.3),('perm',0.7),('perm',1.0)]
THRESHOLDS=[0.80,0.90,0.95,0.99]
if SMOKE:
    CLASS_PAIRS=CLASS_PAIRS[:1]; TRANSFORMS=[('identity',0),('perm',1.0)]


def perm_idx(d,f,seed=123):
    rng=np.random.RandomState(seed); p=np.arange(d); k=int(round(float(f)*d))
    if k>=2:
        idx=rng.choice(d,k,replace=False); p[idx]=idx[rng.permutation(k)]
    return p


def transform(X,kind,val):
    if kind=='identity': return X.copy()
    if kind=='perm': return X[:,perm_idx(X.shape[1],val)].copy()
    if kind=='rot':
        t=torch.as_tensor(X.reshape(-1,1,28,28),dtype=torch.float32)
        return TF.rotate(t,float(val),fill=0.0).numpy().reshape(len(X),-1)
    raise ValueError(kind)


def safe_auc(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float)
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None


def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('mnist'); rows=[]
    for si in range(N_SEEDS):
        seed=BASE_SEED+si; set_seed(seed)
        for c0,c1 in CLASS_PAIRS:
            XAtr,yAtr=binary_subset(Xtr,ytr,c0,c1); XAte,yAte=binary_subset(Xte,yte,c0,c1)
            for kind,val in TRANSFORMS:
                XBtr=transform(XAtr,kind,val); XBte=transform(XAte,kind,val); yBtr=yAtr.copy(); yBte=yAte.copy()
                # B-only reference: establishment is measured relative to attainable B performance.
                set_seed(seed+7000+c0*100+c1)
                mb=make_model('mlp',2,2,scenario='task',dataset='mnist').to(DEVICE)
                train_task(mb,1,XBtr,yBtr,arch='mlp',dataset='mnist',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
                RBB_ref=accuracy(mb,1,XBte,yBte,arch='mlp',dataset='mnist')

                set_seed(seed)
                m=make_model('mlp',2,2,scenario='task',dataset='mnist').to(DEVICE)
                train_task(m,0,XAtr,yAtr,arch='mlp',dataset='mnist',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
                RAA=accuracy(m,0,XAte,yAte,arch='mlp',dataset='mnist')
                rho_raw=rho_pre(m,0,XBte,yBte,arch='mlp',dataset='mnist',seed=seed)
                rho_pi=rho_pre_permutation_invariant(m,0,XBte,yBte,arch='mlp',dataset='mnist',n_classes=2,seed=seed)
                S=act_cov_overlap(m,XAte,XBte,arch='mlp',dataset='mnist')
                train_task(m,1,XBtr,yBtr,arch='mlp',dataset='mnist',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
                RBA=accuracy(m,0,XAte,yAte,arch='mlp',dataset='mnist')
                RBB=accuracy(m,1,XBte,yBte,arch='mlp',dataset='mnist')
                retA=RBA/(RAA+1e-12); estB=RBB/(RBB_ref+1e-12)
                row={'seed':seed,'class_pair':f'{c0}/{c1}','kind':kind,'value':float(val),
                     'transform':f'{kind}{val}','identity':kind=='identity','rho_pre_raw':rho_raw,
                     'rho_pre_pi':rho_pi,'S':S,'R_AA':RAA,'R_BA':RBA,'R_BB':RBB,'R_BB_ref':RBB_ref,
                     'retention_A':retA,'establishment_B':estB,'forgetting':RAA-RBA}
                for th in THRESHOLDS:
                    row[f'mutual_{th:.2f}']=int(retA>=th and estB>=th)
                rows.append(row)
                print(f"s{seed} {c0}/{c1} {kind}{val}: Aret={retA:.3f} Best={estB:.3f} rhoPI={rho_pi:+.3f}",flush=True)

    summary={'n':len(rows),'thresholds':{},'excluding_identity':{}}
    for th in THRESHOLDS:
        key=f'mutual_{th:.2f}'
        y=[r[key] for r in rows]
        summary['thresholds'][str(th)]={'rate':float(np.mean(y)),
            'auc_rho_pre_pi':safe_auc(y,[r['rho_pre_pi'] for r in rows]),
            'auc_rho_pre_raw':safe_auc(y,[r['rho_pre_raw'] for r in rows]),
            'auc_S':safe_auc(y,[r['S'] for r in rows])}
        rr=[r for r in rows if not r['identity']]; yy=[r[key] for r in rr]
        summary['excluding_identity'][str(th)]={'n':len(rr),'rate':float(np.mean(yy)),
            'auc_rho_pre_pi':safe_auc(yy,[r['rho_pre_pi'] for r in rr]),
            'auc_S':safe_auc(yy,[r['S'] for r in rr])}
    summary['corr_rhoPI_forgetting']=pearson_safe([r['rho_pre_pi'] for r in rows],[r['forgetting'] for r in rows])[0]
    out={'rows':rows,'summary':summary,
         '_meta':run_metadata(ROOT,{'experiment':'R1','seeds':N_SEEDS,'epochs':EPOCHS,'smoke':SMOKE,
                                    'definition':'mutual coexistence = A retention AND B establishment',
                                    'elapsed_s':time.time()-t0})}
    json_dump_safe(out,'r1_mutual_coexistence_results.json')

    x=np.array([r['rho_pre_pi'] for r in rows]); a=np.array([r['retention_A'] for r in rows]); b=np.array([r['establishment_B'] for r in rows])
    fig,ax=plt.subplots(1,2,figsize=(11,4.2)); ax[0].scatter(x,a,s=36,alpha=.8); ax[0].axhline(.9,ls='--'); ax[0].set(xlabel='Pre-hoc compatibility (permutation-invariant)',ylabel='Resident retention',title='Resident persistence'); ax[0].grid(alpha=.25)
    ax[1].scatter(x,b,s=36,alpha=.8); ax[1].axhline(.9,ls='--'); ax[1].set(xlabel='Pre-hoc compatibility (permutation-invariant)',ylabel='Invader establishment / B-only',title='Invader establishment'); ax[1].grid(alpha=.25)
    plt.tight_layout(); plt.savefig('r1_mutual_coexistence.png',dpi=220,bbox_inches='tight'); plt.close()
    print('saved r1_mutual_coexistence_results.json')

if __name__=='__main__': main()
