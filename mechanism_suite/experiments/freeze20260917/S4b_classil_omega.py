#!/usr/bin/env python3
"""S4b — class-IL pre-hoc compatibility ordering and coexistence.

Direct answer to reviewer Q2: same directional pre-hoc protocol in a single-head,
no-task-oracle evaluation regime.  Uses incoming-task TRAIN labels only.
"""
from __future__ import annotations
import copy,itertools,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_freeze import *
from slt_confirmatory import train_sgd_theory, run_metadata

EXP='S4b_classil_omega'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
CLASS_LR=float(os.environ.get('SLT_CLASSIL_LR','0.03')); CLASS_BATCH=int(os.environ.get('SLT_CLASSIL_BATCH','32'))
SEEDS=int(os.environ.get('SLT_S4B_SEEDS','1' if SMOKE else '3')); EPOCHS=int(os.environ.get('SLT_S4B_EPOCHS','2' if SMOKE else '20')); NBOOT=200 if SMOKE else 5000; MAX_PAIRS=2 if SMOKE else None


def seen_classes(a,b=None):
    z=list(range(2*a,2*a+2))
    if b is not None: z+=list(range(2*b,2*b+2))
    return sorted(set(z))


def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('cifar10'); tr=make_tasks(Xtr,ytr,SPLITS['cifar10']); te=make_tasks(Xte,yte,SPLITS['cifar10']); pairs=list(itertools.permutations(range(5),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs; rows=[]
    for si in range(SEEDS):
        seed=BASE+si; set_seed(seed); residents={}; bref={}
        for t in range(5):
            m=make_model('resnet18',5,2,scenario='class',dataset='cifar10').to(DEVICE); train_sgd_theory(m,t,*tr[t],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=CLASS_LR,batch=CLASS_BATCH,seed=seed+100+t,freeze_bn=True,weight_decay=1e-4,global_labels=True,class_offset=2*t); residents[t]=m
            bref[t]=accuracy(m,t,*te[t],arch='resnet18',dataset='cifar10',global_labels=True,class_offset=2*t,classil_classes=seen_classes(t))
        for a,b in pairs:
            mA=residents[a]; RAA=bref[a]; Xp,yp,_=train_probe_pool(*tr[b],n=min(256,len(tr[b][0])),seed=seed+1000+a*10+b)
            rho=rho_pre_classil(mA,a,Xp,yp,arch='resnet18',dataset='cifar10',n_cls=2,n_samples=None,n_shuffle=20,seed=seed+2000+a*10+b)
            m=copy.deepcopy(mA); train_sgd_theory(m,b,*tr[b],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=CLASS_LR,batch=CLASS_BATCH,seed=seed+3000+a*10+b,freeze_bn=True,weight_decay=1e-4,global_labels=True,class_offset=2*b)
            seen=seen_classes(a,b); RBA=accuracy(m,a,*te[a],arch='resnet18',dataset='cifar10',global_labels=True,class_offset=2*a,classil_classes=seen); RBBseq=accuracy(m,b,*te[b],arch='resnet18',dataset='cifar10',global_labels=True,class_offset=2*b,classil_classes=seen); RBBref=bref[b]; coex=bool(RBA/(RAA+1e-12)>=.9 and RBBseq/(RBBref+1e-12)>=.9)
            rows.append({'seed':seed,'task_a':a,'task_b':b,'rho_pre_pi':rho,'forgetting':RAA-RBA,'R_AA':RAA,'R_BA':RBA,'R_BB_ref':RBBref,'R_BB_seq':RBBseq,'coexist_0p90':coex}); print(seed,a,b,rho,RAA-RBA,coex,flush=True)
    means=[]
    for a,b in pairs:
        z=[r for r in rows if r['task_a']==a and r['task_b']==b]
        if z: means.append({'task_a':a,'task_b':b,'rho_pre_pi':float(np.mean([q['rho_pre_pi'] for q in z])),'forgetting':float(np.mean([q['forgetting'] for q in z])),'coexist_0p90':int(np.mean([q['coexist_0p90'] for q in z])>=.5)})
    r,p=pearson_safe([z['rho_pre_pi'] for z in means],[z['forgetting'] for z in means]); auc=auc_safe([z['coexist_0p90'] for z in means],[z['rho_pre_pi'] for z in means])
    out={'rows':rows,'pair_means':means,'analysis':{'pearson_r':r,'p':p,'rho_forgetting_ci':bootstrap_pearson_ci(means,'rho_pre_pi','forgetting',n_boot=NBOOT,seed=BASE+111),'coexistence_auc_rho':auc,'coexistence_auc_ci':bootstrap_auc_ci(means,'rho_pre_pi','coexist_0p90',n_boot=NBOOT,seed=BASE+112),'slope_ci':bootstrap_slope(means,'rho_pre_pi','forgetting',n_boot=NBOOT,seed=BASE+113)},'protocol':{'scenario':'class-IL single head','optimizer':'plain SGD','lr':CLASS_LR,'batch':CLASS_BATCH,'task_oracle_at_eval':False,'evaluation_classes':'only classes seen in A and B; future untrained logits masked','rho_supervision':'balanced B-train probe only','coexistence':'A retention and B establishment each >=90% of single-task reference'},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}; p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
