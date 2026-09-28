#!/usr/bin/env python3
"""S5 — broader pretrained-backbone confirmation.

Default: ImageNet-pretrained ResNet-50 on the frozen semantic CIFAR-100 task
suite.  Optional: torchvision ViT-B/16 via SLT_S5_ARCHS.  This experiment is
independent and uses train-only pre-hoc compatibility probes.
"""
from __future__ import annotations
import copy,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_revision import rho_pre_permutation_invariant
from slt_freeze import *
from slt_confirmatory import run_metadata

EXP='S5_pretrained_backbone'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_S5_SEEDS','1' if SMOKE else '3')); EPOCHS=int(os.environ.get('SLT_S5_EPOCHS','1' if SMOKE else '5')); NBOOT=200 if SMOKE else 5000
ARCHS=[x.strip() for x in os.environ.get('SLT_S5_ARCHS','resnet50_imagenet').split(',') if x.strip()]
PAIR_LIMIT=int(os.environ.get('SLT_S5_PAIR_LIMIT','3' if SMOKE else '30'))


def hp(arch):
    return (1e-4,64) if arch=='resnet50_imagenet' else (2e-5,32)


def main():
    t0=time.time(); Xtr,ytr,Xte,yte,splits,class_names=load_cifar100_semantic(); tr=make_tasks(Xtr,ytr,splits); te=make_tasks(Xte,yte,splits); pairs=fixed_pairs(10,PAIR_LIMIT); rows=[]
    for arch in ARCHS:
        lr,batch=hp(arch)
        for si in range(SEEDS):
            seed=BASE+si; set_seed(seed); residents={}
            for a in sorted(set(a for a,_ in pairs)):
                m=make_model(arch,10,10,scenario='task',dataset='cifar100').to(DEVICE)
                train_task_optimizer(m,a,*tr[a],arch=arch,dataset='cifar100',optimizer='adam',lr=lr,epochs=EPOCHS,batch=batch,wd=1e-4,seed=seed+100*a); residents[a]=m
            for a,b in pairs:
                mA=residents[a]; RAA=accuracy(mA,a,*te[a],arch=arch,dataset='cifar100',batch=64)
                Xp,yp,_=train_probe_pool(*tr[b],n=min(256,len(tr[b][0])),seed=seed+1000+a*100+b)
                rho=rho_pre_permutation_invariant(mA,a,Xp,yp,arch=arch,dataset='cifar100',n_classes=10,n_shuffle=20,seed=seed+2000+a*100+b)
                m=copy.deepcopy(mA); train_task_optimizer(m,b,*tr[b],arch=arch,dataset='cifar100',optimizer='adam',lr=lr,epochs=EPOCHS,batch=batch,wd=1e-4,seed=seed+3000+a*100+b)
                RBA=accuracy(m,a,*te[a],arch=arch,dataset='cifar100',batch=64); rows.append({'arch':arch,'seed':seed,'task_a':a,'task_b':b,'rho_pre_pi':rho,'omega':1-rho,'R_AA':RAA,'R_BA':RBA,'forgetting':RAA-RBA})
                print(arch,seed,a,b,'rho',round(rho,3),'F',round(RAA-RBA,3),flush=True)
    analysis={}
    for arch in ARCHS:
        means=[]
        for a,b in pairs:
            z=[r for r in rows if r['arch']==arch and r['task_a']==a and r['task_b']==b]
            if z: means.append({'task_a':a,'task_b':b,'rho_pre_pi':float(np.mean([q['rho_pre_pi'] for q in z])),'forgetting':float(np.mean([q['forgetting'] for q in z]))})
        r,p=pearson_safe([z['rho_pre_pi'] for z in means],[z['forgetting'] for z in means]); sr,sp=spearman_safe([z['rho_pre_pi'] for z in means],[z['forgetting'] for z in means])
        analysis[arch]={'pearson_r':r,'pearson_p':p,'spearman_r':sr,'spearman_p':sp,'slope_ci':bootstrap_slope(means,'rho_pre_pi','forgetting',n_boot=NBOOT,seed=BASE+31),'rho_ci':bootstrap_pearson_ci(means,'rho_pre_pi','forgetting',n_boot=NBOOT,seed=BASE+32),'loto':loto_incoming_signs(means,'rho_pre_pi','forgetting',10)}
    out={'rows':rows,'analysis':analysis,'architectures':ARCHS,'semantic_task_pairs':CIFAR100_SEM_PAIRS,'protocol':{'pretrained':'ImageNet torchvision weights; local path supported by SLT_PRETRAINED_WEIGHTS_PATH','rho_supervision':'256 balanced B-train labels only','pair_limit':PAIR_LIMIT,'full_backbone_finetuning':True,'epochs':EPOCHS},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}; p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
