#!/usr/bin/env python3
"""S3 — optimizer/LR/batch/schedule robustness of pre-hoc compatibility ordering.

Directly answers the reviewer question that A1/B1 do not: whether rho/Omega's
*behavioral predictive ordering* persists under training regimes known to change
forgetting.  rho is always computed from incoming-task TRAIN labels only.
"""
from __future__ import annotations
import copy,itertools,os,sys,time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_revision import rho_pre_permutation_invariant
from slt_freeze import *
from slt_confirmatory import run_metadata,freeze_batchnorm_stats

EXP='S3_omega_regime_sweep'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_S3_SEEDS','1' if SMOKE else '3')); EPOCHS=int(os.environ.get('SLT_S3_EPOCHS','2' if SMOKE else '20')); NBOOT=200 if SMOKE else 5000
MAX_PAIRS=2 if SMOKE else None
REGIMES=[
 {'name':'adam_lr1e-3_bs128','optimizer':'adam','lr':1e-3,'batch':128,'momentum':0.,'schedule':'constant','wd':1e-4},
 {'name':'sgd_lr0.03_bs32','optimizer':'sgd','lr':.03,'batch':32,'momentum':0.,'schedule':'constant','wd':0.},
 {'name':'sgd_lr0.03_bs128','optimizer':'sgd','lr':.03,'batch':128,'momentum':0.,'schedule':'constant','wd':0.},
 {'name':'sgdm_lr0.03_bs32','optimizer':'sgd','lr':.03,'batch':32,'momentum':.9,'schedule':'constant','wd':1e-4},
 {'name':'sgdm_lr0.1_cosine_bs32','optimizer':'sgd','lr':.1,'batch':32,'momentum':.9,'schedule':'cosine','wd':5e-4},
]
if SMOKE: REGIMES=REGIMES[:2]


def train_regime(model,t,X,y,cfg,seed):
    _set_trainable(model,t); xr=_reshape_for('resnet18','cifar10',X); dl=DataLoader(TensorDataset(torch.as_tensor(xr),torch.as_tensor(y)),batch_size=cfg['batch'],shuffle=True,generator=torch.Generator().manual_seed(seed))
    params=[p for p in model.parameters() if p.requires_grad]
    if cfg['optimizer']=='adam': opt=torch.optim.Adam(params,lr=cfg['lr'],weight_decay=cfg['wd'])
    else: opt=torch.optim.SGD(params,lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['wd'])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,EPOCHS)) if cfg['schedule']=='cosine' else None
    for _ in range(EPOCHS):
        model.train()
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(); loss=F.cross_entropy(model.heads[t](model.features(xb)),yb); loss.backward(); opt.step()
        if sch is not None: sch.step()


def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('cifar10'); tr=make_tasks(Xtr,ytr,SPLITS['cifar10']); te=make_tasks(Xte,yte,SPLITS['cifar10']); pairs=list(itertools.permutations(range(5),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs
    rows=[]
    for cfg in REGIMES:
        for si in range(SEEDS):
            seed=BASE+si; set_seed(seed); residents={}
            for a in range(5):
                m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); train_regime(m,a,*tr[a],cfg,seed+100*a); residents[a]=m
            for a,b in pairs:
                mA=residents[a]; RAA=accuracy(mA,a,*te[a],arch='resnet18',dataset='cifar10')
                Xp,yp,_=train_probe_pool(*tr[b],n=min(256,len(tr[b][0])),seed=seed+1000+a*10+b)
                rho=rho_pre_permutation_invariant(mA,a,Xp,yp,arch='resnet18',dataset='cifar10',n_classes=2,n_shuffle=20,seed=seed+2000+a*10+b)
                m=copy.deepcopy(mA); train_regime(m,b,*tr[b],cfg,seed+3000+a*10+b); RBA=accuracy(m,a,*te[a],arch='resnet18',dataset='cifar10')
                rows.append({'regime':cfg['name'],'seed':seed,'task_a':a,'task_b':b,'rho_pre_pi':rho,'omega':1-rho,'R_AA':RAA,'R_BA':RBA,'forgetting':RAA-RBA})
                print(cfg['name'],seed,a,b,'rho',round(rho,3),'F',round(RAA-RBA,3),flush=True)
    analysis={}; pair_by_reg={}
    for cfg in REGIMES:
        name=cfg['name']; means=[]
        for a,b in pairs:
            z=[r for r in rows if r['regime']==name and r['task_a']==a and r['task_b']==b]
            if z: means.append({'task_a':a,'task_b':b,'rho_pre_pi':float(np.mean([q['rho_pre_pi'] for q in z])),'forgetting':float(np.mean([q['forgetting'] for q in z]))})
        pair_by_reg[name]=means; r,p=pearson_safe([z['rho_pre_pi'] for z in means],[z['forgetting'] for z in means]); sr,sp=spearman_safe([z['rho_pre_pi'] for z in means],[z['forgetting'] for z in means])
        analysis[name]={'pearson_r':r,'pearson_p':p,'spearman_r':sr,'spearman_p':sp,'slope_ci':bootstrap_slope(means,'rho_pre_pi','forgetting',n_boot=NBOOT,seed=BASE+17),'rho_ci':bootstrap_pearson_ci(means,'rho_pre_pi','forgetting',n_boot=NBOOT,seed=BASE+18),'loto':loto_incoming_signs(means,'rho_pre_pi','forgetting',5),'forgetting_sd':float(np.std([z['forgetting'] for z in means]))}
    # Cross-regime stability of pairwise forgetting rank (not a pass criterion).
    names=[c['name'] for c in REGIMES]; cross={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            da={(z['task_a'],z['task_b']):z for z in pair_by_reg[a]}; db={(z['task_a'],z['task_b']):z for z in pair_by_reg[b]}; ks=sorted(set(da)&set(db)); rr,_=spearman_safe([da[k]['forgetting'] for k in ks],[db[k]['forgetting'] for k in ks]); cross[f'{a}__vs__{b}']=rr
    out={'rows':rows,'analysis':analysis,'cross_regime_forgetting_rank':cross,'regimes':REGIMES,'protocol':{'rho_supervision':'256 balanced B-train labels only','question':'does compatibility ordering persist across optimizer, LR, batch size, momentum and cosine schedule?','no_posthoc_regime_selection':True},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}
    p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
