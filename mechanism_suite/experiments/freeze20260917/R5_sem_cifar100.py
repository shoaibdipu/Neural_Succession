#!/usr/bin/env python3
from __future__ import annotations
import copy,itertools,os,sys,time,hashlib,datetime
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_revision import rho_pre_permutation_invariant
from slt_confirmatory import stratified_train_val, run_metadata
from slt_freeze import *

EXP='R5_sem_cifar100'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_R5SEM_SEEDS','1' if SMOKE else '3')); EPOCHS=int(os.environ.get('SLT_R5SEM_EPOCHS','2' if SMOKE else '20')); MAX_PAIRS=3 if SMOKE else None

def main():
    pre_out=exp_outdir(ROOT,EXP); pre_seal=pre_out/'SEALED_README.txt'
    if pre_seal.exists(): raise RuntimeError(f'refusing to overwrite sealed R5-sem output at {pre_out}; use a new SLT_RESULTS_ROOT')
    t0=time.time(); Xtr,ytr,Xte,yte,splits,class_names=load_cifar100_semantic(); tr=make_tasks(Xtr,ytr,splits); te=make_tasks(Xte,yte,splits)
    pairs=list(itertools.permutations(range(10),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs; rows=[]
    for si in range(SEEDS):
        seed=BASE+si; set_seed(seed); residents={}
        for a in sorted(set(a for a,_ in pairs)):
            m=make_model('resnet18',10,10,scenario='task',dataset='cifar100').to(DEVICE)
            train_task(m,a,*tr[a],arch='resnet18',dataset='cifar100',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4); residents[a]=m
        for a,b in pairs:
            mA=residents[a]; RAA=accuracy(mA,a,*te[a],arch='resnet18',dataset='cifar100')
            Xp,yp,_=train_probe_pool(*tr[b],n=min(256,len(tr[b][0])),seed=seed+900+a*100+b)
            rho=rho_pre_permutation_invariant(mA,a,Xp,yp,arch='resnet18',dataset='cifar100',n_classes=10,n_shuffle=20,seed=seed+1000+a*100+b)
            # Independent pre-hoc B probe for Plan-B diagnostics. Fit/evaluate split is disjoint.
            (Bfit,yfit),(Bev,yev),_,_=stratified_train_val(*tr[b],val_frac=.20,seed=seed+2000+b)
            ploss,pacc,clf=linear_probe_loss_and_acc(mA,Bfit,yfit,Bev,yev,arch='resnet18',dataset='cifar100')
            mk=copy.deepcopy(mA); copy_probe_to_head(mk,b,clf)
            gA,_=backbone_gradient(mk,a,*tr[a],arch='resnet18',dataset='cifar100',max_n=128,seed=seed+3000+a)
            gB,_=backbone_gradient(mk,b,Bev,yev,arch='resnet18',dataset='cifar100',max_n=128,seed=seed+4000+b)
            den=float(torch.dot(gB,gB).cpu()); kappa0=float(-torch.dot(gA,gB).cpu()/(den+1e-12))
            m=copy.deepcopy(mA); train_task(m,b,*tr[b],arch='resnet18',dataset='cifar100',epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4)
            RBA=accuracy(m,a,*te[a],arch='resnet18',dataset='cifar100'); fgt=RAA-RBA
            rows.append({'seed':seed,'task_a':a,'task_b':b,'rho_pre_pi':rho,'omega':1-rho,'probe_loss_B':ploss,'probe_acc_B':pacc,'kappa0_backbone':kappa0,'R_AA':RAA,'R_BA':RBA,'forgetting':fgt})
            print(seed,a,b,'rho',round(rho,3),'k0',round(kappa0,4),'F',round(fgt,3),flush=True)
    # aggregate across seeds before task-level inference
    means=[]
    for a,b in pairs:
        rs=[r for r in rows if r['task_a']==a and r['task_b']==b]
        if not rs: continue
        means.append({'task_a':a,'task_b':b,**{k:float(np.mean([r[k] for r in rs])) for k in ['rho_pre_pi','omega','probe_loss_B','probe_acc_B','kappa0_backbone','R_AA','R_BA','forgetting']}})
    slope=bootstrap_slope(means,'rho_pre_pi','forgetting',n_boot=500 if SMOKE else 5000,seed=BASE)
    loto=loto_incoming_signs(means,'rho_pre_pi','forgetting',10)
    r,p=pearson_safe([z['rho_pre_pi'] for z in means],[z['forgetting'] for z in means])
    # frozen nested predictive diagnostics for contingency; no model selection.
    x1=np.asarray([[z['rho_pre_pi']] for z in means]); x2=np.asarray([[z['rho_pre_pi'],z['kappa0_backbone']] for z in means]); y=np.asarray([z['forgetting'] for z in means])
    def loo_mae(X):
        ps=[]
        for i in range(len(y)):
            mask=np.arange(len(y))!=i; m=LinearRegression().fit(X[mask],y[mask]); ps.append(float(m.predict(X[i:i+1])[0]))
        rr,_=pearson_safe(ps,y); return {'mae':float(np.mean(np.abs(np.asarray(ps)-y))),'r':rr}
    analysis={'pearson_r':r,'pair_p':p,'cluster_bootstrap_slope':slope,'LOTO_incoming':loto,'rho_only_LOO':loo_mae(x1),'rho_plus_kappa0_LOO':loo_mae(x2),
              'primary_pass':bool(slope['ci95'] and slope['ci95'][1]<0 and loto['n_negative_slope_folds']>=8),
              'strong_replication_target_abs_beta_ge_0p46':bool(slope['slope'] is not None and abs(slope['slope'])>=.46)}
    out={'rows':rows,'pair_means':means,'analysis':analysis,'semantic_task_pairs':CIFAR100_SEM_PAIRS,'class_names':class_names,'protocol':{'rho_supervision':'256 class-balanced labeled examples from incoming-task TRAIN only','benchmark_test_labels_used_for_rho':False,'primary_exists':'cluster-bootstrap slope CI excludes 0 in predicted direction and >=8/10 negative LOTO slopes','strong_prior_effect_target':'abs(beta)>=0.46; prior CIFAR10 lower-CI replication target'},'_meta':run_metadata(ROOT,{'experiment':EXP,'seeds':SEEDS,'epochs':EPOCHS,'elapsed_s':time.time()-t0})}
    pth=dump_result(ROOT,EXP,out)
    digest=hashlib.sha256(pth.read_bytes()).hexdigest(); seal=pth.parent/'SEALED_README.txt'
    seal.write_text(f'R5-sem confirmatory output sealed immediately after write.\nfile={pth.name}\nsha256={digest}\nsealed_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\nDo not modify or overwrite this result before the frozen analysis script is run.\n')
    (pth.parent/(pth.name+'.sha256')).write_text(f'{digest}  {pth.name}\n'); print('saved and sealed',pth)
if __name__=='__main__': main()
