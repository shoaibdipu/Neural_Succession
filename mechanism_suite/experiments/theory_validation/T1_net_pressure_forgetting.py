#!/usr/bin/env python3
"""T1 — Theorem-1 / Corollary-1.1: pre-hoc pressure predicts feature decay.

This confirmatory test contains two pre-registered pressure branches:

A) Original Definition-4 branch
   alpha_ij = <phi_i^A,phi_j^B>/||phi_i^A||^2,
   lambda_i^D4 = sum_j alpha_ij x_j^B(pre).
   alpha is computed before Task-B training and is never fitted.

B) Revised signed-local branch
   pi_i = <grad_theta x_i^A, grad_theta L_B>,
   lambda_i^local = pi_i/x_i^A.

The outcome is the observed finite-time feature decay rate during plain-SGD Task-B
training.  The primary scale-control analysis is the partial correlation between
pressure and decay controlling log x_i^A(0).  The Definition-4 branch additionally
uses a shuffled-alpha null that preserves coefficient magnitudes while breaking
invader-feature identity.  A correlation that does not beat this null is not
counted as mechanistic support.

Task-level forgetting is retained as the downstream behavioural consequence; it
is not allowed to substitute for the feature-level theorem test.
"""
from __future__ import annotations
import copy, itertools, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *  # noqa
from slt_confirmatory import stratified_probe, json_dump_safe, run_metadata, train_sgd_theory
from slt_revision import (signed_feature_pressure_virtual_step, qap_task_label_permutation,
                          pair_means, partial_corr_residualized, shuffled_matrix_null)
import slt_theory as T

SMOKE=os.environ.get('SLT_SMOKE','0')=='1'
DATASET=os.environ.get('SLT_T1_DATASET','cifar10').lower(); ARCH='mlp' if DATASET=='mnist' else 'resnet18'
SEEDS=int(os.environ.get('SLT_T1_SEEDS','1' if SMOKE else '3')); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
EPOCHS=int(os.environ.get('SLT_T1_EPOCHS','2' if SMOKE else ('15' if DATASET=='mnist' else '20')))
NPROBE=int(os.environ.get('SLT_T1_PROBE','48' if SMOKE else '256')); NJAC=int(os.environ.get('SLT_T1_JAC_PROBE','24' if SMOKE else ('128' if DATASET=='mnist' else '64')))
EPS=float(os.environ.get('SLT_T1_EPS','1e-4')); LR=float(os.environ.get('SLT_T1_LR','0.001')); BATCH=int(os.environ.get('SLT_T1_BATCH','128'))
N_SHUFFLE=int(os.environ.get('SLT_T1_SHUFFLE','50' if SMOKE else '1000'))
MAX_PAIRS=int(os.environ.get('SLT_MAX_PAIRS','2')) if SMOKE else None
SHARED_RESIDENT_DIR=os.environ.get('SLT_SHARED_RESIDENT_DIR'); SHARED_RESIDENT_MODE=os.environ.get('SLT_SHARED_RESIDENT_MODE','')


def corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12: return (None,None)
    r,p=pearsonr(x[ok],y[ok]); return float(r),float(p)


def vector_shuffle_null(x,y,control,seed,n_perm):
    x=np.asarray(x,float); y=np.asarray(y,float); c=np.asarray(control,float)
    obs=partial_corr_residualized(x,y,c); robs=obs['partial_r']
    if robs is None: return {'observed_partial_r':None,'p_shuffle':None,'n':obs['n']}
    rng=np.random.RandomState(seed); vals=[]
    for _ in range(int(n_perm)):
        z=x[rng.permutation(len(x))]; rr=partial_corr_residualized(z,y,c)['partial_r']
        if rr is not None: vals.append(rr)
    vals=np.asarray(vals,float); p=float((1+np.sum(np.abs(vals)>=abs(robs)))/(1+len(vals))) if len(vals) else None
    return {'observed_partial_r':float(robs),'p_shuffle':p,'n':obs['n'],'n_shuffle':int(len(vals)),
            'null_mean':float(vals.mean()) if len(vals) else None,'null_std':float(vals.std()) if len(vals) else None}


def weighted_summary(lam,x0,head_importance=None):
    lam=np.asarray(lam,float); x0=np.asarray(x0,float); ok=np.isfinite(lam)&np.isfinite(x0)&(x0>1e-12)
    if not ok.any(): return {'mean':None,'strength_weighted':None,'positive_weighted':None,'frac_positive':None}
    w=x0[ok].copy()
    if head_importance is not None and len(head_importance)==len(x0): w*=np.asarray(head_importance,float)[ok]
    w=np.clip(w,0,None); w=w/(w.sum()+1e-12); z=lam[ok]
    return {'mean':float(np.mean(z)),'strength_weighted':float(np.sum(w*z)),
            'positive_weighted':float(np.sum(w*np.clip(z,0,None))),'frac_positive':float(np.mean(z>0))}


def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np(DATASET)
    tr=[binary_subset(Xtr,ytr,*cs) for cs in SPLITS[DATASET]]; te=[binary_subset(Xte,yte,*cs) for cs in SPLITS[DATASET]]
    pairs=list(itertools.permutations(range(len(tr)),2)); pairs=pairs[:MAX_PAIRS] if MAX_PAIRS else pairs
    rows=[]
    for si in range(SEEDS):
        seed=BASE+si; set_seed(seed); residents={}
        for a in sorted(set(a for a,_ in pairs)):
            m=make_model(ARCH,len(tr),2,scenario='task',dataset=DATASET).to(DEVICE)
            ckpt=(Path(SHARED_RESIDENT_DIR)/f'seed{seed}_task{a}.pt') if SHARED_RESIDENT_DIR else None
            if ckpt is not None and SHARED_RESIDENT_MODE=='load' and ckpt.exists():
                obj=torch.load(ckpt,map_location=DEVICE); m.load_state_dict(obj['state_dict'] if isinstance(obj,dict) and 'state_dict' in obj else obj)
            else:
                train_sgd_theory(m,a,*tr[a],arch=ARCH,dataset=DATASET,epochs=EPOCHS,lr=LR,batch=BATCH,seed=seed+10*a,freeze_bn=(ARCH=='resnet18'))
                if ckpt is not None and SHARED_RESIDENT_MODE=='save':
                    ckpt.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':m.state_dict(),'seed':seed,'task':a,'lr':LR,'epochs':EPOCHS,'batch':BATCH},ckpt)
            residents[a]=m
        for a,b in pairs:
            mA=residents[a]; RAA=accuracy(mA,a,*te[a],arch=ARCH,dataset=DATASET)
            XAp,yAp,_=stratified_probe(*tr[a],NPROBE,seed=seed+100+a); XBp,yBp,_=stratified_probe(*tr[b],NPROBE,seed=seed+200+b)
            kw=dict(arch=ARCH,dataset=DATASET,reshape_fn=_reshape_for,device=DEVICE)
            # Signed local pressure branch.
            pr=signed_feature_pressure_virtual_step(mA,XAp,XBp,yBp,arch=ARCH,dataset=DATASET,n_classes=2,
                 X_B_probe_train=tr[b][0],y_B_probe_train=tr[b][1],eps=EPS,seed=seed+300+a*10+b)
            x0=np.asarray(pr['x0'],float); floor=max(1e-10,float(np.quantile(x0[x0>1e-12],.10)) if np.any(x0>1e-12) else 1e-10)
            local=np.full_like(x0,np.nan); elig_local=x0>=floor; local[elig_local]=np.asarray(pr['pi'])[elig_local]/np.clip(x0[elig_local],1e-12,None)

            # Original Definition-4 branch, fixed before B training.
            XAj=XAp[:min(NJAC,len(XAp))]; XBj=XBp[:min(NJAC,len(XBp))]
            JA=T.feature_jacobians(mA,XAj,n_probe=len(XAj),**kw); JB=T.feature_jacobians(mA,XBj,n_probe=len(XBj),**kw)
            alpha=T.competitive_alpha_AB(JA,JB); xBpre=T.feature_strength(mA,XBj,max_n=len(XBj),**kw)
            elig_d4=np.isfinite(alpha).all(axis=1)&(x0>floor)
            lam_d4=np.full_like(x0,np.nan); lam_d4[elig_d4]=alpha[elig_d4]@xBpre

            try: imp=np.linalg.norm(mA.heads[a].weight.detach().cpu().numpy(),axis=0)
            except Exception: imp=None

            # Observe actual feature decay under plain-SGD B training.
            m=copy.deepcopy(mA)
            train_sgd_theory(m,b,*tr[b],arch=ARCH,dataset=DATASET,epochs=EPOCHS,lr=LR,batch=BATCH,seed=seed+400+b,freeze_bn=(ARCH=='resnet18'))
            x1=T.feature_strength(m,XAp,max_n=len(XAp),**kw); tau=max(T.lv_time(EPOCHS*int(np.ceil(len(tr[b][0])/BATCH)),LR),1e-12)
            decay=-np.log(np.clip(x1,1e-12,None)/np.clip(x0,1e-12,None))/tau
            RBA=accuracy(m,a,*te[a],arch=ARCH,dataset=DATASET); F=float(RAA-RBA)

            # Feature-level scale-controlled tests.
            okL=np.isfinite(local)&np.isfinite(decay)&(x0>floor)
            local_pc=partial_corr_residualized(local[okL],decay[okL],np.log(np.clip(x0[okL],1e-12,None)))
            local_null=vector_shuffle_null(local[okL],decay[okL],np.log(np.clip(x0[okL],1e-12,None)),seed+500+a*10+b,N_SHUFFLE) if okL.sum()>=4 else {}
            okD=elig_d4&np.isfinite(decay)
            d4_pc=partial_corr_residualized(lam_d4[okD],decay[okD],np.log(np.clip(x0[okD],1e-12,None)))
            # Primary null: shuffle alpha columns relative to x_B strength, preserving alpha scale.
            d4_null=shuffled_matrix_null(alpha[okD],xBpre,decay[okD],control=np.log(np.clip(x0[okD],1e-12,None)),
                                         n_perm=N_SHUFFLE,seed=seed+600+a*10+b) if okD.sum()>=4 else {}

            wsL=weighted_summary(local,x0,imp); wsD=weighted_summary(lam_d4,x0,imp)
            row={'seed':seed,'task_a':a,'task_b':b,'R_AA':float(RAA),'R_BA':float(RBA),'forgetting':F,
                 'b_probe_loss':pr['b_probe_loss'],'b_grad_norm':pr['b_grad_norm'],'tau_observed':tau,
                 'local_mean':wsL['mean'],'local_strength_weighted':wsL['strength_weighted'],'local_positive_weighted':wsL['positive_weighted'],'local_frac_positive':wsL['frac_positive'],
                 'def4_mean':wsD['mean'],'def4_strength_weighted':wsD['strength_weighted'],'def4_positive_weighted':wsD['positive_weighted'],'def4_frac_positive':wsD['frac_positive'],
                 'feature_test_local':{'partial_corr_controlling_log_x0':local_pc,'shuffle_null':local_null,'n_eligible':int(okL.sum())},
                 'feature_test_definition4':{'partial_corr_controlling_log_x0':d4_pc,'shuffled_alpha_null':d4_null,'n_eligible':int(okD.sum())},
                 'alpha_diagnostics':T.alpha_diagnostics(alpha[okD]) if okD.sum() else None,
                 'x0_mean':float(np.mean(x0)),'observed_decay_mean':float(np.nanmean(decay))}
            rows.append(row)
            print(f"{DATASET} s{seed} T{a}->T{b} F={F:.3f} local_pr={local_pc.get('partial_r')} D4_pr={d4_pc.get('partial_r')} D4_pnull={d4_null.get('p_shuffled_alpha')}",flush=True)

    mean_keys=['forgetting','local_mean','local_strength_weighted','local_positive_weighted','local_frac_positive',
               'def4_mean','def4_strength_weighted','def4_positive_weighted','def4_frac_positive','x0_mean','observed_decay_mean']
    means=pair_means(rows,mean_keys); summary={}
    for k in ['local_strength_weighted','local_positive_weighted','def4_strength_weighted','def4_positive_weighted']:
        r,p=corr([z[k] for z in means],[z['forgetting'] for z in means]); summary[k]={'pearson_r':r,'pair_level_p':p}
        if len(means) and MAX_PAIRS is None: summary[k]['task_permutation']=qap_task_label_permutation(means,k,'forgetting',n_tasks=len(tr),seed=BASE)
    for branch,path in [('local','feature_test_local'),('definition4','feature_test_definition4')]:
        prs=[r[path]['partial_corr_controlling_log_x0'].get('partial_r') for r in rows if r[path]['partial_corr_controlling_log_x0'].get('partial_r') is not None]
        if branch=='definition4': pnull=[r[path]['shuffled_alpha_null'].get('p_shuffled_alpha') for r in rows if r[path]['shuffled_alpha_null'].get('p_shuffled_alpha') is not None]
        else: pnull=[r[path]['shuffle_null'].get('p_shuffle') for r in rows if r[path]['shuffle_null'].get('p_shuffle') is not None]
        summary[f'{branch}_feature_level']={'median_partial_r_controlling_log_x0':float(np.median(prs)) if prs else None,
                                            'fraction_positive_partial_r':float(np.mean(np.asarray(prs)>0)) if prs else None,
                                            'fraction_null_p_lt_0p05':float(np.mean(np.asarray(pnull)<.05)) if pnull else None,
                                            'n_transition_tests':len(prs)}
    out={'rows':rows,'pair_means':means,'summary':summary,
         '_meta':run_metadata(ROOT,{'experiment':'T1','dataset':DATASET,'seeds':SEEDS,'epochs':EPOCHS,'lr':LR,'eps':EPS,'jacobian_probe':NJAC,'n_shuffle':N_SHUFFLE,
          'primary_feature_test':'partial corr pressure vs observed decay controlling log x_i(0); Definition-4 shuffled-alpha null','elapsed_s':time.time()-t0})}
    json_dump_safe(out,f't1_net_pressure_{DATASET}_results.json')

if __name__=='__main__': main()
