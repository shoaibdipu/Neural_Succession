#!/usr/bin/env python3
"""R6b — decisive LV test with Definition-4 alpha fixed, fitting only g_i.

Purpose
-------
R6 fits alpha from trajectories and therefore asks whether *some* clustered LV
system describes the data.  R6b asks the stricter theory question: does Eq. (1)
predict held-out feature-population dynamics when alpha is fixed *before the
trajectory* by Definition 4?

Pipeline
--------
1. Train resident Task A.
2. Cluster shared backbone features into K populations using resident response
   profiles (clustering is only an empirical coarse-graining; theoretical
   species remain individual features).
3. Compute exact per-feature input Jacobians on A/B probes and Definition-4
   alpha_ij = <phi_i^A,phi_j^B>/||phi_i^A||^2.
4. Aggregate that fixed matrix to cluster-level alpha using pre-B feature-
   strength weights.  alpha is NEVER estimated from the trajectory.
5. On the first 60% of the Task-B trajectory, fit only one scalar g_k per
   resident cluster in
       d log X_Ak / dt = g_k - sum_l alpha_kl X_Bl.
6. Predict the held-out window.

Frozen nulls/comparators
------------------------
* constant-rate: fit d log X/dt = c on the training window;
* P_i=0: use the g_k fitted under fixed alpha but delete competitive pressure;
* shuffled-alpha: permute invader-cluster identities, refit g only, repeat;
* unconstrained LV: fit g and alpha from the same training window (flexible
  descriptive upper comparator, not a confirmatory theory test).

A positive theory result requires fixed-Definition-4 alpha to predict held-out
trajectories better than the shuffled-alpha/null models and to approach the
flexible fitted-LV comparator without pathological conditioning.
"""
from __future__ import annotations
import itertools, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *  # noqa
from slt_confirmatory import stratified_probe, train_sgd_theory, freeze_batchnorm_stats, json_dump_safe, run_metadata
import slt_theory as T

SMOKE=os.environ.get('SLT_SMOKE','0')=='1'
DATASET=os.environ.get('SLT_R6B_DATASET','cifar10').lower(); ARCH='mlp' if DATASET=='mnist' else 'resnet18'
KS=[int(x) for x in os.environ.get('SLT_R6B_K','4,8' if SMOKE else '4,8,12,16').split(',') if x]
BASE=int(os.environ.get('SLT_BASE_SEED','42')); SEEDS=int(os.environ.get('SLT_R6B_SEEDS','1' if SMOKE else '3'))
EPOCHS_A=int(os.environ.get('SLT_R6B_A_EPOCHS','3' if SMOKE else ('15' if DATASET=='mnist' else '20')))
N_STEPS=int(os.environ.get('SLT_R6B_STEPS','40' if SMOKE else '600')); CHECK=int(os.environ.get('SLT_R6B_CHECK','5' if SMOKE else '4'))
LR=float(os.environ.get('SLT_R6B_LR','0.001')); BATCH=int(os.environ.get('SLT_R6B_BATCH','128'))
N_PROBE=int(os.environ.get('SLT_R6B_PROBE','32' if SMOKE else ('128' if DATASET=='mnist' else '64')))
MAX_PAIRS=int(os.environ.get('SLT_R6B_MAX_PAIRS','2' if SMOKE else '10'))
RIDGE=float(os.environ.get('SLT_R6B_RIDGE','1e-5')); N_SHUFFLE=int(os.environ.get('SLT_R6B_SHUFFLE','20' if SMOKE else '2000'))



def balanced_directed_pairs(n,max_pairs=None):
    pairs=[]
    for d in range(1,n):
        for a in range(n): pairs.append((a,(a+d)%n))
    return pairs if max_pairs is None else pairs[:int(max_pairs)]

def response_clusters(model,X,K,seed):
    Fm=feature_matrix(model,X,arch=ARCH,dataset=DATASET,max_n=len(X),device=DEVICE).cpu().numpy().T
    Z=Fm-Fm.mean(1,keepdims=True); Z=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-12)
    kk=min(int(K),len(Z)); return KMeans(n_clusters=kk,n_init=20,random_state=seed).fit_predict(Z)


def aggregate_strength(x,labels,K):
    x=np.asarray(x,float); return np.array([np.nansum(x[labels==k]) for k in range(K)],float)


def aggregate_definition4(alpha,labels,xA0,xB0,K):
    """Pre-B state-weighted coarse-graining of the *fixed* feature-level alpha.

    From sum_i x_i^A sum_j alpha_ij x_j^B ~= X_Ak sum_l A_kl X_Bl,
    freeze the aggregation weights at the pre-B state.  This is a declared
    coarse-graining approximation; no trajectory values are used to choose A_kl.
    """
    A=np.asarray(alpha,float); xA0=np.asarray(xA0,float); xB0=np.asarray(xB0,float)
    out=np.full((K,K),np.nan,float); coverage=np.zeros((K,K),float)
    for ka in range(K):
        I=np.where(labels==ka)[0]
        for kb in range(K):
            J=np.where(labels==kb)[0]
            sub=A[np.ix_(I,J)]; w=np.outer(xA0[I],xB0[J]); ok=np.isfinite(sub)&np.isfinite(w)&(w>=0)
            den=float(w[ok].sum())
            if den>1e-18:
                out[ka,kb]=float((w[ok]*sub[ok]).sum()/den); coverage[ka,kb]=float(ok.mean())
            elif np.isfinite(sub).any():
                out[ka,kb]=float(np.nanmean(sub)); coverage[ka,kb]=float(np.isfinite(sub).mean())
    return out,coverage




def coarse_graining_pressure_diagnostics(alpha_feat,rawA,rawB,labels,Acluster):
    """Check whether frozen cluster aggregation preserves Definition-4 pressure.

    This does NOT change alpha or the confirmatory prediction.  It only tells us
    whether a negative clustered result could plausibly be caused by the
    coarse-graining approximation itself.  Exact cluster pressure is the
    resident-strength-weighted mean of feature-level pressure within each
    cluster.
    """
    A=np.asarray(alpha_feat,float); rawA=np.asarray(rawA,float); rawB=np.asarray(rawB,float)
    K=Acluster.shape[0]; exact=[]; coarse=[]
    for t in range(len(rawA)):
        pf=A@rawB[t]
        xb=np.array([np.nansum(rawB[t][labels==k]) for k in range(K)],float)
        pc=Acluster@xb
        pe=[]
        for k in range(K):
            I=np.where(labels==k)[0]; den=float(np.nansum(rawA[t][I]))
            pe.append(float(np.nansum(rawA[t][I]*pf[I])/(den+1e-12)))
        exact.extend(pe); coarse.extend(pc.tolist())
    x=np.asarray(exact,float); y=np.asarray(coarse,float); ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3:
        return {'n':int(ok.sum()),'pearson':None,'relative_MAE':None,'sign_agreement':None}
    xx=x[ok]; yy=y[ok]
    if np.std(xx)<1e-12 or np.std(yy)<1e-12: corr=None
    else: corr=float(np.corrcoef(xx,yy)[0,1])
    return {'n':int(ok.sum()),'pearson':corr,'relative_MAE':float(np.mean(np.abs(xx-yy))/(np.mean(np.abs(xx))+1e-12)),
            'sign_agreement':float(np.mean(np.sign(xx)==np.sign(yy)))}

def r2(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p)
    if ok.sum()<2: return float('nan')
    yy=y[ok]; pp=p[ok]; den=float(((yy-yy.mean())**2).sum())
    return float('nan') if den<1e-14 else float(1-((yy-pp)**2).sum()/den)


def fit_unconstrained_lv(XA,XB,taus,ridge=1e-5,cond_max=1e8,min_points_per_param=1.5):
    """Flexible comparator: fit g and alpha independently per resident cluster.

    IDENTIFIABILITY IS REPORTED EXPLICITLY. The flexible arm fits K+1 parameters
    per cluster (one g plus K alphas), i.e. K*(K+1) in total, whereas the
    fixed-Definition-4 arm fits only K scalars. With T_train points the flexible
    arm is underdetermined once K+1 > T_train, and near-singular well before
    that. Returning a bare number in that regime would let a PARAMETER-COUNTING
    failure be read as a MODEL failure -- a different scientific claim.

    So: always return flex_status, n_train_points, n_parameters_per_cluster,
    points_per_parameter, rank and condition_number. Only flex_status ==
    'identifiable' may be compared against the fixed-alpha arm.
    """
    XA=np.asarray(XA,float); XB=np.asarray(XB,float); taus=np.asarray(taus,float); Tn,K=XA.shape
    n_par=K+1
    base={'flex_status':None,'n_time_points':int(Tn),'n_parameters_per_cluster':int(n_par),
          'n_parameters_total':int(K*n_par),'cond_max':float(cond_max),
          'min_points_per_param':float(min_points_per_param)}
    split=max(K+3,int(round(.60*Tn))); split=min(split,Tn-3)
    if split<=K+1 or Tn-split<3:
        base.update({'flex_status':'insufficient_time_points','n_train_points':int(max(split,0)),
                     'points_per_parameter':None,'heldout_R2_median':None,'error':'insufficient time points'})
        return base
    n_tr=int(split); ppp=float(n_tr)/float(n_par)
    base.update({'n_train_points':n_tr,'points_per_parameter':ppp})
    if ppp < min_points_per_param:
        # Underdetermined by construction. Report and do NOT fit.
        base.update({'flex_status':'non_identifiable_underdetermined','heldout_R2_median':None,
                     'rank':None,'condition_number':None})
        return base
    rr=[]; cond=[]; neg=[]; ranks=[]
    for k in range(K):
        y=np.gradient(np.log(np.clip(XA[:,k],1e-12,None)),taus)
        X=np.c_[np.ones(Tn),-XB]; Xtr=X[:split]; ytr=y[:split]
        G=Xtr.T@Xtr
        cond.append(float(np.linalg.cond(G))); ranks.append(int(np.linalg.matrix_rank(Xtr)))
        try:
            w=np.linalg.solve(G+ridge*np.eye(n_par),Xtr.T@ytr)
        except Exception as e:                                    # pragma: no cover
            base.update({'flex_status':'solver_failure','exception':repr(e),
                         'heldout_R2_median':None,'condition_number':float(np.median(cond)),
                         'rank':int(np.median(ranks)) if ranks else None})
            return base
        rr.append(r2(y[split:],X[split:]@w)); neg.extend((w[1:]<0).tolist())
    cmed=float(np.median(cond)); rmed=int(np.median(ranks))
    status='identifiable'
    if rmed < n_par:                    status='non_identifiable_rank_deficient'
    elif cmed > cond_max:               status='non_identifiable_ill_conditioned'
    elif not np.isfinite(np.nanmedian(rr)): status='degenerate_r2'
    base.update({'flex_status':status,'rank':rmed,'condition_number':cmed,
                 'heldout_R2_median':float(np.nanmedian(rr)),'heldout_R2_mean':float(np.nanmean(rr)),
                 'per_cluster_R2':rr,'condition_median':cmed,
                 'alpha_negative_fraction':float(np.mean(neg))})
    return base


def fixed_alpha_fit(XA,XB,taus,Afix,n_shuffle,seed):
    XA=np.asarray(XA,float); XB=np.asarray(XB,float); Afix=np.asarray(Afix,float); taus=np.asarray(taus,float)
    Tn,K=XA.shape; split=max(4,int(round(.60*Tn))); split=min(split,Tn-3)
    if Tn-split<3 or Afix.shape!=(K,K) or not np.all(np.isfinite(Afix)):
        return {'error':'invalid fixed-alpha system or insufficient time points'}
    ys=np.stack([np.gradient(np.log(np.clip(XA[:,k],1e-12,None)),taus) for k in range(K)],axis=1)
    P=XB@Afix.T                                  # [T,K], P_k=sum_l alpha_kl X_Bl
    g=np.nanmean(ys[:split]+P[:split],axis=0)    # ONLY fitted parameters
    pred=g[None,:]-P
    # constant-rate null fits its own c_k without using alpha
    c=np.nanmean(ys[:split],axis=0); pred_const=np.broadcast_to(c,(Tn,K))
    # P=0 null uses theory-fitted g but deletes competitive pressure
    pred_p0=np.broadcast_to(g,(Tn,K))
    per=[r2(ys[split:,k],pred[split:,k]) for k in range(K)]
    per_const=[r2(ys[split:,k],pred_const[split:,k]) for k in range(K)]
    per_p0=[r2(ys[split:,k],pred_p0[split:,k]) for k in range(K)]
    obs=float(np.nanmedian(per)); rng=np.random.RandomState(seed); sh=[]
    for _ in range(int(n_shuffle)):
        perm=rng.permutation(K); Ash=Afix[:,perm]
        Psh=XB@Ash.T; gsh=np.nanmean(ys[:split]+Psh[:split],axis=0); pr=gsh[None,:]-Psh
        sh.append(float(np.nanmedian([r2(ys[split:,k],pr[split:,k]) for k in range(K)])))
    sh=np.asarray(sh,float); good=np.isfinite(sh)
    p=float((1+np.sum(sh[good]>=obs))/(1+good.sum())) if good.any() and np.isfinite(obs) else float('nan')
    return {'split_index':split,'n_timepoints':Tn,'g_hat':g.tolist(),
            'heldout_R2_median':obs,'heldout_R2_mean':float(np.nanmean(per)),'per_cluster_R2':per,
            'constant_rate_R2_median':float(np.nanmedian(per_const)),'constant_rate_per_cluster_R2':per_const,
            'P_zero_R2_median':float(np.nanmedian(per_p0)),'P_zero_per_cluster_R2':per_p0,
            'shuffled_alpha_R2_median_mean':float(np.nanmean(sh)) if good.any() else None,
            'shuffled_alpha_R2_median_std':float(np.nanstd(sh)) if good.any() else None,
            'p_fixed_alpha_beats_shuffle':p,'n_shuffle':int(good.sum())}


SHARED_RESIDENT_DIR=os.environ.get('SLT_SHARED_RESIDENT_DIR'); SHARED_RESIDENT_MODE=os.environ.get('SLT_SHARED_RESIDENT_MODE','')

def run_pair(tasks,a,b,seed):
    set_seed(seed); m=make_model(ARCH,len(tasks),2,scenario='task',dataset=DATASET).to(DEVICE)
    ckpt=(Path(SHARED_RESIDENT_DIR)/f'seed{seed}_task{a}.pt') if SHARED_RESIDENT_DIR else None
    if ckpt is not None and SHARED_RESIDENT_MODE=='load' and not ckpt.exists() and SMOKE:
        print(f'[smoke] shared checkpoint {ckpt} missing (T1 smoke pair subset != R6b pair subset); training locally',flush=True); ckpt=None
    if ckpt is not None and SHARED_RESIDENT_MODE=='load':
        if not ckpt.exists(): raise RuntimeError(f'missing shared A1 resident checkpoint: {ckpt}')
        obj=torch.load(ckpt,map_location=DEVICE); m.load_state_dict(obj['state_dict'] if isinstance(obj,dict) and 'state_dict' in obj else obj)
    else:
        train_sgd_theory(m,a,*tasks[a],arch=ARCH,dataset=DATASET,epochs=EPOCHS_A,lr=LR,batch=BATCH,seed=seed,freeze_bn=(ARCH=='resnet18'))
    XAp,yAp,_=stratified_probe(*tasks[a],N_PROBE,seed=seed+100+a); XBp,yBp,_=stratified_probe(*tasks[b],N_PROBE,seed=seed+200+b)
    labels_by_k={K:response_clusters(m,XAp,K,seed+K) for K in KS}
    kw=dict(arch=ARCH,dataset=DATASET,reshape_fn=_reshape_for,device=DEVICE)
    # Definition-4 alpha is frozen BEFORE Task-B training.
    JA=T.feature_jacobians(m,XAp,n_probe=len(XAp),**kw); JB=T.feature_jacobians(m,XBp,n_probe=len(XBp),**kw)
    alpha_feat=T.competitive_alpha_AB(JA,JB)
    xA0=T.feature_strength(m,XAp,max_n=len(XAp),**kw); xB0=T.feature_strength(m,XBp,max_n=len(XBp),**kw)

    _set_trainable(m,b); opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=LR,momentum=0.0)
    XBr=_reshape_for(ARCH,DATASET,tasks[b][0]); yB=tasks[b][1]; rng=np.random.RandomState(seed+3)
    rawA=[]; rawB=[]; taus=[]
    def snap(step):
        rawA.append(T.feature_strength(m,XAp,max_n=len(XAp),**kw)); rawB.append(T.feature_strength(m,XBp,max_n=len(XBp),**kw)); taus.append(T.lv_time(step,LR))
    snap(0)
    for step in range(1,N_STEPS+1):
        idx=rng.choice(len(XBr),min(BATCH,len(XBr)),replace=len(XBr)<BATCH)
        xb=torch.as_tensor(XBr[idx],dtype=torch.float32,device=DEVICE); yb=torch.as_tensor(yB[idx],dtype=torch.long,device=DEVICE)
        m.train();
        if ARCH=='resnet18': freeze_batchnorm_stats(m)
        loss=F.cross_entropy(m.heads[b](m.features(xb)),yb); opt.zero_grad(); loss.backward(); opt.step()
        if step%CHECK==0 or step==N_STEPS: snap(step)
    rawA=np.asarray(rawA); rawB=np.asarray(rawB); fits=[]
    for K,lab in labels_by_k.items():
        kk=int(lab.max()+1); XA=np.stack([aggregate_strength(x,lab,kk) for x in rawA]); XB=np.stack([aggregate_strength(x,lab,kk) for x in rawB])
        Acov,cov=aggregate_definition4(alpha_feat,lab,xA0,xB0,kk)
        fixed=fixed_alpha_fit(XA,XB,taus,Acov,N_SHUFFLE,seed+K)
        flex=fit_unconstrained_lv(XA,XB,taus,RIDGE)
        coarse_diag=coarse_graining_pressure_diagnostics(alpha_feat,rawA,rawB,lab,Acov)
        fx=fixed.get('heldout_R2_median'); shm=fixed.get('shuffled_alpha_R2_median_mean')
        primary={'delta_fixed_minus_shuffled':(float(fx-shm) if (fx is not None and shm is not None) else None),
                 'p_shuffle':fixed.get('p_fixed_alpha_beats_shuffle'),
                 'n_shuffle':fixed.get('n_shuffle',N_SHUFFLE),
                 'p_shuffle_floor':1.0/(float(fixed.get('n_shuffle',N_SHUFFLE))+1.0),
                 'flex_comparable':bool(flex.get('flex_status')=='identifiable'),
                 'note':'PRIMARY comparison is fixed-alpha vs shuffled-alpha (identical parameter count and '
                        'conditioning), which isolates whether Definition 4 interaction STRUCTURE carries '
                        'information. R2>0 is secondary. The flexible arm is only a valid comparator when '
                        'flex_comparable is true.'}
        fits.append({'K':kk,'fixed_definition4':fixed,'unconstrained_lv':flex,'primary_comparison':primary,
                     'coarse_graining_pressure_diagnostics':coarse_diag,
                     'alpha_fixed':Acov.tolist(),
                     'alpha_negative_fraction':float(np.nanmean(Acov<0)),'alpha_finite_fraction':float(np.isfinite(Acov).mean()),
                     'aggregation_coverage_mean':float(np.mean(cov)),'cluster_sizes':[int((lab==j).sum()) for j in range(kk)]})
    return {'task_a':a,'task_b':b,'seed':seed,'resident_source':('shared_checkpoint' if ckpt is not None and SHARED_RESIDENT_MODE=='load' else 'fresh_train'),'fits':fits,'taus':taus,
            'definition4_feature_alpha_diagnostics':T.alpha_diagnostics(alpha_feat[np.isfinite(alpha_feat).all(axis=1)] if np.isfinite(alpha_feat).all(axis=1).any() else np.nan_to_num(alpha_feat))}


def main():
    t0=time.time(); Xtr,ytr,_,_=load_np(DATASET); tasks=[binary_subset(Xtr,ytr,*cs) for cs in SPLITS[DATASET]]
    pairs=balanced_directed_pairs(len(tasks),MAX_PAIRS); rows=[]
    for si in range(SEEDS):
        for a,b in pairs:
            r=run_pair(tasks,a,b,BASE+si); rows.append(r)
            print(DATASET,f'T{a}->T{b}',', '.join([f"K{z['K']}: fixedR2={z['fixed_definition4'].get('heldout_R2_median')} psh={z['fixed_definition4'].get('p_fixed_alpha_beats_shuffle')} flexR2={z['unconstrained_lv'].get('heldout_R2_median')} flexStatus={z['unconstrained_lv'].get('flex_status')}" for z in r['fits']]),flush=True)
    summary={}
    for K in KS:
        zs=[z for r in rows for z in r['fits'] if z['K']==K and 'error' not in z['fixed_definition4']]
        if zs:
            flex_ident=[z for z in zs if z['unconstrained_lv'].get('flex_status')=='identifiable'
                        and z['unconstrained_lv'].get('heldout_R2_median') is not None
                        and np.isfinite(z['unconstrained_lv'].get('heldout_R2_median'))]
            status_counts={}
            for z in zs:
                st=z['unconstrained_lv'].get('flex_status','missing')
                status_counts[st]=status_counts.get(st,0)+1
            pvals=[z['fixed_definition4'].get('p_fixed_alpha_beats_shuffle') for z in zs]
            pvals=[float(v) for v in pvals if v is not None and np.isfinite(v)]
            summary[f'K{K}']={'n':len(zs),
                'fixed_R2_median_across_pairs':float(np.nanmedian([z['fixed_definition4']['heldout_R2_median'] for z in zs])),
                'constant_rate_R2_median_across_pairs':float(np.nanmedian([z['fixed_definition4']['constant_rate_R2_median'] for z in zs])),
                'P_zero_R2_median_across_pairs':float(np.nanmedian([z['fixed_definition4']['P_zero_R2_median'] for z in zs])),
                'unconstrained_R2_median_identifiable_only':(float(np.nanmedian([z['unconstrained_lv']['heldout_R2_median'] for z in flex_ident])) if flex_ident else None),
                'n_unconstrained_identifiable':len(flex_ident),
                'unconstrained_status_counts':status_counts,
                'fraction_pairs_fixed_beats_shuffle_p05':(float(np.mean(np.asarray(pvals)<.05)) if pvals else None)}
    out={'rows':rows,'summary':summary,'_meta':run_metadata(ROOT,{'experiment':'R6b','dataset':DATASET,'K_grid':KS,'steps':N_STEPS,'check_every':CHECK,'probe':N_PROBE,'n_shuffle':N_SHUFFLE,'alpha':'Definition 4 fixed pre-B; only g fitted','elapsed_s':time.time()-t0})}
    json_dump_safe(out,f'r6b_fixed_definition4_lv_{DATASET}_results.json'); print('saved R6b fixed-alpha LV results')

if __name__=='__main__': main()
