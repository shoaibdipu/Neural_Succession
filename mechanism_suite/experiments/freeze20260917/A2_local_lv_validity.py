#!/usr/bin/env python3
from __future__ import annotations
import copy,os,sys,time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_confirmatory import train_sgd_theory,freeze_batchnorm_stats,stratified_probe,run_metadata
from slt_theory import competitive_alpha_AB
from slt_freeze import *

EXP='A2_local_lv_validity'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_A2_SEEDS','1' if SMOKE else '3')); PAIRS=fixed_pairs(5,2 if SMOKE else int(os.environ.get('SLT_A2_PAIRS','10')))
EPOCHS_A=int(os.environ.get('SLT_A2_A_EPOCHS','2' if SMOKE else '20')); LR=float(os.environ.get('SLT_A2_LR','0.03')); BATCH=32
NSTEPS=int(os.environ.get('SLT_A2_STEPS','150' if SMOKE else '600')); SNAP=int(os.environ.get('SLT_A2_SNAP','10')); ALPHA_EVERY=50; WINDOW=100
NPROBE=int(os.environ.get('SLT_A2_PROBE','32' if SMOKE else '64')); NFEAT=int(os.environ.get('SLT_A2_NFEAT','16' if SMOKE else '64')); K=int(os.environ.get('SLT_A2_K','4')); NSHUF=int(os.environ.get('SLT_A2_SHUFFLE','50' if SMOKE else '500'))

def r2(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); den=np.sum((y-y.mean())**2); return None if den<1e-12 else float(1-np.sum((y-p)**2)/den)

def cluster_selected(model,X,ids):
    Fm=feature_matrix(model,X,arch='resnet18',dataset='cifar10',max_n=len(X)).cpu().numpy()[:,ids].T
    Z=Fm-Fm.mean(1,keepdims=True); Z/=np.linalg.norm(Z,axis=1,keepdims=True)+1e-12
    return KMeans(n_clusters=min(K,len(ids)),n_init=20,random_state=BASE).fit_predict(Z)

def aggregate_alpha(alpha,labels,xA,xB):
    kk=int(labels.max()+1); out=np.zeros((kk,kk))
    for ka in range(kk):
        I=np.where(labels==ka)[0]
        for kb in range(kk):
            J=np.where(labels==kb)[0]; sub=alpha[np.ix_(I,J)]; w=np.outer(xA[I],xB[J]); den=w.sum(); out[ka,kb]=(sub*w).sum()/(den+1e-12)
    return out

def aggr(x,lab): return np.asarray([np.sum(x[lab==k]) for k in range(int(lab.max()+1))],float)

def window_fit(XA,XB,tau,A,seed):
    y=np.stack([np.gradient(np.log(np.clip(XA[:,k],1e-12,None)),tau) for k in range(XA.shape[1])],1); n=len(tau); split=max(3,n//2)
    P=XB@A.T; g=np.mean(y[:split]+P[:split],0); pred=g-P; c=np.mean(y[:split],0)
    obs=np.nanmedian([r2(y[split:,k],pred[split:,k]) for k in range(XA.shape[1])]); const=np.nanmedian([r2(y[split:,k],np.full(n-split,c[k])) for k in range(XA.shape[1])])
    rng=np.random.RandomState(seed); sh=[]
    for _ in range(NSHUF):
        Ash=A[:,rng.permutation(A.shape[1])]; Ps=XB@Ash.T; gs=np.mean(y[:split]+Ps[:split],0); pr=gs-Ps
        sh.append(np.nanmedian([r2(y[split:,k],pr[split:,k]) for k in range(XA.shape[1])]))
    return {'fixed_R2':float(obs),'constant_R2':float(const),'shuffle_R2_mean':float(np.nanmean(sh)),'beats_both':bool(obs>const and obs>np.nanmean(sh))}

def main():
    t0=time.time(); Xtr,ytr,_,_=load_np('cifar10'); tasks=make_tasks(Xtr,ytr,SPLITS['cifar10']); rows=[]
    for si in range(SEEDS):
        seed=BASE+si
        for a,b in PAIRS:
            set_seed(seed); m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE)
            train_sgd_theory(m,a,*tasks[a],arch='resnet18',dataset='cifar10',epochs=EPOCHS_A,lr=LR,batch=BATCH,seed=seed+10*a,freeze_bn=True)
            XAp,yAp,_=stratified_probe(*tasks[a],NPROBE,seed+100+a); XBp,yBp,_=stratified_probe(*tasks[b],NPROBE,seed+200+b)
            xfull=layer_strength(m,XAp,arch='resnet18',dataset='cifar10',layer='feat',max_n=NPROBE); mask,_=feature_floor_mask(xfull); ids=stratified_feature_indices(xfull,mask,NFEAT); labels=cluster_selected(m,XAp,ids)
            _set_trainable(m,b); opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=LR,momentum=0); xr=_reshape_for('resnet18','cifar10',tasks[b][0]); yy=tasks[b][1]; rng=np.random.RandomState(seed+300+b)
            snaps={}; alphas={}
            def take(step,need_alpha=False):
                xA=layer_strength(m,XAp,arch='resnet18',dataset='cifar10',layer='feat',max_n=NPROBE)[ids]; xB=layer_strength(m,XBp,arch='resnet18',dataset='cifar10',layer='feat',max_n=NPROBE)[ids]; snaps[step]=(aggr(xA,labels),aggr(xB,labels))
                if need_alpha:
                    JA=layer_feature_jacobians(m,XAp,arch='resnet18',dataset='cifar10',layer='feat',n_probe=NPROBE,feature_indices=ids); JB=layer_feature_jacobians(m,XBp,arch='resnet18',dataset='cifar10',layer='feat',n_probe=NPROBE,feature_indices=ids)
                    alphas[step]=aggregate_alpha(competitive_alpha_AB(JA,JB),labels,xA,xB)
            take(0,True)
            for st in range(1,NSTEPS+1):
                ii=rng.choice(len(xr),min(BATCH,len(xr)),replace=len(xr)<BATCH); xb=torch.as_tensor(xr[ii],dtype=torch.float32,device=DEVICE); yb=torch.as_tensor(yy[ii],dtype=torch.long,device=DEVICE)
                m.train(); freeze_batchnorm_stats(m); loss=F.cross_entropy(m.heads[b](m.features(xb)),yb); opt.zero_grad(); loss.backward(); opt.step()
                if st%SNAP==0: take(st,st%ALPHA_EVERY==0)
            for s in sorted(k for k in alphas if k+WINDOW in alphas):
                e=s+WINDOW; pts=sorted(k for k in snaps if s<=k<=e); XA=np.stack([snaps[k][0] for k in pts]); XB=np.stack([snaps[k][1] for k in pts]); tau=np.asarray(pts,float)*LR
                A0=alphas[s]; A1=alphas[e]; drift=float(np.linalg.norm(A1-A0)/(np.linalg.norm(A0)+1e-12)); fit=window_fit(XA,XB,tau,A0,seed+s+a*100+b)
                rows.append({'seed':seed,'task_a':a,'task_b':b,'window_start':s,'window_end':e,'alpha_drift':drift,'stable':drift<.1,'unstable':drift>.3,**fit})
                print(seed,a,b,s,'drift',round(drift,3),'R2',round(fit['fixed_R2'],3),flush=True)
    stable=[r for r in rows if r['stable']]; unstable=[r for r in rows if r['unstable']]
    summary={'n_windows':len(rows),'n_stable':len(stable),'stable_fraction_beats_both':float(np.mean([r['beats_both'] for r in stable])) if stable else None,'stable_median_R2':float(np.median([r['fixed_R2'] for r in stable])) if stable else None,'unstable_median_R2':float(np.median([r['fixed_R2'] for r in unstable])) if unstable else None,'pass':bool(stable and np.median([r['fixed_R2'] for r in stable])>0 and np.mean([r['beats_both'] for r in stable])>=.5)}
    out={'rows':rows,'summary':summary,'protocol':{'alpha_every':ALPHA_EVERY,'window':WINDOW,'stable_drift_lt':.1,'unstable_drift_gt':.3,'selected_features':NFEAT,'K':K},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}
    p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
