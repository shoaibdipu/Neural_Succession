#!/usr/bin/env python3
"""A3 — final neural R* test using task-relevant input-gradient covariance resources.
The behavioral coexistence outcome is recomputed inside this job; no R1 result file is read.
"""
from __future__ import annotations
import copy,os,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy.stats import fisher_exact
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core')); sys.path.insert(0,str(ROOT/'experiments/theory_validation'))
from slt_common import *
from slt_confirmatory import stratified_probe,train_sgd_theory,run_metadata
from slt_freeze import *
import T3b_rstar_zero_crossing as OLD

EXP='A3_rstar_task_resources'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_A3_SEEDS','1' if SMOKE else '3')); NPROBE=int(os.environ.get('SLT_A3_PROBE','128' if SMOKE else '512')); NRES=int(os.environ.get('SLT_A3_RESOURCES','2' if SMOKE else '8'))
EPOCHS=int(os.environ.get('SLT_A3_EPOCHS','3' if SMOKE else '15')); MAX_CASES=2 if SMOKE else 0

def input_gradient_basis(model,task_id,X,y,head,nres):
    m=copy.deepcopy(model).to(DEVICE); m.heads[task_id].load_state_dict(head.state_dict()); m.eval(); xr=_reshape_for('mlp','mnist',X); grads=[]
    # per-sample input gradients define the task-relevant covariance in the same input space as phi_i
    for i in range(len(xr)):
        xb=torch.as_tensor(xr[i:i+1],dtype=torch.float32,device=DEVICE).requires_grad_(True); yy=torch.as_tensor(y[i:i+1],dtype=torch.long,device=DEVICE)
        loss=F.cross_entropy(m.heads[task_id](m.features(xb)),yy); g=torch.autograd.grad(loss,xb)[0]
        grads.append(g.detach().cpu().reshape(-1).numpy())
    G=np.stack(grads); G-=G.mean(0,keepdims=True); _,_,Vt=np.linalg.svd(G,full_matrices=False); mu=np.mean(X.reshape(len(X),-1),0)
    return mu,Vt[:min(nres,len(Vt))]

def estimate_curve(resident,invader_task,X,y,head,mu,v,seed):
    # use the already-frozen T3b monotonicity/bracketing logic
    rng=np.random.RandomState(seed); curves=[]; reps=2 if SMOKE else 7; grid=np.array([0,.25,.5,.75,1,1.25,1.5,1.75,2.0]); sub=64 if SMOKE else 128
    for rr in range(reps):
        idx=rng.choice(len(X),min(sub,len(X)),replace=False); curves.append([OLD.growth_once(resident,invader_task,X,y,head,mu,v,s,idx) for s in grid])
    arr=np.asarray(curves,float); med=np.nanmedian(arr,0); rho=stats.spearmanr(grid,med,nan_policy='omit').statistic if np.isfinite(med).sum()>=3 else np.nan; brackets=bool(np.nanmin(med)<0<=np.nanmax(med)) if np.isfinite(med).any() else False
    identified=bool(np.isfinite(rho) and rho>=.5 and brackets); crossing=None; reps_cross=[]
    if identified:
        yy=IsotonicRegression(increasing=True,out_of_bounds='clip').fit_transform(grid,med); crossing=OLD.crossing_linear(grid,yy)
        for cur in arr:
            rr=stats.spearmanr(grid,cur,nan_policy='omit').statistic if np.isfinite(cur).sum()>=3 else np.nan
            if np.isfinite(rr) and rr>=.5:
                zz=IsotonicRegression(increasing=True,out_of_bounds='clip').fit_transform(grid,cur); c=OLD.crossing_linear(grid,zz)
                if c is not None: reps_cross.append(c)
        if len(reps_cross)/reps<.5: crossing=None; identified=False
    return {'identified':identified,'R_star':crossing,'spearman':None if not np.isfinite(rho) else float(rho),'brackets_zero':brackets,'rep_crossings':reps_cross}


def build_cases_full(Xtr,ytr,Xte,yte):
    cases=[]
    cps=OLD.CLASS_PAIRS[:1] if SMOKE else OLD.CLASS_PAIRS
    tfs=[('identity',0),('perm',1.0)] if SMOKE else OLD.TRANSFORMS
    for c0,c1 in cps:
        XA,yA=binary_subset(Xtr,ytr,c0,c1); XAt,yAt=binary_subset(Xte,yte,c0,c1)
        for kind,val in tfs:
            XB=OLD.transform(XA,kind,val); XBt=OLD.transform(XAt,kind,val)
            cases.append({'pair_id':f'{c0}/{c1}__{kind}{val}','XA':XA,'yA':yA,'XB':XB,'yB':yA.copy(),
                          'XAte':XAt,'yAte':yAt,'XBte':XBt,'yBte':yAt.copy(),
                          'class_pair':f'{c0}/{c1}','transform':f'{kind}{val}'})
    return cases

def behavioral(case,seed):
    XA,yA,XB,yB=case['XA'],case['yA'],case['XB'],case['yB']; XAt,yAt,XBt,yBt=case['XAte'],case['yAte'],case['XBte'],case['yBte']
    # deterministic held-out probes from full task sets; R1-style same underlying dataset
    set_seed(seed); mb=make_model('mlp',2,2,scenario='task',dataset='mnist').to(DEVICE); train_task(mb,1,XB,yB,arch='mlp',dataset='mnist',epochs=20 if not SMOKE else 2,lr=1e-3,batch=128,wd=1e-4); RBB=accuracy(mb,1,XBt,yBt,arch='mlp',dataset='mnist')
    set_seed(seed); m=make_model('mlp',2,2,scenario='task',dataset='mnist').to(DEVICE); train_task(m,0,XA,yA,arch='mlp',dataset='mnist',epochs=20 if not SMOKE else 2,lr=1e-3,batch=128,wd=1e-4); RAA=accuracy(m,0,XAt,yAt,arch='mlp',dataset='mnist'); train_task(m,1,XB,yB,arch='mlp',dataset='mnist',epochs=20 if not SMOKE else 2,lr=1e-3,batch=128,wd=1e-4); RBA=accuracy(m,0,XAt,yAt,arch='mlp',dataset='mnist'); RBBseq=accuracy(m,1,XBt,yBt,arch='mlp',dataset='mnist')
    return {'R_AA':RAA,'R_BA':RBA,'R_BB_ref':RBB,'R_BB_seq':RBBseq,'coexist_0p90':bool(RBA/(RAA+1e-12)>=.9 and RBBseq/(RBB+1e-12)>=.9)}

def main():
    from scipy import stats
    globals()['stats']=stats
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('mnist'); cases=build_cases_full(Xtr,ytr,Xte,yte); cases=cases[:MAX_CASES] if MAX_CASES else cases; rows=[]
    for si in range(SEEDS):
        seed=BASE+si
        for ci,c in enumerate(cases):
            XA,yA,_=stratified_probe(c['XA'],c['yA'],NPROBE,seed+10+ci); XB,yB,_=stratified_probe(c['XB'],c['yB'],NPROBE,seed+20+ci)
            mA=OLD.resident_on_data(c['XA'],c['yA'],0,seed+100+ci); mB=OLD.resident_on_data(c['XB'],c['yB'],1,seed+200+ci)
            hB=OLD.fit_probe(mA,1,XB,yB); hA=OLD.fit_probe(mB,0,XA,yA); mu,V=input_gradient_basis(mA,1,XB,yB,hB,NRES)
            ra=[]; rb=[]
            for rid,v in enumerate(V):
                ra.append(estimate_curve(mB,0,XA,yA,hA,mu,v,seed+1000+rid+ci*20)); rb.append(estimate_curve(mA,1,XB,yB,hB,mu,v,seed+2000+rid+ci*20))
            aw=[]; bw=[]
            for rid,(aa,bb) in enumerate(zip(ra,rb)):
                if aa['identified'] and bb['identified'] and aa['R_star'] is not None and bb['R_star'] is not None:
                    (aw if aa['R_star']<bb['R_star'] else bw).append(rid) if aa['R_star']!=bb['R_star'] else None
            beh=behavioral(c,seed); part=bool(aw and bw); rows.append({'seed':seed,'pair_id':c['pair_id'],'rstar_A':ra,'rstar_B':rb,'A_superior':aw,'B_superior':bw,'niche_partition':part,**beh})
            print(c['pair_id'],'id',sum(x['identified'] for x in ra),sum(x['identified'] for x in rb),'part',part,'coex',beh['coexist_0p90'],flush=True)
    ident=[(sum(x['identified'] for x in r['rstar_A'])+sum(x['identified'] for x in r['rstar_B']))/(2*NRES) for r in rows]
    y=np.asarray([r['coexist_0p90'] for r in rows],int); p=np.asarray([r['niche_partition'] for r in rows],int)
    tp=int(np.sum((p==1)&(y==1))); fp=int(np.sum((p==1)&(y==0))); fn=int(np.sum((p==0)&(y==1))); tn=int(np.sum((p==0)&(y==0))); base=float(y.mean()) if len(y) else None; prec=tp/max(tp+fp,1)
    _,fpv=fisher_exact([[tp,fp],[fn,tn]]) if len(np.unique(y))==2 and len(np.unique(p))==2 else (None,None)
    summary={'mean_identified_resource_fraction':float(np.mean(ident)) if ident else None,'partition_precision':prec,'behavior_base_rate':base,'precision_minus_base':None if base is None else prec-base,'fisher_p':None if fpv is None else float(fpv),'pass':bool(ident and np.mean(ident)>=.5 and base is not None and prec>=base+.15 and fpv is not None and fpv<.05)}
    out={'rows':rows,'summary':summary,'protocol':{'resources':'top input-gradient covariance eigendirections from incoming-task gradients at resident state','resource_symmetry':'asymmetric incoming-task-only basis; frozen predeclared choice','n_resources':NRES},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}; pth=dump_result(ROOT,EXP,out); print('saved',pth)
if __name__=='__main__': main()
