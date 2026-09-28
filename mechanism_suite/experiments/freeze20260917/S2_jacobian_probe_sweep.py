#!/usr/bin/env python3
from __future__ import annotations
import copy,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_confirmatory import train_sgd_theory, stratified_probe, run_metadata
from slt_revision import partial_corr_residualized
from slt_theory import competitive_alpha_AB
from slt_freeze import *

EXP='S2_jacobian_probe_sweep'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_S2_SEEDS','1' if SMOKE else '3')); PAIRS=fixed_pairs(5,2 if SMOKE else 10)
PROBES=[int(x) for x in os.environ.get('SLT_S2_PROBES','64,256' if SMOKE else '64,256,1024').split(',')]
LAYERS=os.environ.get('SLT_S2_LAYERS','layer3,feat').split(','); NFEAT=int(os.environ.get('SLT_S2_NFEAT','16' if SMOKE else '64'))
EPOCHS=int(os.environ.get('SLT_S2_EPOCHS','2' if SMOKE else '20')); LR=float(os.environ.get('SLT_S2_LR','0.03')); NPERM=int(os.environ.get('SLT_S2_SHUFFLE','100' if SMOKE else '1000'))

def shuffle_p(alpha,xB,decay,control,obs,seed):
    rng=np.random.RandomState(seed); vals=[]
    for _ in range(NPERM):
        pr=alpha[:,rng.permutation(alpha.shape[1])]@xB
        rr=partial_corr_residualized(pr,decay,control).get('partial_r')
        if rr is not None: vals.append(rr)
    vals=np.asarray(vals,float); return None if not len(vals) or obs is None else float((1+np.sum(np.abs(vals)>=abs(obs)))/(1+len(vals)))

def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('cifar10'); tr=make_tasks(Xtr,ytr,SPLITS['cifar10']); rows=[]
    for si in range(SEEDS):
        seed=BASE+si
        for a,b in PAIRS:
            set_seed(seed); mA=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE)
            train_sgd_theory(mA,a,*tr[a],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=LR,batch=32,seed=seed+10*a,freeze_bn=True)
            XAp,yAp,_=stratified_probe(*tr[a],max(PROBES),seed+100+a); XBp,yBp,_=stratified_probe(*tr[b],max(PROBES),seed+200+b)
            m=copy.deepcopy(mA); train_sgd_theory(m,b,*tr[b],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=LR,batch=32,seed=seed+300+b,freeze_bn=True)
            for layer in LAYERS:
                x0_all=layer_strength(mA,XAp,arch='resnet18',dataset='cifar10',layer=layer,max_n=len(XAp)); mask,floor=feature_floor_mask(x0_all)
                ids=stratified_feature_indices(x0_all,mask,NFEAT); x0=x0_all[ids]
                x1=layer_strength(m,XAp,arch='resnet18',dataset='cifar10',layer=layer,max_n=len(XAp))[ids]
                tau=max(EPOCHS*int(np.ceil(len(tr[b][0])/32))*LR,1e-12); decay=-np.log(np.clip(x1,1e-12,None)/np.clip(x0,1e-12,None))/tau
                for npb in PROBES:
                    JA=layer_feature_jacobians(mA,XAp,arch='resnet18',dataset='cifar10',layer=layer,n_probe=npb,feature_indices=ids)
                    JB=layer_feature_jacobians(mA,XBp,arch='resnet18',dataset='cifar10',layer=layer,n_probe=npb,feature_indices=ids)
                    alpha=competitive_alpha_AB(JA,JB); xB=layer_strength(mA,XBp[:npb],arch='resnet18',dataset='cifar10',layer=layer,max_n=npb)[ids]
                    pr=alpha@xB; pc=partial_corr_residualized(pr,decay,np.log(np.clip(x0,1e-12,None))).get('partial_r'); pnull=shuffle_p(alpha,xB,decay,np.log(np.clip(x0,1e-12,None)),pc,seed+npb+a*10+b)
                    rows.append({'seed':seed,'task_a':a,'task_b':b,'layer':layer,'n_probe':npb,'n_features':len(ids),'feature_floor':floor,'partial_r':pc,'p_shuffle':pnull})
                    print(seed,a,b,layer,npb,pc,pnull,flush=True)
    summary={}
    for layer in LAYERS:
        for npb in PROBES:
            z=[r for r in rows if r['layer']==layer and r['n_probe']==npb and r['partial_r'] is not None]
            summary[f'{layer}_n{npb}']={'n':len(z),'median_partial_r':float(np.median([r['partial_r'] for r in z])) if z else None,'fraction_p_shuffle_lt_0p05':float(np.mean([r['p_shuffle']<.05 for r in z if r['p_shuffle'] is not None])) if z else None}
    out={'rows':rows,'summary':summary,'protocol':{'fixed_pairs':PAIRS,'layers':LAYERS,'probe_sizes':PROBES,'n_features_stratified':NFEAT},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}
    p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
