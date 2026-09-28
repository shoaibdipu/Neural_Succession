#!/usr/bin/env python3
from __future__ import annotations
import copy,os,sys,time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_confirmatory import stratified_probe,run_metadata
from slt_freeze import *

EXP='A7_succession_stages'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42')); SEEDS=int(os.environ.get('SLT_A7_SEEDS','1' if SMOKE else '3'))
A_CK=[0,1,2] if SMOKE else [0,1,5,10,20]; B_CK=[0,1,2] if SMOKE else [0,1,5,10]; BATCH=128

def train_epochs(m,t,X,y,n,opt,arch='resnet18',dataset='cifar10'):
    _set_trainable(m,t); xr=_reshape_for(arch,dataset,X); dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(xr),torch.tensor(y)),batch_size=BATCH,shuffle=True)
    for _ in range(n):
        m.train()
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(); loss=F.cross_entropy(m.heads[t](m.features(xb)),yb); loss.backward(); opt.step()

def snap(m,X,prev=None):
    Fm=feature_matrix(m,X,arch='resnet18',dataset='cifar10',max_n=len(X)).cpu().numpy(); x=Fm.mean(0)**2; active=x>=np.quantile(x,.75); diversity=float((x.sum()**2)/(np.sum(x*x)+1e-12)); Z=Fm-Fm.mean(0,keepdims=True); C=np.corrcoef(Z,rowvar=False); off=C[~np.eye(C.shape[0],dtype=bool)]; overlap=float(np.nanmean(np.abs(off)))
    d={'strength':x,'active':active,'diversity':diversity,'overlap':overlap}
    if prev is not None:
        d['plasticity']=float(np.mean(np.abs(np.log(np.clip(x,1e-12,None)/np.clip(prev['strength'],1e-12,None))))); inter=np.sum(active&prev['active']); union=np.sum(active|prev['active']); d['turnover']=float(1-inter/max(union,1))
    else: d['plasticity']=None; d['turnover']=None
    return d

def main():
    t0=time.time(); Xtr,ytr,_,_=load_np('cifar10'); tasks=make_tasks(Xtr,ytr,SPLITS['cifar10']); rows=[]
    for si in range(SEEDS):
        seed=BASE+si
        for a in range(5):
            b=(a+1)%5; set_seed(seed+100*a); m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); Xp,yp,_=stratified_probe(*tasks[a],256 if not SMOKE else 64,seed+1000+a); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
            prev=None; A=[]; last=0
            for ck in A_CK:
                if ck>last: train_epochs(m,a,*tasks[a],ck-last,opt); last=ck
                s=snap(m,Xp,prev); A.append({'epoch':ck,**{k:v for k,v in s.items() if k not in ('strength','active')}}); prev=s
            pre=prev; optB=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4); B=[]; last=0; prev=pre
            for ck in B_CK:
                if ck>last: train_epochs(m,b,*tasks[b],ck-last,optB); last=ck
                s=snap(m,Xp,prev); B.append({'epoch':ck,**{k:v for k,v in s.items() if k not in ('strength','active')}}); prev=s
            rows.append({'seed':seed,'task_a':a,'task_b':b,'A':A,'B':B})
            print(seed,a,'Aearly',A[min(1,len(A)-1)].get('plasticity'),'Alate',A[-1].get('plasticity'),'Binv',B[min(1,len(B)-1)].get('plasticity'),flush=True)
    # predefined contrasts: transition after first A epoch vs last A interval; first B interval vs late-A interval.
    vals={'A_plasticity_early':[],'A_plasticity_late':[],'A_turnover_early':[],'A_turnover_late':[],'B_plasticity_invasion':[],'B_turnover_invasion':[]}
    for r in rows:
        A=r['A']; B=r['B']; vals['A_plasticity_early'].append(A[1]['plasticity']); vals['A_plasticity_late'].append(A[-1]['plasticity']); vals['A_turnover_early'].append(A[1]['turnover']); vals['A_turnover_late'].append(A[-1]['turnover']); vals['B_plasticity_invasion'].append(B[1]['plasticity']); vals['B_turnover_invasion'].append(B[1]['turnover'])
    def diff(a,b):
        d=np.asarray(vals[a])-np.asarray(vals[b]); return {'mean_diff':float(d.mean()),'ci95':quantile_ci(d),'fraction_positive':float(np.mean(d>0))}
    contrasts={'early_vs_late_plasticity':diff('A_plasticity_early','A_plasticity_late'),'early_vs_late_turnover':diff('A_turnover_early','A_turnover_late'),'invasion_vs_preB_plasticity':diff('B_plasticity_invasion','A_plasticity_late'),'invasion_vs_preB_turnover':diff('B_turnover_invasion','A_turnover_late')}
    out={'rows':rows,'contrasts':contrasts,'protocol':{'A_checkpoints':A_CK,'B_checkpoints':B_CK,'active_definition':'top quartile resident feature strength','plasticity':'mean abs log-strength change','turnover':'1-Jaccard active sets'},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}; p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
