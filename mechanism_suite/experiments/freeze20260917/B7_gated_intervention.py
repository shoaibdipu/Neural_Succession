#!/usr/bin/env python3
"""B7 — shared intervention test: Omega-gated EWC-DR and kappa-gated replay.
Every comparator is trained inside this job; no prior result file is consumed.
"""
from __future__ import annotations
import copy,os,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_revision import rho_pre_permutation_invariant
from slt_confirmatory import stratified_train_val,run_metadata
from slt_freeze import *
from cl_baselines import fisher_normalize,online_ewc_update,train_derpp

EXP='B7_gated_intervention'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42')); SEEDS=int(os.environ.get('SLT_B7_SEEDS','1' if SMOKE else '3')); EPOCHS=int(os.environ.get('SLT_B7_EPOCHS','2' if SMOKE else '20')); BATCH=128; M=int(os.environ.get('SLT_B7_MEMORY_PER_TASK','50' if SMOKE else '200'))
LAM_GRID=[1.,10.] if SMOKE else [0.1,1.,10.,100.]

def resident_loss(model,xb,yb,tb):
    feat=model.features(xb); loss=torch.tensor(0.,device=xb.device); n=float(len(yb))
    for tt in torch.unique(tb):
        mk=tb==tt; loss=loss+(mk.sum().float()/n)*F.cross_entropy(model.heads[int(tt.item())](feat[mk]),yb[mk])
    return loss

def flat_bb_grads(loss,model,retain=True):
    named=backbone_named_parameters(model); gs=torch.autograd.grad(loss,[p for _,p in named],retain_graph=retain,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(named,gs)])

def eval_matrix(model,tasks_te,upto):
    return [accuracy(model,j,*tasks_te[j],arch='resnet18',dataset='cifar10') for j in range(upto+1)]

def calc_acc_bwt(hist):
    T=len(hist); final=hist[-1]; acc=float(np.mean(final)); terms=[]
    for j in range(T-1): terms.append(final[j]-hist[j][j])
    return acc,float(np.mean(terms)) if terms else 0.

def make_memory_append(mem,X,y,t,seed):
    rng=np.random.RandomState(seed); idx=rng.choice(len(X),min(M,len(X)),replace=False); rec=(X[idx],y[idx],np.full(len(idx),t,dtype=np.int64))
    if mem is None: return rec
    return tuple(np.concatenate([mem[i],rec[i]]) for i in range(3))

def train_replay_sequence(tasks,te,seed,mode='kappa',fixed_r=None):
    set_seed(seed); m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); mem=None; hist=[]; rs=[]; extra_backward=0
    for t,(X,y) in enumerate(tasks):
        _set_trainable(m,t); xr=_reshape_for('resnet18','cifar10',X); dl=DataLoader(TensorDataset(torch.tensor(xr),torch.tensor(y)),batch_size=BATCH,shuffle=True); opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4); rng=np.random.RandomState(seed+100*t)
        for _ in range(EPOCHS):
            for xb,yb in dl:
                xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(set_to_none=True); lb=F.cross_entropy(m.heads[t](m.features(xb)),yb)
                if mem is None:
                    r=0.; loss=lb
                else:
                    ids=rng.choice(len(mem[0]),min(BATCH,len(mem[0])),replace=len(mem[0])<BATCH); rx=torch.as_tensor(_reshape_for('resnet18','cifar10',mem[0][ids]),dtype=torch.float32,device=DEVICE); ry=torch.as_tensor(mem[1][ids],dtype=torch.long,device=DEVICE); rt=torch.as_tensor(mem[2][ids],dtype=torch.long,device=DEVICE); lr_=resident_loss(m,rx,ry,rt)
                    if mode=='kappa':
                        gB=flat_bb_grads(lb,m,retain=True); gR=flat_bb_grads(lr_,m,retain=True); extra_backward+=2; r=float(torch.clamp(torch.relu(-torch.dot(gR,gB))/(torch.dot(gR,gR)+1e-12),0,1).detach().cpu())
                    else: r=float(fixed_r)
                    loss=lb+r*lr_; rs.append(r)
                loss.backward(); opt.step()
        mem=make_memory_append(mem,X,y,t,seed+1000+t); hist.append(eval_matrix(m,te,t))
    acc,bwt=calc_acc_bwt(hist); return {'ACC':acc,'BWT':bwt,'history':hist,'mean_r':float(np.mean(rs)) if rs else 0.,'extra_gradient_calls':extra_backward,'memory_examples':0 if mem is None else len(mem[0])}

def ewc_train_sequence(tasks,te,seed,lam0,gated=False):
    set_seed(seed); m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); online_f=None; online_p=None; hist=[]; gates=[]
    for t,(X,y) in enumerate(tasks):
        gate=1.
        if gated and t>0:
            vals=[]
            for j in range(t): vals.append(1-rho_pre_permutation_invariant(m,j,X,y,arch='resnet18',dataset='cifar10',n_classes=2,n_samples=min(256,len(X)),n_shuffle=20,seed=seed+100*j+t))
            gate=float(np.mean(vals)); gates.append(gate)
        lam=lam0*gate; _set_trainable(m,t); xr=_reshape_for('resnet18','cifar10',X); dl=DataLoader(TensorDataset(torch.tensor(xr),torch.tensor(y)),batch_size=BATCH,shuffle=True); opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
        for _ in range(EPOCHS):
            for xb,yb in dl:
                xb=xb.to(DEVICE); yb=yb.to(DEVICE); opt.zero_grad(); loss=F.cross_entropy(m.heads[t](m.features(xb)),yb)
                if online_f is not None:
                    pen=torch.tensor(0.,device=DEVICE)
                    for n,p in m.named_parameters():
                        if n in online_f and p.requires_grad: pen=pen+(online_f[n].to(DEVICE)*(p-online_p[n].to(DEVICE)).pow(2)).sum()
                    loss=loss+lam*pen
                loss.backward(); opt.step()
        _set_trainable(m,t); fi=fisher_normalize(compute_fisher(m,t,X,y,arch='resnet18',dataset='cifar10')); online_f=online_ewc_update(online_f,fi); online_p={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}; hist.append(eval_matrix(m,te,t))
    acc,bwt=calc_acc_bwt(hist); return {'ACC':acc,'BWT':bwt,'history':hist,'lambda0':lam0,'gates':gates}

def tune_lam(tasks,seed,gated):
    tr=[]; va=[]
    for i in range(2): a,b,_,_=stratified_train_val(*tasks[i],.1,seed+i); tr.append(a); va.append(b)
    best=None
    for lam in LAM_GRID:
        rec=ewc_train_sequence(tr,va,seed,lam,gated=gated); score=(np.mean(rec['history'][-1]),rec['history'][-1][0])
        if best is None or score>best[0]: best=(score,lam)
    return float(best[1])

def derpp_sequence(tasks,te,seed):
    # reuse the established DER++ implementation with the same per-task memory cap
    set_seed(seed); m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); buf={'X':None,'y':None,'logits':None,'t':None}; hist=[]; rng=np.random.RandomState(seed)
    for t,(X,y) in enumerate(tasks):
        train_derpp(m,t,X,y,buf,arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=BATCH,wd=1e-4,global_labels=False,class_offset=0,seed=seed+t)
        idx=rng.choice(len(X),min(M,len(X)),replace=False); xb=torch.as_tensor(_reshape_for('resnet18','cifar10',X[idx]),dtype=torch.float32,device=DEVICE); m.eval();
        with torch.no_grad(): lg=m.heads[t](m.features(xb)).cpu().numpy()
        rec={'X':X[idx],'y':y[idx],'logits':lg,'t':np.full(len(idx),t)}
        for k in buf: buf[k]=rec[k] if buf[k] is None else np.concatenate([buf[k],rec[k]])
        hist.append(eval_matrix(m,te,t))
    acc,bwt=calc_acc_bwt(hist); return {'ACC':acc,'BWT':bwt,'history':hist}


def paired_seed_diff_ci(rows, a, b, metric, n_boot=5000, seed=42):
    da={r['seed']:r[metric] for r in rows if r['method']==a}; db={r['seed']:r[metric] for r in rows if r['method']==b}; ks=sorted(set(da)&set(db))
    dif=np.asarray([da[k]-db[k] for k in ks],float)
    if len(dif)==0: return {'mean_diff':None,'ci95':None,'n_seeds':0}
    rng=np.random.RandomState(seed); vals=[float(np.mean(rng.choice(dif,size=len(dif),replace=True))) for _ in range(int(n_boot))]
    return {'mean_diff':float(np.mean(dif)),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'n_seeds':len(dif),'bootstrap':'paired-seed'}

def main():
    t0=time.time(); Xtr,ytr,Xte,yte=load_np('cifar10'); tasks=make_tasks(Xtr,ytr,SPLITS['cifar10']); te=make_tasks(Xte,yte,SPLITS['cifar10']); rows=[]
    for si in range(SEEDS):
        seed=BASE+si; lam_ewc=tune_lam(tasks,seed,False); lam_om=tune_lam(tasks,seed,True)
        ewc=ewc_train_sequence(tasks,te,seed,lam_ewc,False); omega=ewc_train_sequence(tasks,te,seed,lam_om,True); kg=train_replay_sequence(tasks,te,seed,'kappa'); um=train_replay_sequence(tasks,te,seed,'uniform',kg['mean_r']); u1=train_replay_sequence(tasks,te,seed,'uniform',1.0); der=derpp_sequence(tasks,te,seed)
        for name,rec in [('ewc_dr',ewc),('omega_ewc_dr',omega),('kappa_replay',kg),('uniform_mean_r',um),('uniform_r1',u1),('derpp',der)]: rows.append({'seed':seed,'method':name,**rec})
        print(seed,'kappa',kg['ACC'],kg['BWT'],'meanr',kg['mean_r'],'uniformmean',um['ACC'],um['BWT'],flush=True)
    def agg(name):
        z=[r for r in rows if r['method']==name]; return {'ACC':float(np.mean([r['ACC'] for r in z])),'BWT':float(np.mean([r['BWT'] for r in z])),'ACC_std':float(np.std([r['ACC'] for r in z]))}
    summary={n:agg(n) for n in sorted(set(r['method'] for r in rows))}; summary['omega_gate_pass']=bool(summary['omega_ewc_dr']['ACC']>=summary['ewc_dr']['ACC']+.01 and summary['omega_ewc_dr']['BWT']>=summary['ewc_dr']['BWT']); summary['kappa_gate_pass']=bool(summary['kappa_replay']['ACC']>=summary['uniform_mean_r']['ACC']+.01 and summary['kappa_replay']['BWT']>=summary['uniform_mean_r']['BWT']); summary['paired_uncertainty']={'omega_vs_ewc_ACC':paired_seed_diff_ci(rows,'omega_ewc_dr','ewc_dr','ACC',200 if SMOKE else 5000,BASE+81),'omega_vs_ewc_BWT':paired_seed_diff_ci(rows,'omega_ewc_dr','ewc_dr','BWT',200 if SMOKE else 5000,BASE+82),'kappa_vs_mean_replay_ACC':paired_seed_diff_ci(rows,'kappa_replay','uniform_mean_r','ACC',200 if SMOKE else 5000,BASE+83),'kappa_vs_mean_replay_BWT':paired_seed_diff_ci(rows,'kappa_replay','uniform_mean_r','BWT',200 if SMOKE else 5000,BASE+84)}
    out={'rows':rows,'summary':summary,'protocol':{'omega_aggregate':'mean across residents','kappa_r':'clip([-<gR,gB>]+/||gR||^2,0,1)','matched_mean_replay_control':True,'memory_per_task':M,'lambda_grid_shared':LAM_GRID},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}; p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
