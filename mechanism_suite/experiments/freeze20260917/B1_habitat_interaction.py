#!/usr/bin/env python3
from __future__ import annotations
import copy,os,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from slt_confirmatory import stratified_probe,stratified_train_val,freeze_batchnorm_stats,run_metadata
from slt_freeze import *

EXP='B1_habitat_interaction'; SMOKE=env_bool('SLT_SMOKE'); BASE=int(os.environ.get('SLT_BASE_SEED','42'))
SEEDS=int(os.environ.get('SLT_B1_SEEDS','1' if SMOKE else '3')); PAIRS=fixed_pairs(5,2 if SMOKE else int(os.environ.get('SLT_B1_PAIRS','20')))
EPOCHS=int(os.environ.get('SLT_B1_EPOCHS','2' if SMOKE else '20'))
BATCH_SGD=int(os.environ.get('SLT_B1_SGD_BATCH','32')); BATCH_ADAM=int(os.environ.get('SLT_B1_ADAM_BATCH','128'))
REGIMES=[('sgd',float(os.environ.get('SLT_B1_SGD_LR','0.03')),BATCH_SGD),('adam',1e-3,BATCH_ADAM)]


def eval_loss(m,t,X,y):
    m.eval(); xr=_reshape_for('resnet18','cifar10',X); tot=0.; n=0
    with torch.no_grad():
        for i in range(0,len(xr),256):
            xb=torch.as_tensor(xr[i:i+256],dtype=torch.float32,device=DEVICE); yy=torch.as_tensor(y[i:i+256],dtype=torch.long,device=DEVICE); z=m.heads[t](m.features(xb)); tot+=float(F.cross_entropy(z,yy,reduction='sum').cpu()); n+=len(yy)
    return tot/max(n,1)


def flat_backbone_grad_from_param_grads(m):
    out=[]
    for n,p in backbone_named_parameters(m): out.append((torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1))
    return torch.cat(out)


def train_pair(base,a,b,trA,trB,regime,lr,batch,seed,lstar_B):
    XA,yA=trA; XB,yB=trB
    # H_hab: fit/readout and evaluation are disjoint B-train subsets.
    (Bfit,yfit),(Bev,yev),_,_=stratified_train_val(XB,yB,.20,seed+10); ploss,pacc,clf=linear_probe_loss_and_acc(base,Bfit,yfit,Bev,yev,arch='resnet18',dataset='cifar10'); Hhab=max(0.,ploss-lstar_B)
    m=copy.deepcopy(base); copy_probe_to_head(m,b,clf); _set_trainable(m,b)
    for p in m.heads[a].parameters(): p.requires_grad=False
    params=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.SGD(params,lr=lr,momentum=0,weight_decay=0) if regime=='sgd' else torch.optim.Adam(params,lr=lr,weight_decay=1e-4)
    XAp,yAp,_=stratified_probe(XA,yA,64 if SMOKE else 128,seed+20); xr=_reshape_for('resnet18','cifar10',XB); rng=np.random.RandomState(seed+30)
    initLA=eval_loss(m,a,XAp,yAp); sum_dotA=0.; sum_wB=0.; sum_head=0.; npos=nneg=nzero=0; nfac=ninh=nneutral=0; kappas=[]
    nsteps=EPOCHS*int(np.ceil(len(xr)/batch)); trace={'dot_A_update':[],'wB_backbone':[],'wB_head':[],'kappa_opt':[],'loss_B_batch':[]}
    for st in range(nsteps):
        ids=rng.choice(len(xr),min(batch,len(xr)),replace=len(xr)<batch); xb=torch.as_tensor(xr[ids],dtype=torch.float32,device=DEVICE); yy=torch.as_tensor(yB[ids],dtype=torch.long,device=DEVICE)
        gA,_=backbone_gradient(m,a,XAp,yAp,arch='resnet18',dataset='cifar10',max_n=len(XAp),seed=0,freeze_bn=True)
        before_bb=snapshot_named(backbone_named_parameters(m)); before_head=[p.detach().clone() for p in m.heads[b].parameters()]
        m.train(); freeze_batchnorm_stats(m); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(m.heads[b](m.features(xb)),yy); loss.backward(); gB=flat_backbone_grad_from_param_grads(m); gBh=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1) for p in m.heads[b].parameters()]); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0); opt.step()
        dBB=optimizer_update_dot(m,before_bb); dH=torch.cat([(p.detach()-q.to(p.device)).reshape(-1) for p,q in zip(m.heads[b].parameters(),before_head)]); dotA=float(torch.dot(gA,dBB).cpu()); wB=float(-torch.dot(gB,dBB).cpu()); wh=float(-torch.dot(gBh,dH).cpu()); sum_dotA+=dotA; sum_wB+=wB; sum_head+=wh
        if wB>0: k=dotA/(wB+1e-12); kappas.append(k); npos+=1
        elif wB<0: k=np.nan; nneg+=1
        else: k=np.nan; nzero+=1
        if dotA>1e-12: ninh+=1
        elif dotA<-1e-12: nfac+=1
        else: nneutral+=1
        trace['dot_A_update'].append(dotA); trace['wB_backbone'].append(wB); trace['wB_head'].append(wh); trace['kappa_opt'].append(k); trace['loss_B_batch'].append(float(loss.detach().cpu()))
    finalLA=eval_loss(m,a,XAp,yAp); delta=finalLA-initLA; residual=delta-sum_dotA
    trace_dir=exp_outdir(ROOT,EXP)/'traces'; trace_dir.mkdir(parents=True,exist_ok=True); trace_name=f'seed{seed}_A{a}_B{b}_{regime}.npz'; np.savez_compressed(trace_dir/trace_name,**{k:np.asarray(v,float) for k,v in trace.items()})
    return {'regime':regime,'lr':lr,'batch':batch,'probe_loss_B':ploss,'probe_acc_B':pacc,'lstar_B':lstar_B,'H_hab':Hhab,'delta_LA':delta,'sum_gA_dot_update':sum_dotA,'R':residual,'relative_R':abs(residual)/(abs(delta)+1e-8),'W_B_backbone_signed':sum_wB,'W_B_head_signed':sum_head,'kappa_mean_positive_progress':float(np.mean(kappas)) if kappas else None,'frac_wB_positive':npos/max(nsteps,1),'frac_wB_nonpositive':(nneg+nzero)/max(nsteps,1),'frac_facilitative_steps':nfac/max(nsteps,1),'frac_inhibitory_steps':ninh/max(nsteps,1),'frac_neutral_steps':nneutral/max(nsteps,1),'steps':nsteps,'trace_npz':str(Path('traces')/trace_name)}


def main():
    t0=time.time(); Xtr,ytr,_,_=load_np('cifar10'); tr=make_tasks(Xtr,ytr,SPLITS['cifar10']); rows=[]
    for si in range(SEEDS):
        seed=BASE+si; lstar={}
        for b in range(5):
            (Bt,By),(Bv,Bvy),_,_=stratified_train_val(*tr[b],.20,seed+500+b); mb=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); train_task(mb,b,Bt,By,arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=BATCH_ADAM,wd=1e-4); lstar[b]=eval_loss(mb,b,Bv,Bvy)
        residents={}
        for a in sorted(set(a for a,_ in PAIRS)):
            m=make_model('resnet18',5,2,scenario='task',dataset='cifar10').to(DEVICE); train_task(m,a,*tr[a],arch='resnet18',dataset='cifar10',epochs=EPOCHS,lr=1e-3,batch=BATCH_ADAM,wd=1e-4); residents[a]=m
        for a,b in PAIRS:
            for regime,lr,batch in REGIMES:
                r=train_pair(residents[a],a,b,tr[a],tr[b],regime,lr,batch,seed+1000+a*100+b,lstar[b]); r.update({'seed':seed,'task_a':a,'task_b':b}); rows.append(r); print(seed,a,b,regime,'delta',round(r['delta_LA'],4),'Rrel',round(r['relative_R'],3),'fac',round(r['frac_facilitative_steps'],3),'inh',round(r['frac_inhibitory_steps'],3),flush=True)
    summary={}
    for regime,_,_ in REGIMES:
        z=[r for r in rows if r['regime']==regime]; rr,_=pearson_safe([r['sum_gA_dot_update'] for r in z],[r['delta_LA'] for r in z]); X=np.asarray([r['sum_gA_dot_update'] for r in z]); Y=np.asarray([r['delta_LA'] for r in z]); slope=float(np.polyfit(X,Y,1)[0]) if len(X)>=2 and np.std(X)>1e-12 else None
        summary[regime]={'r_pred_vs_delta_LA':rr,'slope':slope,'median_relative_R':float(np.median([r['relative_R'] for r in z])),'mean_frac_wB_nonpositive':float(np.mean([r['frac_wB_nonpositive'] for r in z])),'mean_frac_facilitative_steps':float(np.mean([r['frac_facilitative_steps'] for r in z])),'mean_frac_inhibitory_steps':float(np.mean([r['frac_inhibitory_steps'] for r in z])),'primary_identity_pass':bool(rr is not None and rr>=.9 and slope is not None and .8<=slope<=1.2)}
    out={'rows':rows,'summary':summary,'protocol':{'identity_sum':'all steps use <g_A^bb, delta theta^bb>; kappa only defined where w_B^bb>0','per_step_arrays':'compressed NPZ sidecars for every transition/regime','H_hab':'probe fit/eval disjoint; threshold is B-only validation loss','B_head':'initialized from resident-backbone probe and remains trainable','sgd_batch':BATCH_SGD,'adam_batch':BATCH_ADAM},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}; p=dump_result(ROOT,EXP,out); print('saved',p)
if __name__=='__main__': main()
