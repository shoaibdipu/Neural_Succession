#!/usr/bin/env python
from __future__ import annotations
import argparse, copy, itertools
from pathlib import Path
import pandas as pd
from slt.engine import World
from slt.signals import slt_compatibility
from slt.train import ReservoirBuffer,populate_buffer,estimate_fisher,ewc_replay_train_task,derpp_train_task
from slt.controllers import FixedController
from slt.residents import load_resident_from_bank
from slt.stats import delta_r2_baseline_plus,corr,partial_corr
from slt.utils import seed_all,device_from_arg,write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100']); ap.add_argument('--data-root',default='./data')
    ap.add_argument('--out',default='outputs/e03_protection_benefit.csv'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='auto'); ap.add_argument('--arch',default='resnet18')
    ap.add_argument('--resident-epochs',type=int,default=20); ap.add_argument('--incoming-epochs',type=int,default=15); ap.add_argument('--buffer',type=int,default=500)
    ap.add_argument('--ewc-lambda',type=float,default=10.0); ap.add_argument('--probe-n',type=int,default=256); ap.add_argument('--shuffles',type=int,default=20)
    ap.add_argument('--signals-csv',default=None,help='P2 CSV with exact same-transition signals and resident_checkpoint')
    ap.add_argument('--resident-bank-manifest',default=None,help='Exact e00 resident bank used by P2; required for final P3')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); seed_all(args.seed); dev=device_from_arg(args.device)
    w=World(args.dataset,args.data_root,args.arch,dev,128,2,args.seed,args.smoke); tasks=w.tasks[:3] if args.smoke else w.tasks
    pairs=list(itertools.permutations(tasks,2));
    if args.smoke: pairs=pairs[:1]
    if (not args.smoke) and not args.resident_bank_manifest:
        raise SystemExit('Final P3 requires --resident-bank-manifest from e00 so protection outcomes use the same resident checkpoints as P2.')
    p2 = pd.read_csv(args.signals_csv) if args.signals_csv else None
    if (not args.smoke) and p2 is None:
        raise SystemExit('Final P3 requires --signals-csv from P2; protection-benefit prediction must use the exact same signal rows.')
    rows=[]
    for A,B in pairs:
        if args.resident_bank_manifest:
            resident,bank_rec,ckpt_path=load_resident_from_bank(w,A,args.resident_bank_manifest)
            checkpoint=str(ckpt_path.resolve())
        else:
            resident=w.model(); w.train_task(resident,A,args.resident_epochs,0.05); checkpoint='SMOKE_LOCAL_RESIDENT'
        A0=w.acc(resident,A,True)
        keypair=f'{A.name}->{B.name}'
        p2row=None
        if p2 is not None:
            hit=p2[(p2.seed==args.seed)&(p2.pair==keypair)&(p2.resident=='A_only')]
            if len(hit)!=1: raise ValueError(f'expected one P2 row for seed={args.seed}, pair={keypair}; got {len(hit)}')
            p2row=hit.iloc[0]
            if 'resident_checkpoint' in p2row.index and str(p2row.resident_checkpoint) not in (checkpoint,'SMOKE_LOCAL_RESIDENT'):
                raise ValueError(f'P2/P3 resident checkpoint mismatch for {keypair}: {p2row.resident_checkpoint} vs {checkpoint}')
            sig={'rho':float(p2row.rho),'omega':float(p2row.omega)}
        else:
            sig=slt_compatibility(resident,w.probe_loader(B),dev,args.shuffles,args.probe_n,args.seed+100*A.task_id+B.task_id)
        # Unprotected FT reference.
        ft=copy.deepcopy(resident); w.train_task(ft,B,args.incoming_epochs,0.03); FA=A0-w.acc(ft,A,True); Bft=w.acc(ft,B,True)
        # EWC + data replay reference. Replace with archival EWC-DR for final paper if available.
        fisher,theta=estimate_fisher(resident,w.train_loader(A,True),dev,A.classes,max_batches=5 if args.smoke else 20)
        rb=ReservoirBuffer(args.buffer); populate_buffer(resident,w.full_train_loader(A,True),dev,rb,args.buffer,task_id=A.task_id)
        ewc=copy.deepcopy(resident); ewc_replay_train_task(ewc,w.train_loader(B,True),dev,fisher,theta,rb,
            epochs=args.incoming_epochs,lr=0.03,lam=args.ewc_lambda,replay_weight=0.3,replay_batch_size=64,task_classes=B.classes)
        Fewc=A0-w.acc(ewc,A,True); Bewc=w.acc(ewc,B,True)
        # DER++ reference, same resident buffer protocol.
        rb2=ReservoirBuffer(args.buffer); populate_buffer(resident,w.full_train_loader(A,True),dev,rb2,args.buffer,task_id=A.task_id)
        der=copy.deepcopy(resident); derpp_train_task(der,w.train_loader(B,True),dev,rb2,args.incoming_epochs,0.03,
            replay_batch_size=64,alpha=0.5,beta=0.5,task_classes=B.classes,controller=FixedController(0.5),current_task_id=B.task_id)
        Fder=A0-w.acc(der,A,True); Bder=w.acc(der,B,True)
        for method,Fm,Bm in [('ewc_dr',Fewc,Bewc),('derpp',Fder,Bder)]:
            rows.append({'seed':args.seed,'pair':keypair,'resident':'A_only','A':A.name,'B':B.name,'method':method,
                         'resident_checkpoint':checkpoint,
                         'rho':sig['rho'],'omega':sig['omega'],'baseline_forgetting':FA,'protected_forgetting':Fm,
                         'protection_gain':FA-Fm,'plasticity_cost':Bft-Bm,'B_ft':Bft,'B_method':Bm,'A_before':A0})
        print(rows[-2:],flush=True)
    df=pd.DataFrame(rows)
    if p2 is not None:
        keys=['seed','pair','resident']
        if p2.duplicated(keys).any(): raise ValueError('P2 signal CSV contains duplicate seed,pair,resident rows')
        keep=keys+[c for c in p2.columns if c not in keys and c not in df.columns]
        df=df.merge(p2[keep],on=keys,how='left',validate='many_to_one')
        if len(df) != len(rows): raise RuntimeError('P2 merge changed P3 row count')
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.out,index=False)
    preferred=['rho','raw_probe_error','grad_cosine','activation_covariance_similarity','one_step_loss_change','spot','gnr','ntk_overlap','cka','logme','leep']
    sigcols=[c for c in preferred if c in df.columns]
    summary={}
    for m,d in df.groupby('method'):
        summary[m]={
            'gain_corr_rho':corr(d.rho,d.protection_gain),
            'partial_rho_given_forgetting':partial_corr(d,'rho','protection_gain',['baseline_forgetting']),
            'partial_rho_given_forgetting_and_plasticity':partial_corr(d,'rho','protection_gain',['baseline_forgetting','plasticity_cost']),
            'signals':{}
        }
        for c in sigcols:
            summary[m]['signals'][c]=delta_r2_baseline_plus(d,'protection_gain',['baseline_forgetting'],c,group_col='B')
        # Stronger incremental test: does rho add information after baseline
        # forgetting AND an existing pre-task signal are already present?
        for base in ['spot','gnr','ntk_overlap','grad_cosine','cka','logme','leep','raw_probe_error',
                     'activation_covariance_similarity','one_step_loss_change']:
            if base in d.columns and 'rho' in d.columns:
                summary[m].setdefault('incremental_rho_over_existing_signal',{})[base] = \
                    delta_r2_baseline_plus(d,'protection_gain',['baseline_forgetting',base],'rho',group_col='B')
    write_json(summary,Path(args.out).with_suffix('.summary.json')); print(summary)
if __name__=='__main__': main()
