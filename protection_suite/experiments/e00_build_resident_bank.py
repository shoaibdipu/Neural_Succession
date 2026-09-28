#!/usr/bin/env python
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import torch
from slt.engine import World
from slt.utils import seed_all, device_from_arg, write_json


def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description='Build one frozen A-only resident checkpoint per task for P2/P3/P4/P8.')
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100'])
    ap.add_argument('--data-root',default='./data'); ap.add_argument('--out-dir',default='checkpoints/resident_bank_seed0')
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='auto'); ap.add_argument('--arch',default='resnet18')
    ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--lr',type=float,default=0.05)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); seed_all(args.seed); dev=device_from_arg(args.device)
    w=World(args.dataset,args.data_root,args.arch,dev,128,2,args.seed,args.smoke)
    tasks=w.tasks[:3] if args.smoke else w.tasks
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    # All A-only residents in a seed start from the exact same model initialization.
    # This removes initialization as a source of between-task signal variation.
    base=w.model(); base_hash=state_hash(base)
    rows=[]
    for task in tasks:
        m=copy.deepcopy(base)
        w.train_task(m,task,args.epochs,args.lr)
        val=w.val_acc(m,task,True); test=w.acc(m,task,True)
        ckpt=out/f'{task.task_id:02d}_{task.name}.pt'
        payload={
            'dataset':args.dataset,'arch':w.arch,'seed':args.seed,'task_id':task.task_id,
            'task_name':task.name,'classes':list(task.classes),'epochs':args.epochs,'lr':args.lr,
            'base_initialization_sha256':base_hash,'resident_val_acc':val,'resident_test_acc':test,
            'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},
        }
        torch.save(payload,ckpt)
        rows.append({'task_id':task.task_id,'task_name':task.name,'classes':list(task.classes),
                     'checkpoint':ckpt.name,'resident_val_acc':val,'resident_test_acc':test})
        print(rows[-1],flush=True)
    manifest={
        'dataset':args.dataset,'arch':w.arch,'seed':args.seed,'epochs':args.epochs,'lr':args.lr,
        'shared_initialization':True,'base_initialization_sha256':base_hash,
        'purpose':'Exact shared resident checkpoints for P2/P3/P4/P8 and external signal computation',
        'residents':rows,
    }
    write_json(manifest,out/'manifest.json')
    print(f'wrote {out/"manifest.json"}')

if __name__=='__main__': main()
