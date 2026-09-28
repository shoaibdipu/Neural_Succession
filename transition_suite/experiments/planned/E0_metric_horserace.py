#!/usr/bin/env python3
"""E0: metric horse-race, following the attached plan.

Same Split-CIFAR-10 pairwise training protocol as E1: ResNet-18, Task-IL,
20 fixed epochs, Adam 1e-3, weight decay 1e-4, batch 128, 20 ordered pairs,
3 repeats. Metrics are computed pre-hoc on the Task-A model before Task-B training.
"""
from pathlib import Path
import os, sys, json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'core'))
from slt_common import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

SMOKE_TEST = os.environ.get('SLT_SMOKE','0') == '1'
DATASET='cifar10'; ARCH='resnet18'; EPOCHS=20; N_SEEDS=3; BASE_SEED=42
TASKS=SPLITS['cifar10']
USE_JACOBIAN = os.environ.get('SLT_E0_JACOBIAN','1') == '1'
USE_GRADIENT = os.environ.get('SLT_E0_GRADIENT','1') == '1'
if SMOKE_TEST:
    EPOCHS=2; N_SEEDS=1; TASKS=TASKS[:3]

METRICS=['rho_pre','act_cov','repr','jacobian','gradient']
PRETTY={'rho_pre':'rho_pre (behavioral)','act_cov':'Omega act-cov',
        'repr':'repr. sim. (centered cov)','jacobian':'Omega Jacobian',
        'gradient':'task-gradient alignment'}

def run():
    Xtr,ytr,Xte,yte=load_np(DATASET)
    data={t:{'tr':binary_subset(Xtr,ytr,*p),'te':binary_subset(Xte,yte,*p)}
          for t,p in enumerate(TASKS)}
    pairs=[(a,b) for a in range(len(TASKS)) for b in range(len(TASKS)) if a!=b]
    rows=[]
    for s in range(N_SEEDS):
        # Match original E1 seeding: one seed per repetition, then RNG advances by pair.
        set_seed(BASE_SEED+s)
        for tA,tB in pairs:
            XAtr,yAtr=data[tA]['tr']; XAte,yAte=data[tA]['te']
            XBtr,yBtr=data[tB]['tr']; XBte,yBte=data[tB]['te']
            mA=make_model(ARCH,len(TASKS),2,scenario='task',dataset=DATASET).to(DEVICE)
            train_task(mA,tA,XAtr,yAtr,arch=ARCH,dataset=DATASET,epochs=EPOCHS,
                       lr=1e-3,batch=128,wd=1e-4,drop_last=False)
            R_AA=accuracy(mA,tA,XAte,yAte,arch=ARCH,dataset=DATASET)
            kw=dict(arch=ARCH,dataset=DATASET)
            vals={
                'rho_pre':rho_pre(mA,tA,XBte,yBte,n_samples=None,**kw),
                'act_cov':act_cov_overlap(mA,XAte,XBte,**kw),
                'repr':repr_overlap(mA,XAte,XBte,**kw),
                'jacobian':jacobian_overlap(mA,XAte,XBte,**kw) if USE_JACOBIAN else float('nan'),
                'gradient':gradient_overlap(mA,XAtr,yAtr,XBtr,yBtr,n_cls=2,taskA_id=tA,**kw)
                           if USE_GRADIENT else float('nan')
            }
            train_task(mA,tB,XBtr,yBtr,arch=ARCH,dataset=DATASET,epochs=EPOCHS,
                       lr=1e-3,batch=128,wd=1e-4,drop_last=False)
            R_BA=accuracy(mA,tA,XAte,yAte,arch=ARCH,dataset=DATASET)
            row={'seed':BASE_SEED+s,'pair':f'T{tA}->T{tB}','tA':tA,'tB':tB,
                 'R_AA':R_AA,'R_BA':R_BA,'forgetting':pairwise_forgetting(R_AA,R_BA),**vals}
            rows.append(row)
            json_dump(rows,'e0_results.json')
            print(f"[s{s}] {row['pair']} fgt={row['forgetting']:+.3f} " +
                  ' '.join(f"{k}={vals[k]:+.3f}" for k in METRICS))
    return rows

def analyze(rows):
    pairs=sorted(set(r['pair'] for r in rows))
    agg={}
    for p in pairs:
        rr=[r for r in rows if r['pair']==p]
        agg[p]={'forgetting':float(np.mean([x['forgetting'] for x in rr]))}
        for k in METRICS:
            agg[p][k]=float(np.nanmean([x[k] for x in rr]))
    fgt=np.array([agg[p]['forgetting'] for p in pairs])
    stats={}
    print('\nMETRIC HORSE-RACE')
    for k in METRICS:
        x=np.array([agg[p][k] for p in pairs])
        ok=np.isfinite(x)&np.isfinite(fgt)
        if ok.sum()<3: continue
        r,pv=pearsonr(x[ok],fgt[ok]); rs,ps=spearmanr(x[ok],fgt[ok])
        stats[k]={'pearson_r':float(r),'pearson_p':float(pv),'spearman_rho':float(rs),'spearman_p':float(ps)}
        print(f"  {k:<10} r={r:+.3f} p={pv:.4g} Spearman={rs:+.3f}")
    json_dump({'pair_means':agg,'stats':stats},'e0_summary.json')

    ks=list(stats)
    n=len(ks)
    fig,axes=plt.subplots(2,3,figsize=(16,9)); axes=axes.ravel()
    for i,k in enumerate(ks):
        x=np.array([agg[p][k] for p in pairs]); ax=axes[i]
        ax.scatter(x,fgt,s=36,alpha=.85)
        if len(x)>=2 and np.ptp(x)>0:
            m,b=np.polyfit(x,fgt,1); xs=np.linspace(x.min(),x.max(),100); ax.plot(xs,m*xs+b,'k--',lw=1.3)
        ax.set_title(f"{PRETTY[k]}\nr={stats[k]['pearson_r']:+.3f}")
        ax.set_xlabel(k); ax.set_ylabel('forgetting R_AA-R_BA'); ax.grid(alpha=.25)
    if n < len(axes):
        ax=axes[n]; ax.bar(range(n),[abs(stats[k]['pearson_r']) for k in ks])
        ax.set_xticks(range(n)); ax.set_xticklabels(ks,rotation=35,ha='right'); ax.set_ylabel('|Pearson r|')
        ax.set_title('Predictive magnitude'); ax.grid(alpha=.25,axis='y')
        for j in range(n+1,len(axes)): axes[j].axis('off')
    plt.suptitle('E0 — Pre-hoc metric horse-race on Split-CIFAR-10',fontweight='bold')
    plt.tight_layout(); plt.savefig('e0_metric_horserace.png',dpi=200,bbox_inches='tight'); plt.close()
    return stats

if __name__=='__main__':
    print(f'E0 | device={DEVICE} smoke={SMOKE_TEST} jacobian={USE_JACOBIAN} gradient={USE_GRADIENT}')
    analyze(run())
