#!/usr/bin/env python3
"""E6: Chesson decomposition at scale, 45 Split-CIFAR-100 task pairs.

Follows the attached plan: 10 tasks x 10 classes, ResNet-18, 20 epochs, 45
unordered task pairs, regress pairwise forgetting on (ND, FD). We additionally
report the sign-corrected compatibility model (S, FD) without replacing the
planned ND/FD analysis.
"""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from itertools import combinations
import matplotlib.pyplot as plt

SMOKE_TEST=os.environ.get('SLT_SMOKE','0')=='1'
DATASET='cifar100'; ARCH='resnet18'; EPOCHS=20; BASE_SEED=42
N_SEEDS=int(os.environ.get('SLT_E6_SEEDS','1'))
PAIR_LIMIT=None; DATA_FRAC=1.0
if SMOKE_TEST: EPOCHS=2; DATA_FRAC=.1; PAIR_LIMIT=4; N_SEEDS=1

def regress(X,y):
    X=np.asarray(X,float); y=np.asarray(y,float); Xs=(X-X.mean(0))/(X.std(0)+1e-8)
    A=np.c_[np.ones(len(Xs)),Xs]; c,*_=np.linalg.lstsq(A,y,rcond=None); pred=A@c
    ss=((y-pred)**2).sum(); tot=((y-y.mean())**2).sum(); return float(1-ss/(tot+1e-12)),c[1:],float(c[0]),pred

def loo_cv_r2(X,y):
    X=np.asarray(X,float); y=np.asarray(y,float); preds=np.zeros(len(y))
    for i in range(len(y)):
        keep=np.arange(len(y))!=i; Xm=X[keep]; ym=y[keep]
        mu=Xm.mean(0); sd=Xm.std(0)+1e-8; A=np.c_[np.ones(keep.sum()),(Xm-mu)/sd]
        c,*_=np.linalg.lstsq(A,ym,rcond=None); preds[i]=np.r_[1,(X[i]-mu)/sd]@c
    return float(1-((y-preds)**2).sum()/(((y-y.mean())**2).sum()+1e-12))

def run():
    Xtr,ytr,Xte,yte=load_np(DATASET); splits=SPLITS[DATASET]
    tr=make_tasks(Xtr,ytr,splits); te=make_tasks(Xte,yte,splits)
    if DATA_FRAC<1: tr=[subsample(X,y,max(512,int(len(X)*DATA_FRAC)),seed=0) for X,y in tr]
    pairs=list(combinations(range(len(splits)),2)); pairs=pairs[:PAIR_LIMIT] if PAIR_LIMIT else pairs
    rows=[]
    if os.path.exists('e6_results.json'):
        try: rows=json.load(open('e6_results.json'))
        except Exception: rows=[]
    done={(r['i'],r['j'],r['seed']) for r in rows}
    for i,j in pairs:
        for si in range(N_SEEDS):
            sd=BASE_SEED+si
            if (i,j,sd) in done: print(f'skip T{i}->T{j} seed={sd}'); continue
            set_seed(sd); XAtr,yAtr=tr[i]; XAte,yAte=te[i]; XBtr,yBtr=tr[j]
            m=make_model(ARCH,len(splits),10,scenario='task',dataset=DATASET).to(DEVICE)
            train_task(m,i,XAtr,yAtr,arch=ARCH,dataset=DATASET,epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4,drop_last=False)
            R_AA=accuracy(m,i,XAte,yAte,arch=ARCH,dataset=DATASET)
            kw=dict(arch=ARCH,dataset=DATASET); S=act_cov_overlap(m,XAtr,XBtr,**kw); ND,FD=chesson_nd_fd(m,XAtr,yAtr,XBtr,yBtr,n_cls=10,**kw)
            train_task(m,j,XBtr,yBtr,arch=ARCH,dataset=DATASET,epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4,drop_last=False)
            R_BA=accuracy(m,i,XAte,yAte,arch=ARCH,dataset=DATASET); fgt=pairwise_forgetting(R_AA,R_BA); bwt=R_BA-R_AA
            rows.append({'i':i,'j':j,'seed':sd,'S':S,'ND':ND,'FD':FD,'R_AA':R_AA,'R_BA':R_BA,'forgetting':fgt,'BWT':bwt}); json_dump(rows,'e6_results.json')
            print(f'T{i}->T{j} seed={sd} S={S:.3f} ND={ND:+.3f} FD={FD:+.3f} BWT={bwt:+.3f} fgt={fgt:+.3f}')
    return rows

def analyze(rows):
    # Average repeats per pair before regression, preserving the planned n=45 unit of analysis.
    pairs=sorted({(r['i'],r['j']) for r in rows}); a=[]
    for p in pairs:
        rr=[r for r in rows if (r['i'],r['j'])==p]
        vals={}
        for k in ['S','ND','FD','forgetting']:
            vals[k]=float(np.mean([x[k] for x in rr]))
        # Backward transfer for the ordered A->B pair is R_BA-R_AA = -forgetting.
        # The original E6 plan names BWT as the regression target, so BWT is primary.
        vals['BWT']=float(np.mean([x.get('BWT', -x['forgetting']) for x in rr])); a.append(vals)
    bwt=np.array([x['BWT'] for x in a]); S=np.array([x['S'] for x in a]); ND=np.array([x['ND'] for x in a]); FD=np.array([x['FD'] for x in a])
    r2_nd,cnd,_,pnd=regress(np.c_[ND,FD],bwt); r2_s,cs,_,ps=regress(np.c_[S,FD],bwt)
    r2_nd_cv=loo_cv_r2(np.c_[ND,FD],bwt) if len(bwt)>4 else float('nan'); r2_s_cv=loo_cv_r2(np.c_[S,FD],bwt) if len(bwt)>4 else float('nan')
    r2_S,_,_,_=regress(S[:,None],bwt); r2_FD,_,_,_=regress(FD[:,None],bwt)
    summary={'n_pairs':len(a),'target':'BWT = R_BA - R_AA (higher/closer to zero is better)',
             'ND_FD':{'R2':r2_nd,'LOO_R2':r2_nd_cv,'beta_ND':float(cnd[0]),'beta_FD':float(cnd[1])},
             'S_FD':{'R2':r2_s,'LOO_R2':r2_s_cv,'beta_S':float(cs[0]),'beta_FD':float(cs[1])},'univariate_R2':{'S':r2_S,'FD':r2_FD}}
    json_dump(summary,'e6_summary.json'); print(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(13,5)); ax[0].scatter(pnd,bwt,s=38); lo=min(pnd.min(),bwt.min()); hi=max(pnd.max(),bwt.max()); ax[0].plot([lo,hi],[lo,hi],'k--'); ax[0].set_xlabel('predicted BWT (ND,FD)'); ax[0].set_ylabel('actual BWT'); ax[0].set_title(f'Planned ND+FD model: R2={r2_nd:.2f}, LOO={r2_nd_cv:.2f}'); ax[0].grid(alpha=.25)
    ax[1].bar(['ND','FD'],cnd); ax[1].axhline(0,color='k',lw=.8); ax[1].set_ylabel('standardized coefficient'); ax[1].set_title('Planned Chesson coefficients'); ax[1].grid(alpha=.25,axis='y')
    plt.tight_layout(); plt.savefig('e6_chesson_cifar100.png',dpi=200,bbox_inches='tight'); plt.close(); return summary

if __name__=='__main__':
    print(f'E6 | device={DEVICE} smoke={SMOKE_TEST} seeds={N_SEEDS}')
    analyze(run())
