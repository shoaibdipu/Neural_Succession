#!/usr/bin/env python3
"""E8 controlled similarity sweep from the attached plan.

Modes: MNIST partial pixel permutation and CIFAR-10 rotation. Uses the original
20-epoch training protocol and measures rho_pre before Task-B training, forgetting
after Task-B training, and layer-wise representation drift.
"""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'))
from slt_common import *
from scipy.stats import pearsonr
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

SMOKE_TEST=os.environ.get('SLT_SMOKE','0')=='1'; MODE=os.environ.get('SLT_E8_MODE','mnist_perm'); N_SEEDS=3; BASE_SEED=42
if MODE=='mnist_perm': DATASET,ARCH,EPOCHS='mnist','mlp',20; KNOBS=[0,.05,.1,.2,.35,.5,.75,1.0]; CLASS_PAIRS=[(0,1),(3,5),(4,9),(2,7)]
elif MODE=='cifar_rotate': DATASET,ARCH,EPOCHS='cifar10','resnet18',20; KNOBS=[0,15,30,60,90,120,150,180]; CLASS_PAIRS=[(0,1),(3,5)]
else: raise ValueError(MODE)
if SMOKE_TEST: EPOCHS=2; N_SEEDS=1; KNOBS=KNOBS[::3]; CLASS_PAIRS=CLASS_PAIRS[:1]

def perm_indices(d,f,seed):
    rng=np.random.RandomState(seed); p=np.arange(d); k=int(round(f*d))
    if k>=2:
        idx=rng.choice(d,k,replace=False); p[idx]=idx[rng.permutation(k)]
    return p

def apply_knob(X,k,seed):
    if MODE=='mnist_perm': return X[:,perm_indices(X.shape[1],k,seed)].copy()
    t=torch.tensor(X,dtype=torch.float32); fill=[float(-m/s) for m,s in zip(CIFAR10_MEAN,CIFAR10_STD)]
    return TF.rotate(t,float(k),fill=fill).numpy()

@torch.no_grad()
def layer_reps(model,X,max_n=1500):
    model.eval().to(DEVICE); Xr=_reshape_for(ARCH,DATASET,X); Xr,_=subsample(Xr,np.zeros(len(Xr)),max_n,seed=0); xb=torch.tensor(Xr).to(DEVICE); return {k:v.detach().cpu().float() for k,v in model.features_at(xb).items()}

def drift(a,b): return {k:1-_cov_cosine(_cov(a[k],True),_cov(b[k],True)) for k in a}

def run():
    Xtr,ytr,Xte,yte=load_np(DATASET); rows=[]
    for si in range(N_SEEDS):
        set_seed(BASE_SEED+si)
        for c0,c1 in CLASS_PAIRS:
            XAtr,yAtr=binary_subset(Xtr,ytr,c0,c1); XAte,yAte=binary_subset(Xte,yte,c0,c1)
            for k in KNOBS:
                m=make_model(ARCH,2,2,scenario='task',dataset=DATASET).to(DEVICE)
                train_task(m,0,XAtr,yAtr,arch=ARCH,dataset=DATASET,epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4,drop_last=False); R_AA=accuracy(m,0,XAte,yAte,arch=ARCH,dataset=DATASET)
                XBtr=apply_knob(XAtr,k,1234); XBte=apply_knob(XAte,k,1234); rp=rho_pre(m,0,XBte,yAte,arch=ARCH,dataset=DATASET,n_samples=None); ov=act_cov_overlap(m,XAte,XBte,arch=ARCH,dataset=DATASET)
                before=layer_reps(m,XAte); train_task(m,1,XBtr,yAtr,arch=ARCH,dataset=DATASET,epochs=EPOCHS,lr=1e-3,batch=128,wd=1e-4,drop_last=False); after=layer_reps(m,XAte); dr=drift(before,after)
                R_BA=accuracy(m,0,XAte,yAte,arch=ARCH,dataset=DATASET); row={'seed':BASE_SEED+si,'pair':f'{c0}/{c1}','knob':k,'rho_pre':rp,'act_cov':ov,'R_AA':R_AA,'R_BA':R_BA,'forgetting':pairwise_forgetting(R_AA,R_BA),'drift':dr}; rows.append(row); json_dump(rows,'e8_results.json'); print(f's{si} {c0}/{c1} knob={k} rho={rp:+.3f} fgt={row["forgetting"]:+.3f}')
    return rows

def cv_r2(rows,degree):
    groups=sorted(set((r['seed'],r['pair']) for r in rows)); yall=[]; pall=[]
    for g in groups:
        tr=[r for r in rows if (r['seed'],r['pair'])!=g]; te=[r for r in rows if (r['seed'],r['pair'])==g]
        x=np.array([r['rho_pre'] for r in tr]); y=np.array([r['forgetting'] for r in tr]); coef=np.polyfit(x,y,degree)
        xt=np.array([r['rho_pre'] for r in te]); yt=np.array([r['forgetting'] for r in te]); yall.extend(yt); pall.extend(np.polyval(coef,xt))
    y=np.array(yall); p=np.array(pall); return float(1-((y-p)**2).sum()/(((y-y.mean())**2).sum()+1e-12))

def analyze(rows):
    rho=np.array([r['rho_pre'] for r in rows]); f=np.array([r['forgetting'] for r in rows]); knob=np.array([r['knob'] for r in rows]); r,p=pearsonr(rho,f); c2=np.polyfit(rho,f,2); pred=np.polyval(c2,rho); r2q=float(1-((f-pred)**2).sum()/(((f-f.mean())**2).sum()+1e-12)); r2l=float(r*r); cv1=cv_r2(rows,1) if len(set((x['seed'],x['pair']) for x in rows))>2 else float('nan'); cv2=cv_r2(rows,2) if len(set((x['seed'],x['pair']) for x in rows))>3 else float('nan'); vertex=float(-c2[1]/(2*c2[0])) if abs(c2[0])>1e-12 else float('nan'); interior=bool(np.isfinite(vertex) and rho.min()<vertex<rho.max())
    summary={'mode':MODE,'n':len(rows),'linear':{'pearson_r':float(r),'p':float(p),'R2':r2l,'group_CV_R2':cv1},'quadratic':{'R2':r2q,'group_CV_R2':cv2,'coef':c2.tolist(),'vertex_rho':vertex,'vertex_in_range':interior}}
    json_dump(summary,'e8_summary.json'); print(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(14,5)); sc=ax[0].scatter(rho,f,c=knob,s=40); xs=np.linspace(rho.min(),rho.max(),100); ax[0].plot(xs,np.polyval(c2,xs),'k-',label='quadratic'); c1=np.polyfit(rho,f,1); ax[0].plot(xs,np.polyval(c1,xs),'k--',label='linear'); ax[0].set_xlabel('measured rho_pre'); ax[0].set_ylabel('forgetting'); ax[0].legend(); ax[0].grid(alpha=.25); plt.colorbar(sc,ax=ax[0],label='similarity knob')
    for pnm in sorted(set(x['pair'] for x in rows)):
        kk=sorted(set(x['knob'] for x in rows if x['pair']==pnm)); mf=[np.mean([x['forgetting'] for x in rows if x['pair']==pnm and x['knob']==k]) for k in kk]; ax[1].plot(kk,mf,marker='o',label=pnm)
    ax[1].set_xlabel('knob'); ax[1].set_ylabel('forgetting'); ax[1].legend(); ax[1].grid(alpha=.25); plt.tight_layout(); plt.savefig('e8_similarity_sweep.png',dpi=200,bbox_inches='tight'); plt.close()
    layers=list(rows[0]['drift'])
    if len(layers)>1:
        plt.figure(figsize=(8,5)); order=np.argsort(rho)
        for L in layers: plt.plot(rho[order],np.array([x['drift'][L] for x in rows])[order],'.-',alpha=.7,label=L)
        plt.xlabel('rho_pre'); plt.ylabel('representation drift on Task-A inputs'); plt.legend(); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig('e8_layerwise.png',dpi=200,bbox_inches='tight'); plt.close()
    return summary

if __name__=='__main__':
    print(f'E8 | mode={MODE} device={DEVICE} smoke={SMOKE_TEST}')
    analyze(run())
