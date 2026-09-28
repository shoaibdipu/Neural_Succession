from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from scipy.stats import pearsonr, spearmanr, t, ttest_rel
import statsmodels.formula.api as smf


def corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return {"r":np.nan,"p":np.nan,"rho":np.nan,"rho_p":np.nan,"n":int(m.sum())}
    r,p=pearsonr(x[m],y[m]); rs,ps=spearmanr(x[m],y[m])
    return {"r":float(r),"p":float(p),"rho":float(rs),"rho_p":float(ps),"n":int(m.sum())}




def partial_corr(df:pd.DataFrame,x:str,y:str,controls:list[str]):
    """Pearson/Spearman association after linearly residualizing x and y on controls."""
    controls=list(controls or [])
    cols=[x,y]+controls
    d=df[cols].dropna().copy()
    if len(d)<3:
        return {"r":np.nan,"p":np.nan,"rho":np.nan,"rho_p":np.nan,"n":int(len(d)),"controls":controls}
    if controls:
        C=d[controls].to_numpy(float)
        rx=d[x].to_numpy(float)-LinearRegression().fit(C,d[x].to_numpy(float)).predict(C)
        ry=d[y].to_numpy(float)-LinearRegression().fit(C,d[y].to_numpy(float)).predict(C)
    else:
        rx=d[x].to_numpy(float); ry=d[y].to_numpy(float)
    out=corr(rx,ry); out['controls']=controls
    return out

def loo_regression(x,y,groups=None):
    X=np.asarray(x,float).reshape(-1,1); y=np.asarray(y,float)
    if len(y)<2:
        return {'pred':np.full(len(y),np.nan),'mae':np.nan,'r2':np.nan,'pred_r':np.nan,'pred_p':np.nan,'pred_rho':np.nan,'pred_rho_p':np.nan,'pred_n':len(y)}
    preds=np.full(len(y),np.nan)
    splitter=LeaveOneOut().split(X) if groups is None else LeaveOneGroupOut().split(X,y,np.asarray(groups))
    for tr,te in splitter:
        model=LinearRegression().fit(X[tr],y[tr]); preds[te]=model.predict(X[te])
    m=np.isfinite(preds)&np.isfinite(y)
    return {"pred":preds,"mae":float(mean_absolute_error(y[m],preds[m])),
            "r2":float(r2_score(y[m],preds[m])), **{f"pred_{k}":v for k,v in corr(preds[m],y[m]).items()}}


def delta_r2_baseline_plus(df:pd.DataFrame,target:str,baseline_cols:list[str],signal_col:str,group_col:str|None=None):
    cols=baseline_cols+[signal_col,target]+([group_col] if group_col else [])
    d=df[cols].dropna().copy(); y=d[target].to_numpy(float)
    if len(d)<3 or (group_col and d[group_col].nunique()<2):
        return {'r2_baseline':np.nan,'r2_plus_signal':np.nan,'delta_r2':np.nan,'mae_baseline':np.nan,'mae_plus_signal':np.nan}
    if group_col:
        splits=LeaveOneGroupOut().split(d, y, d[group_col])
    else:
        splits=LeaveOneOut().split(d)
    p0=np.full(len(d),np.nan); p1=np.full(len(d),np.nan)
    for tr,te in splits:
        X0=d.iloc[tr][baseline_cols].to_numpy(float); X1=d.iloc[tr][baseline_cols+[signal_col]].to_numpy(float)
        m0=Ridge(alpha=1e-6).fit(X0,y[tr]); m1=Ridge(alpha=1e-6).fit(X1,y[tr])
        p0[te]=m0.predict(d.iloc[te][baseline_cols].to_numpy(float)); p1[te]=m1.predict(d.iloc[te][baseline_cols+[signal_col]].to_numpy(float))
    r20=r2_score(y,p0); r21=r2_score(y,p1)
    return {"r2_baseline":float(r20),"r2_plus_signal":float(r21),"delta_r2":float(r21-r20),
            "mae_baseline":float(mean_absolute_error(y,p0)),"mae_plus_signal":float(mean_absolute_error(y,p1))}


def pair_fixed_effect(df:pd.DataFrame, outcome='forgetting', signal='rho', pair='pair', seed='seed', covariates=None):
    """Within-semantic-pair P1 regression with optional prespecified covariates.

    Pair fixed effects remove semantic A->B identity. If multiple seeds are
    present, seed fixed effects remove seed-wide shifts. Concrete resident
    checkpoint labels are intentionally not included because resident history is
    the within-pair manipulation that induces changes in rho.

    ``covariates`` is used only for prespecified robustness checks, notably
    resident A validation accuracy. It does not replace the frozen primary model.
    """
    covariates=list(covariates or [])
    cols=[outcome,signal,pair]+covariates+([seed] if seed in df.columns else [])
    d=df[cols].dropna().copy()
    rhs=' + '.join([signal]+covariates)+f" + C({pair})"
    if seed in d.columns and d[seed].nunique()>1:
        rhs += f" + C({seed})"
    formula=f"{outcome} ~ {rhs}"
    fit0=smf.ols(formula,data=d).fit()
    if d[pair].nunique()>=4:
        fit=smf.ols(formula,data=d).fit(cov_type='cluster',cov_kwds={'groups':d[pair]})
    else:
        fit=smf.ols(formula,data=d).fit(cov_type='HC3')
    X=np.asarray(fit0.model.exog,float)
    out={"beta":float(fit.params[signal]),"se":float(fit.bse[signal]),"p":float(fit.pvalues[signal]),
         "n":int(len(d)),"n_pairs":int(d[pair].nunique()),"design_cols":int(X.shape[1]),
         "design_rank":int(np.linalg.matrix_rank(X)),"df_resid":float(fit0.df_resid),
         "condition_number":float(np.linalg.cond(X)),"formula":formula,
         "summary":fit.summary().as_text()}
    if covariates:
        out['covariates']={c:{'coef':float(fit.params[c]),'se':float(fit.bse[c]),'p':float(fit.pvalues[c])}
                           for c in covariates if c in fit.params.index}
    return out


def group_cv_regression(df:pd.DataFrame,target:str,feature_cols:list[str],group_col:str|None=None):
    """Out-of-sample linear prediction, optionally holding out an entire task group."""
    cols=feature_cols+[target]+([group_col] if group_col else [])
    d=df[cols].dropna().copy()
    if len(d)<3 or (group_col and d[group_col].nunique()<2):
        return {'n':int(len(d)),'features':feature_cols,'mae':np.nan,'r2':np.nan,
                'pred_r':np.nan,'pred_p':np.nan,'pred_rho':np.nan,'pred_rho_p':np.nan}
    y=d[target].to_numpy(float); X=d[feature_cols].to_numpy(float)
    if group_col:
        splits=LeaveOneGroupOut().split(X,y,d[group_col].to_numpy())
    else:
        splits=LeaveOneOut().split(X,y)
    pred=np.full(len(d),np.nan)
    for tr,te in splits:
        model=Ridge(alpha=1e-6).fit(X[tr],y[tr]); pred[te]=model.predict(X[te])
    m=np.isfinite(pred)&np.isfinite(y)
    co=corr(pred[m],y[m])
    return {'n':int(m.sum()),'features':feature_cols,
            'mae':float(mean_absolute_error(y[m],pred[m])) if m.any() else np.nan,
            'r2':float(r2_score(y[m],pred[m])) if m.sum()>=2 else np.nan,
            **{f'pred_{k}':v for k,v in co.items()}}


def paired_mean_ci(diff,alpha=0.05):
    d=np.asarray(diff,float); d=d[np.isfinite(d)]; n=len(d)
    if n==0: return {'n':0,'mean':np.nan,'ci_low':np.nan,'ci_high':np.nan,'sd':np.nan}
    mean=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else 0.0
    if n>1:
        half=float(t.ppf(1-alpha/2,n-1)*sd/np.sqrt(n)); lo=mean-half; hi=mean+half
    else:
        lo=hi=np.nan
    return {'n':n,'mean':mean,'ci_low':float(lo),'ci_high':float(hi),'sd':sd}
