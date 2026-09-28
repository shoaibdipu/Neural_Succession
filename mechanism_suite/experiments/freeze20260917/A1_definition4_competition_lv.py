#!/usr/bin/env python3
"""A1 independent composite: T1 feature-pressure test + R6b fixed-alpha LV test.
Both subtests recreate resident models and data independently; no prior result is read.
"""
from __future__ import annotations
import json,os,runpy,sys,time,shutil,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'core'))
from slt_freeze import exp_outdir,dump_result,env_bool
from slt_confirmatory import run_metadata
from slt_revision import leave_one_task_out_forecast
EXP='A1_definition4_competition_lv'; OUT=exp_outdir(ROOT,EXP); SMOKE=env_bool('SLT_SMOKE')

def wilson_lower(k,n,z=1.96):
    if not n: return None
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half


def run_script(path, env):
    old=os.getcwd(); os.chdir(OUT)
    prior={k:os.environ.get(k) for k in env}
    try:
        os.environ.update({k:str(v) for k,v in env.items()}); runpy.run_path(str(path),run_name='__main__')
    finally:
        for k,v in prior.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v
        os.chdir(old)

def main():
    t0=time.time(); lr_primary=float(os.environ.get('SLT_A1_LR','0.03')); lr_backup=0.1; shared_dir=OUT/'shared_residents'
    def run_t1(lr):
        shutil.rmtree(shared_dir,ignore_errors=True); shared_dir.mkdir(parents=True,exist_ok=True)
        env={'SLT_SMOKE':'1' if SMOKE else '0','SLT_T1_DATASET':'cifar10','SLT_T1_SEEDS':'1' if SMOKE else '3','SLT_T1_EPOCHS':'2' if SMOKE else '20','SLT_T1_LR':str(lr),'SLT_T1_JAC_PROBE':'64' if SMOKE else '1024','SLT_T1_PROBE':'64' if SMOKE else '1024','SLT_T1_SHUFFLE':'100' if SMOKE else '5000','SLT_T1_BATCH':'32','SLT_SHARED_RESIDENT_DIR':str(shared_dir),'SLT_SHARED_RESIDENT_MODE':'save'}
        run_script(ROOT/'experiments/theory_validation/T1_net_pressure_forgetting.py',env)
        return json.loads((OUT/'t1_net_pressure_cifar10_results.json').read_text())
    t1=run_t1(lr_primary); f=np.asarray([r['forgetting'] for r in t1.get('rows',[])],float); f_sd=float(np.std(f)) if len(f) else 0.; chosen=lr_primary; backup_used=False
    if (not SMOKE) and f_sd<.05:
        chosen=lr_backup; backup_used=True; t1=run_t1(chosen); f=np.asarray([r['forgetting'] for r in t1.get('rows',[])],float); f_sd=float(np.std(f)) if len(f) else 0.
    lv={'SLT_SMOKE':'1' if SMOKE else '0','SLT_R6B_DATASET':'cifar10','SLT_R6B_K':'4,8','SLT_R6B_SEEDS':'1' if SMOKE else '3','SLT_R6B_A_EPOCHS':'2' if SMOKE else '20','SLT_R6B_STEPS':'60' if SMOKE else '1200','SLT_R6B_CHECK':'5' if SMOKE else '4','SLT_R6B_MAX_PAIRS':'2' if SMOKE else '20','SLT_R6B_SHUFFLE':'100' if SMOKE else '5000','SLT_R6B_PROBE':'64' if SMOKE else '1024','SLT_R6B_LR':str(chosen),'SLT_R6B_BATCH':'32','SLT_SHARED_RESIDENT_DIR':str(shared_dir),'SLT_SHARED_RESIDENT_MODE':'load'}
    run_script(ROOT/'experiments/revision/R6b_fixed_definition4_lv.py',lv)
    r6=json.loads((OUT/'r6b_fixed_definition4_lv_cifar10_results.json').read_text())
    d4=t1.get('summary',{}).get('definition4_feature_level',{}); task=t1.get('summary',{}).get('def4_strength_weighted',{})
    fits=[z for r in r6.get('rows',[]) for z in r.get('fits',[]) if z.get('K')==4 and 'error' not in z.get('fixed_definition4',{})]
    wins=0
    for z in fits:
        ff=z['fixed_definition4']; obs=ff.get('heldout_R2_median'); c=ff.get('constant_rate_R2_median'); sh=ff.get('shuffled_alpha_R2_median_mean')
        if all(v is not None and np.isfinite(v) for v in [obs,c,sh]) and obs>c and obs>sh: wins+=1
    # Frozen task-level inference: exact task-label permutation from T1 plus LOTO sign stability.
    qap=task.get('task_permutation') or {}
    loto=leave_one_task_out_forecast(t1.get('pair_means',[]),'def4_strength_weighted','forgetting',n_tasks=5,incoming_only=True)
    loto_pos=sum(1 for f0 in loto.get('folds',[]) if f0.get('coef') is not None and f0['coef']>0)
    r2vals=[z['fixed_definition4'].get('heldout_R2_median') for z in fits]
    r2vals=[float(v) for v in r2vals if v is not None and np.isfinite(v)]
    med_r2=float(np.median(r2vals)) if r2vals else None
    criteria={'forgetting_sd':f_sd,'identifiable_regime':bool(f_sd>=.05),'chosen_sgd_lr':chosen,'backup_lr_used':backup_used,
              'feature_median_partial_r_gt_0p10':bool(d4.get('median_partial_r_controlling_log_x0') is not None and d4['median_partial_r_controlling_log_x0']>.10),
              'feature_fraction_significant_gt_half':bool(d4.get('fraction_null_p_lt_0p05') is not None and d4['fraction_null_p_lt_0p05']>.5),
              'task_r_ge_0p50':bool(task.get('pearson_r') is not None and task['pearson_r']>=.5),
              'task_permutation_p_le_0p05':bool(qap.get('p_task_permutation') is not None and qap['p_task_permutation']<=.05),
              'task_permutation':qap,'loto_positive_slope_folds':loto_pos,'loto_n_folds':len(loto.get('folds',[])),
              'loto_sign_ge_4_of_5':bool(loto_pos>=4),
              'fixed_alpha_median_heldout_R2':med_r2,'fixed_alpha_median_R2_gt_zero':bool(med_r2 is not None and med_r2>0),
              'fixed_alpha_wins_majority':bool(len(fits)>0 and wins>len(fits)/2),'fixed_alpha_wilson_lower':wilson_lower(wins,len(fits)),'fixed_alpha_wilson_lower_gt_half':bool(len(fits)>0 and wilson_lower(wins,len(fits))>0.5),'fixed_alpha_wins':wins,'fixed_alpha_n':len(fits),'shared_resident_checkpoints':True}
    criteria['task_corollary_pass']=bool(criteria['task_r_ge_0p50'] and criteria['task_permutation_p_le_0p05'] and criteria['loto_sign_ge_4_of_5'])
    criteria['lv_global_pass']=bool(criteria['fixed_alpha_median_R2_gt_zero'] and criteria['fixed_alpha_wilson_lower_gt_half'])
    out={'criteria':criteria,'paths':{'T1':'t1_net_pressure_cifar10_results.json','R6b':'r6b_fixed_definition4_lv_cifar10_results.json'},'_meta':run_metadata(ROOT,{'experiment':EXP,'elapsed_s':time.time()-t0})}
    std=dump_result(ROOT,EXP,out); (OUT/'A1_summary.json').write_text(json.dumps(out,indent=2)); print(json.dumps(criteria,indent=2))
if __name__=='__main__': main()
