#!/usr/bin/env bash
# Reproduces every result in the paper. Each experiment is independent and can be run separately.
# Datasets are downloaded by torchvision into $SLT_DATA on first use. A single GPU is assumed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export SLT_DATA="${SLT_DATA:-$ROOT/data}"
OUT="$ROOT/runs"; mkdir -p "$OUT"

# 1. Succession experiments E1-E5 (write their records into the working directory)
cd "$OUT"
python "$ROOT/succession/E1_multipleRun.py"            # e1_checkpoint.json
python "$ROOT/succession/E1_singleRun.py"              # e1_single_results.json
python "$ROOT/succession/E1o5.py"                      # e15_checkpoint.json
python "$ROOT/succession/E2.py"                        # e2_checkpoint.json
python "$ROOT/succession/E3_v1.py" | tee e3v1_stdout.txt
python "$ROOT/succession/E3_v2.py"                     # e3v2_checkpoint.json
python "$ROOT/succession/E4_v3.py" | tee e4_stdout.txt
python "$ROOT/succession/E5.py"    | tee e5_stdout.txt
python "$ROOT/analysis/records_from_logs.py" --e3v1 e3v1_stdout.txt --e4 e4_stdout.txt --e5 e5_stdout.txt --out "$OUT/succession"

# 2. Transition suite: E6 predictor comparison, niche/fitness decomposition, E8 controlled relation
cd "$OUT"
python "$ROOT/transition_suite/experiments/planned/E0_metric_horserace.py"
python "$ROOT/transition_suite/experiments/planned/E6_chesson_cifar100.py"
SLT_E8_MODE=mnist_perm   python "$ROOT/transition_suite/experiments/planned/E8_similarity_sweep.py"
SLT_E8_MODE=cifar_rotate python "$ROOT/transition_suite/experiments/planned/E8_similarity_sweep.py"

# 3. Mechanism suite (results under $SLT_RESULTS_ROOT)
export SLT_RESULTS_ROOT="$OUT/mechanism_suite"
M="$ROOT/mechanism_suite"
python "$M/scripts/prepare_data.py"
python "$M/experiments/revision/R0_revalidate_rho_cifar10.py"
python "$M/experiments/revision/R1_mutual_coexistence.py"
python "$M/experiments/revision/R5_cifar100_directional_rho.py"
SLT_R5_FRESH_PARTITION=1 python "$M/experiments/revision/R5_cifar100_directional_rho.py"
for e in A1_definition4_competition_lv A2_local_lv_validity A3_rstar_task_resources A7_succession_stages B1_habitat_interaction \
         B7_gated_intervention R5_sem_cifar100 S1_rho_sensitivity S2_jacobian_probe_sweep S3_omega_regime_sweep S4b_classil_omega \
         S6_label_light_omega; do
  python "$M/experiments/freeze20260917/$e.py"
done
python "$M/scripts/prepare_pretrained.py" && python "$M/experiments/freeze20260917/S5_pretrained_backbone.py"

# 4. Protection study (three seeds)
cd "$ROOT/protection_suite"
for s in 0 1 2; do
  python experiments/e00_build_resident_bank.py --seed $s --device cuda --data-root "$SLT_DATA" --out-dir "$OUT/protection_suite/bank_seed$s"
  python experiments/e03_protection_benefit.py --seed $s --device cuda --data-root "$SLT_DATA" \
         --resident-bank-manifest "$OUT/protection_suite/bank_seed$s/manifest.json" --out "$OUT/protection_suite/seed${s}_p3_protection.csv"
done

# 5. Figures and every derived statistic, from the records shipped in records/
cd "$ROOT"
python analysis/make_figures_and_stats.py
