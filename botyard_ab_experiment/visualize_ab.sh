#!/usr/bin/env bash
set -euo pipefail
cd /home/grelcy/hora
MODE="${1:-}"
if [[ "$MODE" != A && "$MODE" != B ]]; then echo '用法: bash visualize_ab.sh A 或 bash visualize_ab.sh B'; exit 2; fi
if [[ "$MODE" == A ]]; then OUT=BotyardHandHora/ab_A_continuation_6m; WEIGHT=0.0; else OUT=BotyardHandHora/ab_B_recovery_6m; WEIGHT=0.10; fi
CKPT="/home/grelcy/hora/outputs/$OUT/stage1_nn/best.pth"
[[ -f "$CKPT" ]] || { echo "模型不存在: $CKPT"; exit 3; }
python -u train_botyard.py \
 task=BotyardHandHora pipeline=cpu sim_device=cuda:0 rl_device=cuda:0 test=True headless=False \
 '++task.env.trainingMode=constrained' '++task.env.graspCacheCountTag=50k' \
 '++task.env.randomization.randomizeScaleList=[1.0]' \
 '++task.env.randomization.randomizeScale=true' '++task.env.randomization.scaleListInit=true' \
 '++task.env.forceScale=0.0' '++task.env.randomForceProbScalar=0.0' \
 '++task.env.controller.actionStepRad=0.006' '++task.env.numEnvs=1' \
 "++task.env.abRecovery.weight=$WEIGHT" \
 "train.ppo.output_name=$OUT" "checkpoint=$CKPT"
