#!/usr/bin/env bash
set -euo pipefail
cd /home/grelcy/hora
MODE="${1:-}"
if [[ "$MODE" != A && "$MODE" != B ]]; then echo '用法: bash run_ab.sh A 或 bash run_ab.sh B'; exit 2; fi
CKPT="/home/grelcy/hora/outputs/BotyardHandHora/constrained_debug_1m/stage1_nn/best.pth"
[[ -f "$CKPT" ]] || { echo "找不到源 checkpoint: $CKPT"; exit 3; }
mkdir -p ab_runs
if [[ "$MODE" == A ]]; then
  OUT="BotyardHandHora/ab_A_continuation_6m"
  WEIGHT=0.0
else
  OUT="BotyardHandHora/ab_B_recovery_6m"
  WEIGHT=0.10
fi
LOG="ab_runs/${MODE}_6m.log"
# Reward differs in B; resetting the best-score bookkeeping is essential.
# Keep identical actor/critic weights and optimizer state in both groups.
if [[ "$MODE" == B ]]; then
  export AB_SOURCE_CKPT="$CKPT"
  python - <<'PYCHECK'
import os, torch
src=os.environ['AB_SOURCE_CKPT']
dst='ab_runs/B_start_from_1m.pth'
x=torch.load(src,map_location='cpu')
x['best_rewards']=float('-inf')
x['best_step']=0
torch.save(x,dst)
print('[AB] B checkpoint prepared with reset best-score bookkeeping')
PYCHECK
  CKPT="/home/grelcy/hora/ab_runs/B_start_from_1m.pth"
fi
echo "[AB] experiment=$MODE, base=$CKPT, total=6M, recovery_weight=$WEIGHT, output=$OUT"
# Keep the same physical setup for both experiments.
python -u train_botyard.py \
  task=BotyardHandHora pipeline=cpu sim_device=cuda:0 rl_device=cuda:0 \
  '++task.env.trainingMode=constrained' \
  '++task.env.graspCacheCountTag=50k' \
  '++task.env.randomization.randomizeScale=true' \
  '++task.env.randomization.scaleListInit=true' \
  '++task.env.randomization.randomizeScaleList=[1.0]' \
  '++task.env.baseObjScale=1.0' \
  '++task.env.forceScale=0.0' \
  '++task.env.randomForceProbScalar=0.0' \
  '++task.env.controller.actionStepRad=0.006' \
  '++task.env.debugMetricsEvery=100' \
  "++task.env.metricsPath=ab_runs/${MODE}_metrics.jsonl" \
  "++task.env.abRecovery.weight=$WEIGHT" \
  '++train.ppo.learning_rate=0.0003' \
  '++train.ppo.max_agent_steps=6000000' \
  '++train.ppo.save_best_after=20' \
  "++train.ppo.output_name=$OUT" \
  "checkpoint=$CKPT" \
  headless=True \
  2>&1 | tee "$LOG" | grep --line-buffered -E '^\[PPO\]|^\[AB\]|^\[SAFETY\]|^\[RESUME\]|^\[FINAL CHECKPOINT\]|^\[TRAIN\]|Traceback|Error|Exception|max steps achieved'

# If A never exceeds the source best, preserve source policy as best baseline.
if [[ "$MODE" == A && ! -f "outputs/$OUT/stage1_nn/best.pth" ]]; then
  cp /home/grelcy/hora/outputs/BotyardHandHora/constrained_debug_1m/stage1_nn/best.pth "outputs/$OUT/stage1_nn/best.pth"
  echo '[AB] No new best in A, preserved source 1M best.pth for visualization.'
fi
