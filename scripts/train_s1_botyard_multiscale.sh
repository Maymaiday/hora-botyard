#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/train_s1_botyard_multiscale.sh GPU_ID SEED OUTPUT_NAME [extra Hydra args...]

GPU_ID="${1:-0}"
SEED="${2:-2026}"
OUTPUT_NAME="${3:-botyard_stage1_viewer3_multiscale}"
shift 3 || true
EXTRA_ARGS="$@"

for f in \
  cache/botyard_cylinder_viewer3_grasp_50k_s08.npy \
  cache/botyard_cylinder_viewer3_grasp_50k_s09.npy \
  cache/botyard_cylinder_viewer3_grasp_50k_s10.npy
do
  if [ ! -f "$f" ]; then
    echo "ERROR: required cache missing: $f"
    exit 2
  fi
done

mkdir -p logs/viewer3_multiscale_stage1

echo "[Stage1 multiscale]"
echo "  scales      = [0.8, 0.9, 1.0]"
echo "  output_name = ${OUTPUT_NAME}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python train.py \
  task=BotyardHandHora \
  headless=True \
  seed="${SEED}" \
  task.env.object.type=cylinder_default \
  task.env.grasp_cache_name=botyard_cylinder_viewer3 \
  task.env.baseObjScale=1.0 \
  task.env.randomization.randomizeScale=True \
  task.env.randomization.scaleListInit=True \
  task.env.randomization.randomizeScaleList='[0.8,0.9,1.0]' \
  task.env.forceScale=2 \
  task.env.randomForceProbScalar=0.25 \
  train.algo=PPO \
  train.ppo.priv_info=True \
  train.ppo.proprio_adapt=False \
  train.ppo.output_name=BotyardHandHora/"${OUTPUT_NAME}" \
  ${EXTRA_ARGS} \
  2>&1 | tee logs/viewer3_multiscale_stage1/train.log
