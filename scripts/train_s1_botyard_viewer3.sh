#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/train_s1_botyard_viewer3.sh GPU_ID SEED OUTPUT_NAME [extra Hydra args...]

GPU_ID=$1
SEED=$2
OUTPUT_NAME=$3
shift 3
EXTRA_ARGS="$@"

echo "extra ${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPU_ID} \
python train.py \
  task=BotyardHandHora \
  headless=True \
  seed=${SEED} \
  task.env.object.type=cylinder_default \
  task.env.baseObjScale=1.0 \
  task.env.grasp_cache_name=botyard_cylinder_viewer3 \
  task.env.randomization.randomizeScale=False \
  task.env.randomization.scaleListInit=False \
  task.env.forceScale=2 \
  task.env.randomForceProbScalar=0.25 \
  train.algo=PPO \
  train.ppo.priv_info=True \
  train.ppo.proprio_adapt=False \
  train.ppo.output_name=BotyardHandHora/"${OUTPUT_NAME}" \
  ${EXTRA_ARGS}
