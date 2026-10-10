#!/bin/bash
# Usage: scripts/train_s1_botyard.sh GPU_ID SEED OUTPUT_NAME [extra Hydra args...]
GPUS=$1
SEED=$2
CACHE=$3
shift 3
EXTRA_ARGS="$@"

echo extra "${EXTRA_ARGS}"
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=BotyardHandHora headless=True seed=${SEED} \
  task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
  train.algo=PPO \
  train.ppo.priv_info=True train.ppo.proprio_adapt=False \
  train.ppo.output_name=BotyardHandHora/"${CACHE}" \
  ${EXTRA_ARGS}
