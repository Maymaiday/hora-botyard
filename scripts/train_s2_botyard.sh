#!/bin/bash
# Usage: scripts/train_s2_botyard.sh GPU_ID SEED OUTPUT_NAME [extra Hydra args...]
GPUS=$1
SEED=$2
CACHE=$3
shift 3
EXTRA_ARGS="$@"

echo extra "${EXTRA_ARGS}"
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=BotyardHandHora headless=True seed=${SEED} \
  task.env.numEnvs=1024 \
  task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
  train.algo=ProprioAdapt \
  train.ppo.priv_info=True train.ppo.proprio_adapt=True \
  train.ppo.output_name=BotyardHandHora/"${CACHE}" \
  checkpoint=outputs/BotyardHandHora/"${CACHE}"/stage1_nn/best.pth \
  ${EXTRA_ARGS}
