#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/train_s1_botyard_seeded.sh GPU_ID SEED OUTPUT_NAME COUNT_TAG [Hydra overrides...]
GPU_ID="${1:?GPU_ID}"; RUN_SEED="${2:?SEED}"; OUTPUT_NAME="${3:?OUTPUT_NAME}"; COUNT_TAG="${4:?COUNT_TAG}"
shift 4
CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py task=BotyardHandHora headless=True seed="$RUN_SEED" \
  task.env.graspCacheCountTag="$COUNT_TAG" train.algo=PPO \
  train.ppo.priv_info=True train.ppo.proprio_adapt=False \
  train.ppo.output_name="BotyardHandHora/$OUTPUT_NAME" "$@"
