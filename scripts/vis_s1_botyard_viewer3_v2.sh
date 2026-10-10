#!/bin/bash
set -euo pipefail

# Visualize Viewer3 BotYard Stage1 using the generated grasp cache.
# Run from ~/hora:
#   bash scripts/vis_s1_botyard_viewer3_v2.sh
#
# Optional:
#   SCALE=0.9 bash scripts/vis_s1_botyard_viewer3_v2.sh
#   SCALE=0.8 bash scripts/vis_s1_botyard_viewer3_v2.sh

SCALE="${SCALE:-1.0}"
RUN_NAME="botyard_stage1_viewer3_multiscale"
CKPT="outputs/BotyardHandHora/${RUN_NAME}/stage1_nn/best.pth"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 2
fi

case "${SCALE}" in
  1.0|0.9|0.8) ;;
  *)
    echo "ERROR: SCALE must be 1.0, 0.9, or 0.8"
    exit 2
    ;;
esac

echo "============================================================"
echo "BotYard Viewer3 Stage1 visualization"
echo "checkpoint : ${CKPT}"
echo "scale      : ${SCALE}"
echo "============================================================"

python train.py \
  task=BotyardHandHora \
  headless=False \
  pipeline=gpu \
  task.env.numEnvs=1 \
  test=True \
  task.on_evaluation=False \
  task.env.object.type=cylinder_default \
  task.env.baseObjScale="${SCALE}" \
  task.env.grasp_cache_name=botyard_cylinder_viewer3 \
  task.env.randomization.randomizeMass=False \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizePDGains=False \
  task.env.randomization.randomizeScale=True \
  task.env.randomization.scaleListInit=True \
  task.env.randomization.randomizeScaleList="[${SCALE}]" \
  task.env.forceScale=0 \
  task.env.randomForceProbScalar=0 \
  task.env.reset_height_threshold=0.50 \
  train.algo=PPO \
  train.ppo.priv_info=True \
  train.ppo.proprio_adapt=False \
  train.ppo.output_name=BotyardHandHora/"${RUN_NAME}" \
  checkpoint="${CKPT}"
