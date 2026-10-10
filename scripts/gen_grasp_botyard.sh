#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/gen_grasp_botyard.sh GPU_ID SCALE TARGET_COUNT [extra Hydra args...]
#
# Examples:
#   scripts/gen_grasp_botyard.sh 0 0.9 500
#   scripts/gen_grasp_botyard.sh 0 0.9 50000 task.env.numEnvs=2048

GPU_ID="${1:-0}"
SCALE="${2:-1.0}"
TARGET="${3:-5000}"
shift 3 || true
EXTRA_ARGS="$@"

NUM_ENVS="${NUM_ENVS:-2048}"
EP_LEN="${EP_LEN:-80}"

echo "[Viewer3 cylinder grasp generation]"
echo "  scale     = ${SCALE}"
echo "  target    = ${TARGET}"
echo "  numEnvs   = ${NUM_ENVS}"
echo "  episode   = ${EP_LEN}"
echo "  extra     = ${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python gen_grasp.py \
  task=BotyardHandGrasp \
  headless=True \
  pipeline=cpu \
  task.env.numEnvs="${NUM_ENVS}" \
  test=True \
  task.env.controller.controlFrequencyInv=8 \
  task.env.episodeLength="${EP_LEN}" \
  task.env.controller.torque_control=False \
  task.env.genGrasps=True \
  task.env.targetGraspCount="${TARGET}" \
  task.env.baseObjScale="${SCALE}" \
  task.env.object.type=cylinder_default \
  task.env.grasp_cache_name=botyard_cylinder_viewer3 \
  task.env.graspPoseNoise=0.01 \
  task.env.graspObjectXYNoise=0.003 \
  task.env.graspObjectZNoise=0.002 \
  task.env.minNearFingerCount=4 \
  task.env.minGraspContacts=4 \
  task.env.requireThumbContact=True \
  task.env.rejectSupportContact=True \
  task.env.maxCylinderTiltDeg=20.0 \
  task.env.minGraspObjectHeight=0.50 \
  task.env.randomization.randomizeMass=False \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizePDGains=False \
  task.env.randomization.randomizeScale=False \
  train.ppo.priv_info=True \
  ${EXTRA_ARGS}
