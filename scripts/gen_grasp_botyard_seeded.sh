#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/gen_grasp_botyard_seeded.sh SCALE COUNT [Hydra overrides...]
SCALE="${1:-1.0}"
TARGET="${2:-500}"
if (( $# >= 2 )); then shift 2; else set --; fi
# CPU physics/contact queries; the GPU is not used for this production filter.
python gen_grasp.py task=BotyardHandGrasp \
  pipeline=cpu sim_device=cpu rl_device=cpu headless=True \
  task.env.baseObjScale="$SCALE" task.env.targetGraspCount="$TARGET" \
  task.env.numEnvs="${NUM_ENVS:-128}" "$@"
