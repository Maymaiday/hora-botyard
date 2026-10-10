#!/usr/bin/env bash
set -euo pipefail
# Only run after find_upright_grasp.sh reports a validated winner.
# Uses the validated torque-PD holding targets and a fresh cache name.
mkdir -p cache
python gen_grasp.py task=BotyardUprightGrasp train=BotyardHandGrasp \
  pipeline=cpu sim_device=cpu rl_device=cpu headless=True test=True \
  task.env.numEnvs=128 task.env.genGrasps=True \
  task.env.controller.controlFrequencyInv=8 task.env.episodeLength=80 \
  task.env.randomization.randomizeMass=False \
  task.env.randomization.randomizePDGains=False \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizeScale=False \
  task.env.randomization.scaleListInit=False \
  task.env.baseObjScale=1.0 task.env.targetGraspCount=50000 "$@"
