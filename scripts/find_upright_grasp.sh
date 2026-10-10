#!/usr/bin/env bash
set -euo pipefail
# Run from the HORA repository root. Does not require cache or checkpoints.
python tools/validate_upright_candidates.py --candidates upright_candidates.json --steps 600 --per-seed 24 --robust-trials 64 --viewer
cp upright_search/BotyardUprightGrasp.yaml configs/task/BotyardUprightGrasp.yaml
printf '\nValidated config copied to configs/task/BotyardUprightGrasp.yaml\n'
