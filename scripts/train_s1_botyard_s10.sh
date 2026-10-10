#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -u tools/run_botyard_full_pipeline.py \
  --phase baseline --scales 1.0 --cache-count 50000 \
  --envs 512 --sim-device cuda:0 --rl-device cuda:0 \
  --train-steps 10000000 "$@"
