#!/bin/bash
# Sweep all spec-decoding traces + AR baseline, one container run each.
# Usage: bash scripts/sweep.sh [results_dir]
set -u
RESULTS=${1:-results/raw}
mkdir -p "$RESULTS"

TRACES="rq1_ar p0_spec_g1 p0_spec_g2 p0_spec_b1 sw_g8_b1 sw_g2_b4 p0_spec sw_g8_b4 probe_bspec"

for t in $TRACES; do
  out="$RESULTS/$t.json"
  if [ -f "$out" ]; then echo ">>> skip $t (exists)"; continue; fi
  echo ">>> $t"
  docker run --rm --device=/dev/neuron0 \
    -v /home/ec2-user/neuronspec:/workspace/neuronspec \
    -v /data/models:/home/ubuntu/models \
    -v /data/traced_model:/home/ubuntu/traced_model \
    -w /workspace/neuronspec \
    neuronmm:repro -c "python scripts/run_spec.py \
      --compiled-model-path /home/ubuntu/traced_model/$t \
      --benchmark --json-out $RESULTS/$t.json" \
    > "$RESULTS/$t.log" 2>&1
  echo "    exit=$? $(python3 -c "
import json,sys
try:
    r=json.load(open('$out'))
    print(f\"b={r['batch_size']} spec={r['speculation_length']} {r['tok_per_s']:.1f} tok/s (generate)\")
except Exception: print('no result')" 2>/dev/null)"
done
echo "done"
