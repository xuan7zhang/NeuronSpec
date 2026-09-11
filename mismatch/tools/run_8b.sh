#!/bin/bash
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
export TMPDIR=/scratch/ncc-work BASE_COMPILE_WORK_DIR=/scratch/ncc-work/nxd_model
LOG=results/logs/run_8b.log; log(){ echo "$(date -Is) $*" | tee -a $LOG; }
grep -q PROBE_OK results/logs/probe_8b_train.log && log "probe already OK, skipping" || { log "START probe_8b_train"; timeout 2400 python -u tools/probe_8b_train.py > results/logs/probe_8b_train.log 2>&1; }
if grep -q PROBE_OK results/logs/probe_8b_train.log; then log "PROBE_OK $(grep PROBE_OK results/logs/probe_8b_train.log)"; else log "PROBE_FAIL $(grep -E 'NCC_|Error|Killed' results/logs/probe_8b_train.log | head -1 | cut -c1-200)"; exit 1; fi
log "START sample_8b"; NEURON_COMPILED_ARTIFACTS=/scratch/nxd-artifacts/q8b_tp4_lnc2 NEURON_COMPILE_CACHE_URL=/scratch/neuron-cache/infer \
  python -u sample.py --backend neuron --model Qwen/Qwen3-8B --tp 4 --lnc 2 --tag q8b_tp4_lnc2 > results/logs/sample_8b.log 2>&1; log "END sample_8b rc=$?"
[ -s results/samples/q8b_tp4_lnc2.jsonl ] || exit 1
log "START recompute_8b"; NEURON_COMPILE_CACHE_URL=/scratch/neuron-cache/train python -u recompute.py --backend neuron --sample results/samples/q8b_tp4_lnc2.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag rc_q8b_lnc2 > results/logs/recompute_8b.log 2>&1; log "END recompute_8b rc=$?"
python analyze.py --sample results/samples/q8b_tp4_lnc2.jsonl --recompute results/recompute/rc_q8b_lnc2.jsonl --out results/analysis/q8b_tp4_lnc2 --title "Qwen3-8B NEFF tp4 lnc2 vs XLA tp1 lnc2" > results/logs/analyze_8b.log 2>&1
python rejection_analysis.py --sample results/samples/q8b_tp4_lnc2.jsonl --recompute results/recompute/rc_q8b_lnc2.jsonl --prefill-cost 0.05 > results/analysis/q8b_rejection.txt 2>&1
log "ALL_DONE disk_root=$(df --output=avail -h / | tail -1) scratch=$(df --output=avail -h /scratch | tail -1)"
