#!/bin/bash
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
LOG=results/logs/overnight3.log; log(){ echo "$(date -Is) $*" | tee -a $LOG; }
free(){ df --output=avail -BG / | tail -1 | tr -dc 0-9; }
step(){ local name=$1; shift; [ "$(free)" -lt 6 ] && { log "ABORT $name: disk $(free)G"; return 99; }; log "START $name"
  { echo "START_EPOCH $(date +%s)"; "$@"; echo "EXIT_CODE $? END_EPOCH $(date +%s)"; } > results/logs/$name.log 2>&1
  local rc=$(grep -o 'EXIT_CODE [0-9]*' results/logs/$name.log | tail -1 | awk '{print $2}'); log "END $name rc=$rc"; return ${rc:-1}; }
vr(){ local name=$1; shift; [ -s results/vr/$name/stats.json ] && { log "SKIP $name"; return 0; }; rm -rf results/vr/$name/ipc; step vr_$name python -u verify_rollback.py run --run $name "$@"; rm -rf results/vr/$name/ipc; }
log "===== overnight3 start, disk $(free)G ====="
# 1. 可复现性
vr tp2_C64_v2_rep --tp 2 --lnc 2 --chunk 64 --n 512 --verifiers 2 --lanes 2 --fused
python - <<'PY' 2>&1 | tee -a $LOG
import json
def load(p): return {json.loads(l)["seq_id"]: json.loads(l) for l in open(p) if "_meta" not in l}
a, b = load("results/vr/tp2_C64_v2/final.jsonl"), load("results/vr/tp2_C64_v2_rep/final.jsonl")
same = sum(a[k]["token_ids"] == b[k]["token_ids"] and a[k]["logprobs"] == b[k]["logprobs"] for k in a)
print(f"REPRO tp2_C64_v2 vs rep: identical final trajectories {same}/{len(a)}")
PY
# 2. T=1024, bucket 2048
vr tp2_C64_T1024_v2 --tp 2 --lnc 2 --chunk 64 --n 512 --verifiers 2 --lanes 2 --fused --max-new 1024 --bucket-len 2048
[ -s results/vr/tp2_C64_T1024_v2/final.jsonl ] && step check_T1024 bash -c "python -u recompute.py --backend neuron --sample results/vr/tp2_C64_T1024_v2/final.jsonl --bucket-len 2048 --lnc 2 --tp 1 --batch-size 1 --tag vr_check_tp2_C64_T1024_v2 && python analyze.py --sample results/vr/tp2_C64_T1024_v2/final.jsonl --recompute results/recompute/vr_check_tp2_C64_T1024_v2.jsonl --out results/analysis/vr_check_tp2_C64_T1024_v2 --title T1024"
# 3. Qwen3-8B
if [ "$(free)" -ge 22 ]; then
  step dl_8b python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))"
  [ -s results/samples/q8b_tp4_lnc2.jsonl ] || step sample_8b python -u sample.py --backend neuron --model Qwen/Qwen3-8B --tp 4 --lnc 2 --tag q8b_tp4_lnc2
  [ -s results/samples/q8b_tp4_lnc2.jsonl ] && step recompute_8b python -u recompute.py --backend neuron --sample results/samples/q8b_tp4_lnc2.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag rc_q8b_lnc2
  [ -s results/recompute/rc_q8b_lnc2.jsonl ] && step analyze_8b python analyze.py --sample results/samples/q8b_tp4_lnc2.jsonl --recompute results/recompute/rc_q8b_lnc2.jsonl --out results/analysis/q8b_tp4_lnc2 --title "Qwen3-8B NEFF tp4 lnc2 vs XLA tp1 lnc2"
  [ -s results/recompute/rc_q8b_lnc2.jsonl ] && python rejection_analysis.py --sample results/samples/q8b_tp4_lnc2.jsonl --recompute results/recompute/rc_q8b_lnc2.jsonl --prefill-cost 0.05 > results/analysis/q8b_rejection.txt 2>&1
else log "SKIP 8B: disk $(free)G"; fi
python tools/vr_summary.py > results/vr/SUMMARY.md 2>&1
log "===== ALL DONE, disk $(free)G ====="
