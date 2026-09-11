#!/bin/bash
# 编译器 vs 图 分解 + 分桶验证器可行性
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
export TMPDIR=/scratch/ncc-work NEURON_COMPILE_CACHE_URL=/scratch/neuron-cache/train
LOG=results/logs/decomp.log; log(){ echo "$(date -Is) $*" | tee -a $LOG; }
log "===== START ====="
log "--- 分桶验证器可行性 ---"
timeout 1800 python -u tools/probe_cross_bucket.py > results/logs/probe_cross_bucket.log 2>&1
grep -E "RESULT|done|NCC_|Traceback" results/logs/probe_cross_bucket.log | tee -a $LOG
unset NEURON_COMPILE_CACHE_URL
log "--- CTE 探针 (同前缀不同编译器) ---"
export NEURON_COMPILE_CACHE_URL=/scratch/neuron-cache/infer
python - <<'PY'
import json
S="results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"
rows=[json.loads(l) for l in open(S)][1:]
for k in (1,64,256):
    with open(f"results/samples/cte_probe_k{k}.in.jsonl","w") as f:
        for r in rows:
            if len(r["logprobs"])>k: f.write(json.dumps({"seq_id":r["seq_id"],"token_ids":r["token_ids"][:r["prompt_len"]+k]})+"\n")
PY
for k in 1 64 256; do
  [ -s results/samples/cte_probe_k$k.jsonl ] && { log "SKIP k=$k"; continue; }
  log "START cte_probe_k$k"
  python -u sample.py --backend neuron --tp 4 --lnc 2 --max-new 1 --topk-logprobs 256 \
    --prefix-file results/samples/cte_probe_k$k.in.jsonl --tag cte_probe_k$k > results/logs/cte_probe_k$k.log 2>&1
  log "END cte_probe_k$k rc=$? ($(grep -c 'Compilation Successfully' results/logs/cte_probe_k$k.log) NEFF)"
done
log "===== ALL DONE ====="
