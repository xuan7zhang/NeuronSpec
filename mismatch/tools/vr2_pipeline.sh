#!/bin/bash
# v2 工程链: 等融合图测试 -> 决定 --fused -> 冒烟 (16 条) -> 全量 tp2_C64_v2 (2 验证器, 2 lane) -> 接线检查 -> 汇总
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
LOG=results/logs/vr2_pipeline.log; log(){ echo "$(date -Is) $*" | tee -a $LOG; }
while ! grep -q DONE results/logs/test_fused_row.log 2>/dev/null; do sleep 15; done
RES=$(grep RESULT results/logs/test_fused_row.log); log "fused test: $RES"
FUSED=""
if echo "$RES" | grep -q "(100.000%)" && echo "$RES" | grep -qE "idx device==cpu ([0-9]+)/\1 " ; then FUSED="--fused"; log "fused graph bit-exact & idx consistent -> using --fused"; else log "fused graph NOT adopted"; fi
run(){ local name=$1; shift; rm -rf results/vr/$name/ipc
  { echo "START_EPOCH $(date +%s)"; python -u verify_rollback.py run --run $name "$@"; echo "EXIT_CODE $? END_EPOCH $(date +%s)"; } > results/logs/vr_$name.log 2>&1
  local rc=$(grep -o "EXIT_CODE [0-9]*" results/logs/vr_$name.log | tail -1 | awk '{print $2}'); log "END $name rc=$rc"; return ${rc:-1}; }
log "START smoke_v2"; run smoke_v2 --tp 2 --lnc 2 --chunk 64 --n 16 --verifiers 2 --lanes 2 $FUSED || { log "smoke failed, stop"; exit 1; }
python - <<'PY' || { log "smoke stats check failed, stop"; exit 1; }
import json; s=json.load(open("results/vr/smoke_v2/stats.json"))
d=abs(s["accept_rate_empirical"]-s["accept_rate_theoretical"]); print("smoke accept emp/theo", s["accept_rate_empirical"], s["accept_rate_theoretical"], "fallbacks", s["fused_fallbacks"])
assert d < 0.005, d
PY
log "START tp2_C64_v2"; run tp2_C64_v2 --tp 2 --lnc 2 --chunk 64 --n 512 --verifiers 2 --lanes 2 $FUSED
log "START check"; python -u recompute.py --backend neuron --sample results/vr/tp2_C64_v2/final.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag vr_check_tp2_C64_v2 > results/logs/vr_check_rc_tp2_C64_v2.log 2>&1
python analyze.py --sample results/vr/tp2_C64_v2/final.jsonl --recompute results/recompute/vr_check_tp2_C64_v2.jsonl --out results/analysis/vr_check_tp2_C64_v2 --title "v2" > results/logs/vr_check_an_tp2_C64_v2.log 2>&1
python tools/vr_summary.py > results/vr/SUMMARY.md 2>&1; log "ALL_DONE"
