#!/bin/bash
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
LOG=results/logs/vr2_more.log; log(){ echo "$(date -Is) $*" | tee -a $LOG; }
run(){ local name=$1; shift; [ -s results/vr/$name/stats.json ] && { log "SKIP $name"; return 0; }; rm -rf results/vr/$name/ipc
  log "START $name"; { echo "START_EPOCH $(date +%s)"; python -u verify_rollback.py run --run $name "$@"; echo "EXIT_CODE $? END_EPOCH $(date +%s)"; } > results/logs/vr_$name.log 2>&1
  log "END $name rc=$(grep -o 'EXIT_CODE [0-9]*' results/logs/vr_$name.log | tail -1 | awk '{print $2}')"; }
check(){ local name=$1; [ -s results/analysis/vr_check_$name/stats.json ] && return 0; [ -s results/vr/$name/final.jsonl ] || return 0
  python -u recompute.py --backend neuron --sample results/vr/$name/final.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag vr_check_$name > results/logs/vr_check_rc_$name.log 2>&1
  python analyze.py --sample results/vr/$name/final.jsonl --recompute results/recompute/vr_check_$name.jsonl --out results/analysis/vr_check_$name --title "$name" > results/logs/vr_check_an_$name.log 2>&1; log "check $name done"; }
run tp1_C64_v2  --tp 1 --lnc 2 --chunk 64  --n 512 --verifiers 3 --lanes 2 --fused; check tp1_C64_v2
run tp2_C32_v2  --tp 2 --lnc 2 --chunk 32  --n 512 --verifiers 2 --lanes 2 --fused; check tp2_C32_v2
run tp2_C128_v2 --tp 2 --lnc 2 --chunk 128 --n 512 --verifiers 2 --lanes 2 --fused; check tp2_C128_v2
python tools/vr_summary.py > results/vr/SUMMARY.md 2>&1; log "ALL_DONE"
