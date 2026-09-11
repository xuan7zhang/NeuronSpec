#!/bin/bash
# overnight 队列 (验证-回滚): 串行独占设备。每步幂等, 失败不中断。
M=/home/ec2-user/mismatch; cd $M
source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
LOGDIR=$M/results/logs; OV=$LOGDIR/overnight_vr.log
log(){ echo "$(date -Is) $*" | tee -a $OV; }
step(){ local name=$1; shift
  if [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -lt 5 ]; then log "ABORT $name: 根盘 <5G"; return 99; fi
  log "START $name"
  { echo "START_EPOCH $(date +%s)"; "$@"; echo "EXIT_CODE $? END_EPOCH $(date +%s)"; } > $LOGDIR/$name.log 2>&1
  local rc=$(grep -o "EXIT_CODE [0-9]*" $LOGDIR/$name.log | tail -1 | awk '{print $2}'); log "END   $name rc=$rc"; return ${rc:-1}; }
vr(){ local run=$1 tp=$2 chunk=$3
  [ -s results/vr/$run/stats.json ] && { log "SKIP vr $run"; return 0; }
  rm -rf results/vr/$run/ipc
  step vr_$run python -u verify_rollback.py run --run $run --tp $tp --lnc 2 --chunk $chunk --n 512; }
check(){ local run=$1
  [ -s results/analysis/vr_check_$run/stats.json ] && { log "SKIP check $run"; return 0; }
  [ -s results/vr/$run/final.jsonl ] || { log "SKIP check $run (no final)"; return 0; }
  step vr_check_rc_$run python -u recompute.py --backend neuron --sample results/vr/$run/final.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag vr_check_$run \
  && step vr_check_an_$run python analyze.py --sample results/vr/$run/final.jsonl --recompute results/recompute/vr_check_$run.jsonl --out results/analysis/vr_check_$run --title "verify-rollback $run: p_t vs recompute (must be 0)"; }

log "===== overnight_vr start ====="
vr tp2_CT   2 0   ; check tp2_CT
vr tp2_C64  2 64  ; check tp2_C64
vr tp1_C64  1 64  ; check tp1_C64
vr tp2_C128 2 128 ; check tp2_C128
python tools/vr_summary.py > results/vr/SUMMARY.md 2>&1; log "summary → results/vr/SUMMARY.md"
log "===== ALL DONE ====="
