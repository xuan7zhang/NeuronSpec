#!/bin/bash
# overnight 队列: 串行独占 Neuron 设备。每步幂等 (输出已存在则跳过), 失败不中断队列。
M=/home/ec2-user/mismatch; cd $M
source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
LOGDIR=$M/results/logs; mkdir -p $LOGDIR; OV=$LOGDIR/overnight.log
log(){ echo "$(date -Is) $*" | tee -a $OV; }
MODEL=Qwen/Qwen3-1.7B

run_step(){ local name=$1; shift; local lg=$LOGDIR/$name.log
  if [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -lt 5 ]; then log "ABORT $name: 根盘剩余 <5G"; return 99; fi
  log "START $name"
  { echo "START_EPOCH $(date +%s) $(date -Is)"; "$@"; echo "EXIT_CODE $? END_EPOCH $(date +%s) $(date -Is)"; } > $lg 2>&1
  local rc=$(grep -o "EXIT_CODE [0-9]*" $lg | tail -1 | awk '{print $2}'); log "END   $name rc=$rc"; return ${rc:-1}; }
sample(){ local tp=$1 lnc=$2 tag=$3; shift 3
  [ -s results/samples/$tag.jsonl ] && { log "SKIP sample $tag"; return 0; }
  run_step sample_$tag python -u sample.py --backend neuron --model $MODEL --tp $tp --lnc $lnc --tag $tag "$@"; }
recompute(){ local stag=$1 lnc=$2 bucket=$3 tag=$4
  [ -s results/recompute/$tag.jsonl ] && { log "SKIP recompute $tag"; return 0; }
  run_step recompute_$tag python -u recompute.py --backend neuron --sample results/samples/$stag.jsonl --bucket-len $bucket --lnc $lnc --tp 1 --batch-size 1 --tag $tag; }
analyze(){ local stag=$1 rtag=$2 out=$3 title=$4
  [ -s results/analysis/$out/stats.json ] && { log "SKIP analyze $out"; return 0; }
  run_step analyze_$out python analyze.py --sample results/samples/$stag.jsonl --recompute results/recompute/$rtag.jsonl --out results/analysis/$out --title "$title"; }

log "===== overnight start ====="
# 0. 等正在跑的 LNC=1 recompute 结束
while ! grep -q EXIT_CODE $LOGDIR/recompute_b1024_tp1_lnc1.log 2>/dev/null; do sleep 20; done
S2=neuron_Qwen3-1.7B_tp4_lnc2_b1024; S1=neuron_Qwen3-1.7B_tp4_lnc1_b1024
R2=neuron_Qwen3-1.7B_b1024_sNone_lnc2_tp1; R1=neuron_Qwen3-1.7B_b1024_sNone_lnc1_tp1

# 1. LNC=1 分析
analyze $S1 $R1 lnc1 "Qwen3-1.7B trn2 | NEFF tp4 lnc1 vs XLA tp1 lnc1 | bucket 1024"
# 2. bucket 2048 (同样本, 只换训练侧桶长)
recompute $S2 2 2048 rc_lnc2_b2048 && analyze $S2 rc_lnc2_b2048 b2048_lnc2 "NEFF tp4 lnc2 (b1024) vs XLA tp1 lnc2 bucket 2048"
recompute $S1 1 2048 rc_lnc1_b2048 && analyze $S1 rc_lnc1_b2048 b2048_lnc1 "NEFF tp4 lnc1 (b1024) vs XLA tp1 lnc1 bucket 2048"
# 3. TP 扫描
sample 1 2 tp1_lnc2 && recompute tp1_lnc2 2 1024 rc_tp1_lnc2 && analyze tp1_lnc2 rc_tp1_lnc2 tp1_lnc2 "NEFF tp1 lnc2 vs XLA tp1 lnc2 | bucket 1024"
sample 2 2 tp2_lnc2 && recompute tp2_lnc2 2 1024 rc_tp2_lnc2 && analyze tp2_lnc2 rc_tp2_lnc2 tp2_lnc2 "NEFF tp2 lnc2 vs XLA tp1 lnc2 | bucket 1024"
sample 8 1 tp8_lnc1 && recompute tp8_lnc1 1 1024 rc_tp8_lnc1 && analyze tp8_lnc1 rc_tp8_lnc1 tp8_lnc1 "NEFF tp8 lnc1 vs XLA tp1 lnc1 | bucket 1024"
# 4. 确定性: 同 seed 重跑
sample 4 2 rep2_tp4_lnc2
recompute $S2 2 1024 rep2_rc_lnc2_b1024
# 5. max_new 868: 模型会不会停 / span 可行性
sample 4 2 maxnew868_tp4_lnc2 --max-new 868 && recompute maxnew868_tp4_lnc2 2 1024 rc_maxnew868_lnc2 && analyze maxnew868_tp4_lnc2 rc_maxnew868_lnc2 maxnew868_lnc2 "NEFF tp4 lnc2 max_new 868 vs XLA tp1 lnc2 | bucket 1024"
# 6. 汇总
python tools/summarize.py > results/analysis/SUMMARY.md 2>&1; log "summary → results/analysis/SUMMARY.md"
log "===== ALL DONE ====="
