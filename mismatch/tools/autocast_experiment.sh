#!/bin/bash
# 训练侧编译 flag 对齐 NxDI (--auto-cast=none): 同一份样本重算, 看失配是否变化. 排在 8B 之后.
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
while ! grep -q ALL_DONE results/logs/run_8b.log 2>/dev/null; do sleep 60; done
S=results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl
MISMATCH_CC_EXTRA="--auto-cast=none" NEURON_COMPILE_CACHE_URL=/scratch/neuron-cache/train-autocast-none \
  python -u recompute.py --backend neuron --sample $S --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag rc_autocast_none_lnc2 > results/logs/recompute_autocast_none.log 2>&1
python analyze.py --sample $S --recompute results/recompute/rc_autocast_none_lnc2.jsonl --out results/analysis/autocast_none_tp4_lnc2 --title "train-side --auto-cast=none" > results/logs/analyze_autocast_none.log 2>&1
echo "AUTOCAST_DONE $(date -Is)" >> results/logs/run_8b.log
