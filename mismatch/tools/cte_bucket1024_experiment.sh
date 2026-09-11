#!/bin/bash
# CTE 桶强制为 1024 (与训练图相同): 序列头部尖峰是否消失。等 nokernel 实验结束后跑。
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
while ! grep -q EXIT_CODE results/logs/cte_nokernel_experiment.log 2>/dev/null; do sleep 20; done
echo "START_EPOCH $(date +%s) $(date -Is)"
python -u sample.py --backend neuron --model Qwen/Qwen3-1.7B --tp 4 --lnc 2 --tag cte_b1024_tp4_lnc2 --neuron-override '{"context_encoding_buckets": [1024]}' 2>&1 | grep -E "override_neuron_config|bucket|Compilation Successfully|\[done\]|Traceback|Error:" | cut -c1-300
echo "SAMPLE_EXIT ${PIPESTATUS[0]} $(date -Is)"
python -u recompute.py --backend neuron --sample results/samples/cte_b1024_tp4_lnc2.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag rc_cte_b1024_lnc2 2>&1 | grep -E "\[done\]|Traceback" | cut -c1-200
python analyze.py --sample results/samples/cte_b1024_tp4_lnc2.jsonl --recompute results/recompute/rc_cte_b1024_lnc2.jsonl --out results/analysis/cte_b1024_tp4_lnc2 --title "NEFF tp4 lnc2 CTE bucket 1024 vs XLA tp1 lnc2" 2>&1 | grep -E "abs_diff_mean|ratio_outside|kl_k3|→"
echo "EXIT_CODE 0 END_EPOCH $(date +%s) $(date -Is)"
