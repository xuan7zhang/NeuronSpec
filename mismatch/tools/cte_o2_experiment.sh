#!/bin/bash
# CTE 图 -O1 -> -O2 实验: 等探针结束后独占设备跑 sample -> recompute -> analyze
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
while ! grep -q ALL_DONE results/logs/probe_batched_verifier.log 2>/dev/null; do sleep 20; done
echo "START_EPOCH $(date +%s) $(date -Is)"
MISMATCH_CTE_OPT=2 PYTHONPATH=$M/hooks python -u sample.py --backend neuron --model Qwen/Qwen3-1.7B --tp 4 --lnc 2 --tag cte_O2_tp4_lnc2 2>&1 | grep -E "mismatch_cte_hook|compiler_args are|Compilation Successfully|\[done\]|\[warn\]|Traceback|Error:" | cut -c1-400
echo "SAMPLE_EXIT ${PIPESTATUS[0]} $(date -Is)"
python -u recompute.py --backend neuron --sample results/samples/cte_O2_tp4_lnc2.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1 --tag rc_cte_O2_lnc2 2>&1 | grep -E "\[done\]|Traceback" | cut -c1-200
python analyze.py --sample results/samples/cte_O2_tp4_lnc2.jsonl --recompute results/recompute/rc_cte_O2_lnc2.jsonl --out results/analysis/cte_O2_tp4_lnc2 --title "NEFF tp4 lnc2 CTE=-O2 vs XLA tp1 lnc2 | bucket 1024" 2>&1 | grep -E "abs_diff_mean|ratio_outside|kl_k3|→"
echo "EXIT_CODE 0 END_EPOCH $(date +%s) $(date -Is)"
