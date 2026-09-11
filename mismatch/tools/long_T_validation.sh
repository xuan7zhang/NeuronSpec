#!/bin/bash
# 直接验证 σ ∝ √T: T≈1900 (桶 2048), 与 T=511/868 的两点连成三点
M=/home/ec2-user/mismatch; cd $M; source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
export TMPDIR=/scratch/ncc-work
LOG=results/logs/long_T.log; log(){ echo "$(date -Is) $*" | tee -a $LOG; }
log "START sample T=1900 bucket 2048"
[ -s results/samples/longT_tp4_lnc2.jsonl ] || NEURON_COMPILE_CACHE_URL=/scratch/neuron-cache/infer \
  python -u sample.py --backend neuron --tp 4 --lnc 2 --bucket-len 2048 --max-new 1900 --tag longT_tp4_lnc2 > results/logs/sample_longT.log 2>&1
log "END sample rc=$? ($(grep -c 'Compilation Successfully' results/logs/sample_longT.log) NEFF)"
[ -s results/samples/longT_tp4_lnc2.jsonl ] || exit 1
log "START recompute bucket 2048"
NEURON_COMPILE_CACHE_URL=/scratch/neuron-cache/train python -u recompute.py --backend neuron \
  --sample results/samples/longT_tp4_lnc2.jsonl --bucket-len 2048 --lnc 2 --tp 1 --batch-size 1 --tag rc_longT_lnc2 > results/logs/recompute_longT.log 2>&1
log "END recompute rc=$?"
python analyze.py --sample results/samples/longT_tp4_lnc2.jsonl --recompute results/recompute/rc_longT_lnc2.jsonl --out results/analysis/longT_tp4_lnc2 --title "T=1900" > /dev/null 2>&1
python - <<'PY' 2>&1 | tee -a $LOG
import json, numpy as np
def load(p):
    d={}
    for l in open(p):
        o=json.loads(l)
        if "_meta" not in o: d[o["seq_id"]]=o
    return d
pts=[]
for tag,(s,r) in {"T=511":("neuron_Qwen3-1.7B_tp4_lnc2_b1024","neuron_Qwen3-1.7B_b1024_sNone_lnc2_tp1"),
                  "T=868":("maxnew868_tp4_lnc2","rc_maxnew868_lnc2"),
                  "T=long":("longT_tp4_lnc2","rc_longT_lnc2")}.items():
    A=load(f"results/samples/{s}.jsonl"); B=load(f"results/recompute/{r}.jsonl"); k=sorted(set(A)&set(B))
    S=np.array([(np.array(B[i]["logprobs"])-np.array(A[i]["logprobs"])).sum() for i in k])
    T=np.mean([len(A[i]["logprobs"]) for i in k]); W=np.exp(S)
    pts.append((T,S.std(),(W.sum()**2)/(len(W)*(W**2).sum())))
    print(f"RESULT {tag}: T={T:.0f}, sigma_seq={S.std():.3f}, ESS/N={(W.sum()**2)/(len(W)*(W**2).sum()):.3f}, exp(-s^2)={np.exp(-S.std()**2):.3f}")
T0,s0,_=pts[0]
for T,s,e in pts: print(f"  预测 sigma(T={T:.0f}) = {s0*np.sqrt(T/T0):.3f} (由 T={T0:.0f} 外推), 实测 {s:.3f}, 偏差 {100*(s/(s0*np.sqrt(T/T0))-1):+.1f}%")
sl=np.polyfit(np.log([p[0] for p in pts]),np.log([p[1] for p in pts]),1)[0]
print(f"  三点幂律拟合 sigma ~ T^{sl:.3f}  (理论 0.5, iid 误差)")
PY
log "ALL DONE"
