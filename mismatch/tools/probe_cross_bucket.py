"""分桶验证器可行性: 同一前缀在桶 512 / 1024 / 2048 下 prefill, 重叠位置是否逐位相同?
逐位相同 -> 验证器可按前缀长度选最小桶, prefill 成本随之下降, 且目标分布不变。"""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/ec2-user/mismatch")
import recompute as R
R.setup_neuron_env(lnc=2, cores=None)
import torch, torch_xla.core.xla_model as xm
from transformers import AutoTokenizer
S = "results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"
rows = [json.loads(l) for l in open(S)][1:11]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model, dev = R.build_model("neuron", "Qwen/Qwen3-1.7B", 1); model.eval()
def prefill(tids, B):
    ids, attn = R.pad_batch([{"token_ids": tids}], B, pad)
    with torch.no_grad():
        lp = R.forward_scalar(model, ids.to(dev), attn.to(dev)); xm.mark_step()
    return lp[0].float().cpu().numpy()
BUCKETS = [512, 1024, 2048]
t0 = time.time(); agg = {}
for r in rows:
    tids = r["token_ids"][:400]                      # 同一前缀, 所有桶都放得下
    out = {B: prefill(tids, B) for B in BUCKETS}
    n = len(tids) - 1
    for B in BUCKETS[:-1]:
        a, b = out[B][:n], out[1024][:n] if B != 1024 else out[2048][:n]
        pass
    for i, B1 in enumerate(BUCKETS):
        for B2 in BUCKETS[i+1:]:
            a, b = out[B1][:n], out[B2][:n]; d = np.abs(a - b)
            k = f"{B1}vs{B2}"; agg.setdefault(k, [0, 0, 0.0])
            agg[k][0] += int((a == b).sum()); agg[k][1] += n; agg[k][2] = max(agg[k][2], float(d.max()))
print(f"[done] {len(rows)} seqs x {len(BUCKETS)} buckets in {time.time()-t0:.0f}s")
for k, (eq, n, mx) in agg.items():
    print(f"RESULT bucket {k}: bitwise equal {eq}/{n} ({100*eq/n:.3f}%), max|Δ| {mx:.3e}")
