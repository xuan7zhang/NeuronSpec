"""H_prefix 测试: 训练图对同一序列的前 256 / 512 / 全长分别 prefill (同一桶长 1024),
重叠位置的 logprob 是否逐位相同。逐位相同 -> 可以分块验证; 否则只能采完再验。"""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/ec2-user/mismatch")
import recompute as R                      # 复用 recompute.py 的训练图路径, 不另起炉灶
R.setup_neuron_env(lnc=2, cores=None)      # 必须在 import torch_xla 之前
import torch
from transformers import AutoTokenizer

SAMPLE = "/home/ec2-user/mismatch/results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"
BUCKET, N, PREFIXES = 1024, 10, (256, 512, None)
rows, meta = R.load_sample(SAMPLE); rows = rows[:N]
tok = AutoTokenizer.from_pretrained(meta["model"])
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model, device = R.build_model("neuron", meta["model"], 1); model.eval()
import torch_xla.core.xla_model as xm

def prefill(token_ids):
    ids, attn = R.pad_batch([{"token_ids": token_ids}], BUCKET, pad_id)
    with torch.no_grad():
        lp = R.gather_logprobs(model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits, ids.to(device))
        xm.mark_step()
    return lp[0].float().cpu().numpy()     # [BUCKET-1], 下标 t = 给定前缀<=t 预测 token t+1

report = []
t0 = time.time()
for r in rows:
    full = r["token_ids"]; L = len(full)
    lps = {P: prefill(full[:P] if P else full) for P in PREFIXES}
    ref = lps[None]
    for P in (256, 512):
        k = P - 1                           # 前缀 P 个 token 的合法 lp 下标 0..P-2
        a, b = lps[P][:k], ref[:k]
        eq = int((a == b).sum()); d = np.abs(a - b)
        report.append({"seq_id": r["seq_id"], "len": L, "prefix": P, "n_overlap": k,
                       "bitwise_equal": eq, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                       "n_diff_gt_1e-6": int((d > 1e-6).sum()), "n_diff_gt_1e-3": int((d > 1e-3).sum())})
    # 额外: 512 前缀 vs 256 前缀 的重叠段 (两者都不是全长)
    a, b = lps[256][:255], lps[512][:255]; d = np.abs(a - b)
    report.append({"seq_id": r["seq_id"], "len": L, "prefix": "256_vs_512", "n_overlap": 255,
                   "bitwise_equal": int((a == b).sum()), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                   "n_diff_gt_1e-6": int((d > 1e-6).sum()), "n_diff_gt_1e-3": int((d > 1e-3).sum())})
print(f"[done] {len(rows)} seqs x {len(PREFIXES)} prefills in {time.time()-t0:.1f}s")
json.dump(report, open("/home/ec2-user/mismatch/results/analysis/h_prefix.json", "w"), indent=1)
for key in (256, 512, "256_vs_512"):
    sub = [x for x in report if x["prefix"] == key]
    tot = sum(x["n_overlap"] for x in sub); eq = sum(x["bitwise_equal"] for x in sub)
    print(f"prefix={key}: overlap {tot} positions, bitwise equal {eq} ({100*eq/tot:.2f}%), "
          f"max|Δ| {max(x['max_abs_diff'] for x in sub):.3e}, mean|Δ| {np.mean([x['mean_abs_diff'] for x in sub]):.3e}, "
          f">1e-6: {sum(x['n_diff_gt_1e-6'] for x in sub)}, >1e-3: {sum(x['n_diff_gt_1e-3'] for x in sub)}")
