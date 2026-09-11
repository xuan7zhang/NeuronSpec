"""3b: 前缀含已生成 token 时 Neuron 后端行为是否正常。
取 8 条已有序列, 前缀 = prompt + completion[:128], 续 decode 64 步, 记 top-256。
检查: prompt_len 对齐; logprob 有限; 原序列在位置 128 采到的 token 出现在前缀跑的 top-256 里,
且其 logprob 与原采样时的 logprob 接近 (差异 = CTE 图 vs TKG 图, 预期 ~0.01-0.05)。"""
import json, numpy as np, subprocess, sys, os
os.chdir("/home/ec2-user/mismatch")
S = "results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"
orig = {}
for l in open(S):
    o = json.loads(l)
    if "_meta" not in o: orig[o["seq_id"]] = o
ids = sorted(orig)[:8]; K = 128
with open("results/samples/prefix_test.in.jsonl", "w") as f:
    for i in ids:
        r = orig[i]; f.write(json.dumps({"seq_id": i, "token_ids": r["token_ids"][: r["prompt_len"] + K]}) + "\n")
cmd = [sys.executable, "-u", "sample.py", "--backend", "neuron", "--tp", "4", "--lnc", "2", "--max-new", "64",
       "--topk-logprobs", "256", "--prefix-file", "results/samples/prefix_test.in.jsonl", "--tag", "prefix_test"]
rc = subprocess.run(cmd, stdout=open("results/logs/sample_prefix_test.log", "w"), stderr=subprocess.STDOUT).returncode
print("sample rc", rc)
if rc: sys.exit(rc)
out = {}
for l in open("results/samples/prefix_test.jsonl"):
    o = json.loads(l)
    if "_meta" not in o: out[o["seq_id"]] = o
z = np.load("results/samples/prefix_test.top256.npz")
seq_ids = list(z["seq_ids"])
ok_align = all(out[i]["prompt_len"] == orig[i]["prompt_len"] + K and out[i]["token_ids"][: out[i]["prompt_len"]] == orig[i]["token_ids"][: out[i]["prompt_len"]] for i in ids)
finite = all(np.isfinite(out[i]["logprobs"]).all() and len(out[i]["logprobs"]) > 0 for i in ids)
print("prompt_len/prefix 对齐:", ok_align, "| logprobs 有限非空:", finite,
      "| 每条生成 token 数:", [len(out[i]["logprobs"]) for i in ids])
# 位置 K 的分布: 原采样 token 是否在前缀跑的 top-256 里, logprob 差多少
res = []
for i in ids:
    si = seq_ids.index(i); m = (z["seq_idx"] == si) & (z["pos"] == 0)
    row_ids, row_lp = z["ids"][m][0], z["logprobs"][m][0]
    t_orig, q_orig = orig[i]["token_ids"][orig[i]["prompt_len"] + K], orig[i]["logprobs"][K]
    hit = np.where(row_ids == t_orig)[0]
    res.append((i, t_orig, q_orig, float(row_lp[hit[0]]) if len(hit) else None, int(hit[0]) if len(hit) else -1,
                float(np.log(np.exp(row_lp[row_lp > -np.inf]).sum()))))
print("seq_id | 原 token | 原 logprob(TKG) | 前缀跑 logprob(CTE) | 在 top256 的名次 | top256 总质量(log)")
for r in res: print("  ", r)
d = [abs(r[2] - r[3]) for r in res if r[3] is not None]
print(f"命中 top-256: {sum(r[3] is not None for r in res)}/8 | |Δ| mean {np.mean(d):.4f} max {np.max(d):.4f} | "
      f"top-256 质量覆盖 mean {np.mean([np.exp(r[5]) for r in res]):.4f} min {np.min([np.exp(r[5]) for r in res]):.4f}")
# 采样 token 列 0 与 jsonl logprobs 逐位相同?
same = all(np.allclose(z["logprobs"][z["seq_idx"] == seq_ids.index(i)][:, 0], out[i]["logprobs"]) for i in ids)
print("npz 列0 == jsonl logprobs:", same)
