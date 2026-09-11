"""3c: recompute.py --logits-positions 输出的整行 log_softmax 是否与默认路径的标量逐位一致 (换图后的数值验证)。"""
import json, numpy as np, subprocess, sys, os
os.chdir("/home/ec2-user/mismatch")
S = "results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"; R = "results/recompute/neuron_Qwen3-1.7B_b1024_sNone_lnc2_tp1.jsonl"
orig = {}; ref = {}
for p, d in ((S, orig), (R, ref)):
    for l in open(p):
        o = json.loads(l)
        if "_meta" not in o: d[o["seq_id"]] = o
ids = sorted(orig)[:8]
pos = {i: [0, 1, 100, 300, 500] if i != ids[0] else [0, 250] for i in ids}     # 不等长, 测 padding 逻辑
json.dump(pos, open("results/recompute/rows_test.pos.json", "w"))
cmd = [sys.executable, "-u", "recompute.py", "--backend", "neuron", "--sample", S, "--bucket-len", "1024", "--lnc", "2", "--tp", "1",
       "--batch-size", "1", "--logits-positions", "results/recompute/rows_test.pos.json", "--tag", "rows_test"]
rc = subprocess.run(cmd, stdout=open("results/logs/recompute_rows_test.log", "w"), stderr=subprocess.STDOUT).returncode
print("recompute rc", rc)
if rc: sys.exit(rc)
new = {}
for l in open("results/recompute/rows_test.jsonl"):
    o = json.loads(l)
    if "_meta" not in o: new[o["seq_id"]] = o
z = np.load("results/recompute/rows_test.rows.npz")
# (1) 换图后的标量路径 vs 默认图的标量 (原 recompute jsonl)
d_scalar = np.concatenate([np.abs(np.array(new[i]["logprobs"]) - np.array(ref[i]["logprobs"])) for i in ids])
print(f"新图标量 vs 默认图标量: 逐位相同 {(d_scalar == 0).mean()*100:.2f}%  max|Δ| {d_scalar.max():.3e}")
# (2) 行[token] vs 同一次跑的标量;  (3) 行[token] vs 默认图标量;  (4) 行归一化
d_row_same, d_row_ref, lse = [], [], []
for k in range(len(z["pos"])):
    i, q = str(z["seq_ids"][k]), int(z["pos"][k]); row = z["rows"][k].astype(np.float64)
    tok = orig[i]["token_ids"][orig[i]["prompt_len"] + q]
    d_row_same.append(abs(row[tok] - new[i]["logprobs"][q])); d_row_ref.append(abs(row[tok] - ref[i]["logprobs"][q]))
    lse.append(np.log(np.exp(row).sum()))
print(f"行[token] vs 同跑标量: max|Δ| {max(d_row_same):.3e} (n={len(d_row_same)}) | vs 默认图标量: max|Δ| {max(d_row_ref):.3e}")
print(f"行的 logsumexp: mean {np.mean(lse):.2e} max|.| {np.max(np.abs(lse)):.2e}  (应≈0)")
print("行数/位置:", z["rows"].shape, list(zip(z["seq_ids"][:7], z["pos"][:7])))
