"""汇总 results/analysis/*/stats.json 成一张表, 并做确定性 / 截断 / span 可行性检查。
    python tools/summarize.py > results/analysis/SUMMARY.md
只读, 不改任何实验文件。"""
import glob, json, os, sys
import numpy as np

M = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(M)

def load_rows(p):
    rows, meta = [], None
    for l in open(p):
        o = json.loads(l)
        if "_meta" in o: meta = o["_meta"]; continue
        rows.append(o)
    return meta, rows

print("# Overnight summary\n")
print("## stats.json 汇总\n")
keys = ["n_tokens", "abs_diff_mean", "abs_diff_p50", "abs_diff_p99", "ratio_outside_tis_band", "kl_k3", "seq_log_ratio_std", "len_corr"]
print("| run | " + " | ".join(keys) + " |")
print("|---|" + "---|" * len(keys))
for d in sorted(glob.glob("results/analysis/*/stats.json")):
    s = json.load(open(d)); name = d.split("/")[2]
    print(f"| {name} | " + " | ".join(f"{s[k]:.4g}" if isinstance(s[k], float) else str(s[k]) for k in keys) + " |")

print("\n## 采样文件: 截断率 / 长度 / 耗时\n")
print("| sample | tp | lnc | max_new | rows | truncated | total_len p10/p50/p90/max | elapsed_s |")
print("|---|---|---|---|---|---|---|---|")
samples = {}
for p in sorted(glob.glob("results/samples/*.jsonl")):
    meta, rows = load_rows(p); samples[os.path.basename(p)[:-6]] = (meta, rows)
    L = sorted(len(r["token_ids"]) for r in rows)
    tr = sum(r["finish_reason"] != "stop" for r in rows)
    mx = max(len(r["logprobs"]) for r in rows)
    q = lambda f: L[min(len(L)-1, int(f*len(L)))]
    print(f"| {os.path.basename(p)} | {meta['tp']} | {meta['lnc']} | {mx} | {len(rows)} | {tr} ({100*tr/len(rows):.1f}%) | {q(.1)}/{q(.5)}/{q(.9)}/{L[-1]} | {meta['elapsed_s']:.0f} |")

print("\n## span 可行性 (bucket 1024, tol 0.08)\n")
for name, (meta, rows) in samples.items():
    L = [len(r["token_ids"]) for r in rows]
    feas = {sp: sum(sp*1024*0.92 <= x <= sp*1024*1.08 for x in L) for sp in (0.5, 0.75, 0.95)}
    print(f"- {name}: " + ", ".join(f"span {sp} → {n} 条" for sp, n in feas.items()))

def compare(pa, pb, label):
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"- {label}: 文件缺失, 跳过"); return
    _, A = load_rows(pa); _, B = load_rows(pb)
    A = {r["seq_id"]: r for r in A}; B = {r["seq_id"]: r for r in B}
    ids = sorted(set(A) & set(B))
    same_tok = sum(A[i]["token_ids"] == B[i]["token_ids"] for i in ids)
    same_lp = sum(A[i]["token_ids"] == B[i]["token_ids"] and A[i]["logprobs"] == B[i]["logprobs"] for i in ids)
    d = [abs(x - y) for i in ids if A[i]["token_ids"] == B[i]["token_ids"] for x, y in zip(A[i]["logprobs"], B[i]["logprobs"])]
    print(f"- {label}: token 序列相同 {same_tok}/{len(ids)}; 序列且 logprob 逐位相同 {same_lp}/{len(ids)}; "
          f"相同序列上 |Δlogp| mean {np.mean(d) if d else float('nan'):.2e} max {np.max(d) if d else float('nan'):.2e}")

print("\n## 确定性 (同 seed 重跑)\n")
compare("results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl", "results/samples/rep2_tp4_lnc2.jsonl", "sample NEFF tp4/lnc2 run1 vs run2")
compare("results/recompute/neuron_Qwen3-1.7B_b1024_sNone_lnc2_tp1.jsonl", "results/recompute/rep2_rc_lnc2_b1024.jsonl", "recompute XLA lnc2 run1 vs run2 (同一输入)")
print("\n## 跨配置: 同 seed 下采样轨迹是否相同\n")
compare("results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl", "results/samples/neuron_Qwen3-1.7B_tp4_lnc1_b1024.jsonl", "tp4: lnc2 vs lnc1")
compare("results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl", "results/samples/tp1_lnc2.jsonl", "lnc2: tp4 vs tp1")
compare("results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl", "results/samples/tp2_lnc2.jsonl", "lnc2: tp4 vs tp2")
compare("results/samples/neuron_Qwen3-1.7B_tp4_lnc1_b1024.jsonl", "results/samples/tp8_lnc1.jsonl", "lnc1: tp4 vs tp8")

print("\n## 各步骤耗时 (logs)\n")
for p in sorted(glob.glob("results/logs/*.log")):
    if "attempt" in p: continue
    t = open(p).read()
    import re
    s = re.search(r"START_EPOCH (\d+)", t); e = re.search(r"EXIT_CODE (\d+) END_EPOCH (\d+)", t)
    ncomp = len(re.findall(r"Compilation Successfully Completed", t))
    if s and e:
        print(f"- {os.path.basename(p)}: rc={e.group(1)}, wall {int(e.group(2))-int(s.group(1))}s, 编译 {ncomp} 张图")
    elif s:
        print(f"- {os.path.basename(p)}: 未结束")
