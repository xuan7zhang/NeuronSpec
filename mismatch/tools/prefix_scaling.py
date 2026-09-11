"""前缀长度扫描 + bootstrap 置信区间。只读现有 jsonl, 不改 analyze.py。
对每条序列取 completion 前 L 个 token, 算 seq_log_ratio_std(L); 拟合 log std ~ k log L。
k=0.5 是独立噪声的 CLT 基线, k>0.5 说明误差沿序列正相关。"""
import json, sys, numpy as np
rng = np.random.default_rng(0)

def load(p):
    d = {}
    for l in open(p):
        o = json.loads(l)
        if "_meta" not in o: d[o["seq_id"]] = o
    return d

def pair(s, r):
    A, B = load(f"results/samples/{s}.jsonl"), load(f"results/recompute/{r}.jsonl")
    ids = sorted(set(A) & set(B))
    return [(np.array(A[i]["logprobs"]), np.array(B[i]["logprobs"])) for i in ids]

def prefix_scan(seqs, Ls):
    Lmax = Ls[-1]
    keep = [(a, b) for a, b in seqs if len(a) >= Lmax]     # 同一批序列贯穿所有 L, 避免选择效应
    out = []
    for L in Ls:
        s = np.array([(b[:L] - a[:L]).sum() for a, b in keep])
        out.append((L, s.std(), len(keep)))
    k, c = np.polyfit(np.log([L for L, _, _ in out]), np.log([sd for _, sd, _ in out]), 1)
    return out, k

def boot_mean(seqs, n=2000):
    """序列级 bootstrap: token 在序列内相关, 按序列重采样才是合法的独立单元。"""
    per = np.array([np.abs(b - a).sum() for a, b in seqs]); cnt = np.array([len(a) for a, _ in seqs])
    idx = rng.integers(0, len(seqs), (n, len(seqs)))
    est = per[idx].sum(1) / cnt[idx].sum(1)
    return per.sum() / cnt.sum(), np.percentile(est, [2.5, 97.5])

runs = {
    "tp1_lnc2": ("tp1_lnc2", "rc_tp1_lnc2"),
    "tp2_lnc2": ("tp2_lnc2", "rc_tp2_lnc2"),
    "tp4_lnc2": ("neuron_Qwen3-1.7B_tp4_lnc2_b1024", "neuron_Qwen3-1.7B_b1024_sNone_lnc2_tp1"),
    "tp4_lnc1": ("neuron_Qwen3-1.7B_tp4_lnc1_b1024", "neuron_Qwen3-1.7B_b1024_sNone_lnc1_tp1"),
    "tp8_lnc1": ("tp8_lnc1", "rc_tp8_lnc1"),
    "maxnew868_tp4_lnc2": ("maxnew868_tp4_lnc2", "rc_maxnew868_lnc2"),
}
print("## abs_diff_mean, 序列级 bootstrap 95% CI\n")
data = {}
for name, (s, r) in runs.items():
    data[name] = pair(s, r)
    m, (lo, hi) = boot_mean(data[name])
    print(f"- {name}: {m:.5f}  [{lo:.5f}, {hi:.5f}]")

print("\n## 前缀扫描: seq_log_ratio_std(L) 与拟合指数 k  (k=0.5 = CLT 基线)\n")
for name, seqs in data.items():
    Ls = [64, 128, 256, 512] + ([868] if "868" in name else [])
    out, k = prefix_scan(seqs, Ls)
    print(f"- {name} (n={out[0][2]}): " + ", ".join(f"L={L}: {sd:.3f}" for L, sd, _ in out) + f"  → k = {k:.3f}")

print("\n## 同一批序列上, 相邻 token Δ 的自相关 (lag-1), 直接看误差是否沿序列相关\n")
for name, seqs in data.items():
    ac = []
    for a, b in seqs:
        d = b - a
        if len(d) > 64: ac.append(np.corrcoef(d[:-1], d[1:])[0, 1])
    print(f"- {name}: lag-1 autocorr mean {np.mean(ac):.4f}  (n={len(ac)})")
