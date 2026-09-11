"""汇总 results/vr/*/stats.json + 接线检查 + overhead(λ) 曲线 (规格 §7-6)。
    python tools/vr_summary.py > results/vr/SUMMARY.md
纯 CPU。曲线: §4 模型 (实测 prefill_cost) 的解析曲线 + 各 run 的经验点。"""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import rejection_analysis as RA

runs = {}
for p in sorted(glob.glob("results/vr/*/stats.json")):
    name = p.split("/")[2]
    if name.startswith("smoke"): continue
    r = json.load(open(p))
    r.setdefault("total_overhead_wallclock", r["wall_s"] / (r["decode_busy_s"] * r["accepted_tokens_total"] / r["decode_tokens_total"]) - 1)
    r.setdefault("prefills_per_seq_scalar", r.get("prefills_scalar", 0) / r["n_sequences"]); r.setdefault("prefills_per_seq_row", r.get("prefills_row", 0) / r["n_sequences"])
    runs[name] = r

print("# 验证-回滚 overnight 汇总\n")
print("## 各 run (regulation §4/§6)\n")
cols = ["tp", "chunk", "n_sequences", "rounds", "tested_tokens", "accept_rate_empirical", "accept_rate_theoretical",
        "lambda_empirical", "rejections_per_seq", "seqs_zero_rejection", "decode_overhead",
        "prefills_per_seq_scalar", "prefills_per_seq_row", "prefill_overhead_measured_wall", "total_overhead_measured_wall", "total_overhead_wallclock",
        "resid_mass_outside_top256_mean", "tv_at_rejections_mean", "a_new_in_top256_frac", "wall_s"]
print("| run | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
for n, s in runs.items():
    print(f"| {n} | " + " | ".join(f"{s[c]:.4g}" if isinstance(s[c], float) else str(s[c]) for c in cols) + " |")
print("\nq 在 top-256 外的质量 (拒绝位置):")
for n, s in runs.items():
    print(f"- {n}: {s['q_out_mass_at_rejections']}")

print("\n## 接线检查 (§6): final.jsonl 用原版 recompute.py 重算, Δ 应恒为 0\n")
for n in runs:
    p = f"results/analysis/vr_check_{n}/stats.json"
    if os.path.exists(p):
        a = json.load(open(p))
        print(f"- {n}: abs_diff_max = {a['abs_diff_max']:.3e}, abs_diff_mean = {a['abs_diff_mean']:.3e}, n_tokens = {a['n_tokens']}  "
              + ("✓ 接线正确" if a["abs_diff_max"] == 0.0 else "✗ 非零!"))
    else:
        print(f"- {n}: 未做")

print("\n## 与基线 λ 的对照 (rejection_analysis.py 在纯采样数据上的估计)\n")
base = {"tp1": ("tp1_lnc2", "rc_tp1_lnc2"), "tp2": ("tp2_lnc2", "rc_tp2_lnc2")}
lam_base = {}
for k, (s, r) in base.items():
    st, _, _, _ = RA.run(f"results/samples/{s}.jsonl", f"results/recompute/{r}.jsonl", 0.039)
    lam_base[k] = st["lambda_per_token"]
    print(f"- {k}: λ_baseline = {st['lambda_per_token']:.5f}, E[rej]/seq = {st['expected_rejections_per_seq']:.2f}")

# ---- overhead(λ) 曲线 ----
# §4 模型在 Trainium 上要改两个常数 (由实测反推):
#   (1) XLA 静态桶: 每次验证 prefill 都是整个 bucket 的成本, len_frac = 1.0 (不是 0.5)
#   (2) 本实现拒绝时多一次行 prefill 取 p_t 整行: prefill 次数 = T/C + 2λT (不是 T/C + λT)
#   (3) prefill_cost 用各 run 实测: 0.120 s / (decode busy 时间 / 接受的序列数)
def model_overhead(lam, T, C, pc, trainium=True):
    n_rej = lam * T; c_eff = T if C is None else C
    dec = n_rej * c_eff / 2 / T
    n_pre = (1 if C is None else T / c_eff) + (2 if trainium else 1) * n_rej
    return dec + n_pre * (1.0 if trainium else 0.5) * pc
print("\n## §4 模型 vs 实测 (修正常数: len_frac=1, prefill 次数 T/C+2λT, prefill_cost 按 run 实测)\n")
print("| run | λ | C | prefill_cost 实测 | decode 开销 实测/模型 | prefill 开销 实测/模型 | 总开销 实测/模型 | 原 §4 模型 |")
print("|---|---|---|---|---|---|---|---|")
pcs = {}
for n, s in runs.items():
    T = s["T"]; C = None if s["chunk"] == T else s["chunk"]; lam = s["lambda_empirical"]
    pc = 0.120 / (s["decode_busy_s"] * s["accepted_tokens_total"] / s["decode_tokens_total"] / s["n_sequences"]); pcs[n] = pc
    m_dec = lam * T * (T if C is None else C) / 2 / T
    m_pre = ((1 if C is None else T / (C or T)) + 2 * lam * T) * pc
    print(f"| {n} | {lam:.5f} | {s['chunk']} | {pc:.4f} | {s['decode_overhead']:.3f} / {m_dec:.3f} | {s['prefill_overhead_measured_wall']:.3f} / {m_pre:.3f} | "
          f"{s['total_overhead_measured_wall']:.3f} / {m_dec + m_pre:.3f} | {model_overhead(lam, T, C, 0.039, trainium=False):.3f} |")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = 512; pc = float(np.mean([pcs[n] for n in runs if n.startswith("tp2")]))
    lams = np.linspace(0.0005, 0.02, 120)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for Cc, col in ((32, "#4C72B0"), (64, "#55A868"), (128, "#DD8452"), (None, "#8172B2")):
        ax.plot(lams, [model_overhead(l, T, Cc, pc) for l in lams], color=col, lw=1.6, label=f"model (trn2 const.), C={'T' if Cc is None else Cc}")
        ax.plot(lams, [model_overhead(l, T, Cc, 0.039, trainium=False) for l in lams], color=col, lw=1, ls=":", alpha=0.7)
    mk = {64: "o", 128: "^", 512: "s"}
    for n, s in runs.items():
        ax.scatter([s["lambda_empirical"]], [s["total_overhead_measured_wall"]], s=52, zorder=5, marker=mk.get(s["chunk"], "o"),
                   edgecolor="k", label=f"measured {n}")
    ax.axhspan(0.56, 1.35, color="grey", alpha=0.12, label="TBIK 56–135%")
    ax.axhline(0.34, color="grey", ls="--", lw=1, label="SGLang deterministic ~34%")
    ax.set_xlabel("λ  (per-token rejection prob = mean max(0, 1−p/q))"); ax.set_ylabel("overhead vs plain decode")
    ax.set_ylim(0, 1.6); ax.set_xlim(0, 0.02)
    ax.set_title(f"Verify-rollback overhead vs mismatch (Qwen3-1.7B, trn2)\nsolid: model w/ trn2 constants (prefill_cost={pc:.3f}, full-bucket prefill, +row prefill); dotted: §4 as written", fontsize=8)
    ax.legend(fontsize=6.5, ncol=2)
    # 右图: 端到端 wall-clock 口径 (流水线 + DP 验证器把验证藏进 decode)
    for n, s in runs.items():
        ax2.scatter([s["lambda_empirical"]], [s["total_overhead_wallclock"]], s=52, zorder=5, marker=mk.get(s["chunk"], "o"),
                    edgecolor="k", color=("#C44E52" if s.get("fused") else "#8C8C8C"), label=n)
    ax2.axhspan(0.56, 1.35, color="grey", alpha=0.12); ax2.axhline(0.34, color="grey", ls="--", lw=1)
    ax2.set_ylim(0, 1.6); ax2.set_xlim(0, 0.02); ax2.set_xlabel("λ"); ax2.set_ylabel("end-to-end wall-clock overhead")
    ax2.set_title("Wall-clock overhead: v1 (grey, serial, 1 verifier) vs v2 (red, pipelined, DP verifiers, fused graph)", fontsize=8)
    ax2.legend(fontsize=6.5, ncol=2); fig.tight_layout()
    fig.savefig("results/vr/overhead_vs_lambda.png", dpi=150)
    print("\n图: results/vr/overhead_vs_lambda.png")
except Exception as e:
    print("绘图失败:", e)
