"""
诊断: 在已有结果上排除 top-k 重归一化 artifact, 并给出误差棒。

    python diagnose.py --sample <s.jsonl> --recompute <r.jsonl>
    python diagnose.py --selftest

纯 CPU, 不需要重跑实验。回答四个问题:
  1. 失配是单侧的还是对称的?      -> 区分 artifact 和真实数值差异
  2. 重尾集中在低概率 token 吗?    -> 同上, 且指出该看哪些位置
  3. 配置间的差异超出抽样噪声吗?    -> 现有比较全都没有误差棒
  4. 长度相关性为什么是 0?         -> 可能只是长度没有方差
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from analyze import load_pair


def sign_asymmetry(d: np.ndarray) -> dict:
    """*** 最关键的判别式 ***

    top-k 重归一化: 采样侧在截断分布上归一, logprob 系统性偏高,
                    Δ = logp_recompute - logp_sample = log(M) <= 0, 单侧。
    kernel 数值差异: 归约顺序不同导致的舍入, 关于 0 对称。

    两者的 frac_negative 期望值分别是 ~1.0 和 ~0.5。中间值说明两者都有。
    """
    n = len(d)
    frac_neg = float((d < 0).mean())
    nonzero = d[d != 0]
    return {
        "frac_negative": frac_neg,
        "frac_negative_nonzero": float((nonzero < 0).mean()) if len(nonzero) else float("nan"),
        "signed_mean": float(d.mean()),
        "signed_median": float(np.median(d)),
        # |signed_mean| / mean(|d|): 接近 1 = 完全单侧, 接近 0 = 完全对称
        "onesidedness": float(abs(d.mean()) / np.abs(d).mean()) if np.abs(d).mean() > 0 else 0.0,
        "n": n,
    }


def topk_renorm_prediction(d: np.ndarray, k: int = 256, vocab: int = 151936) -> dict:
    """把观测到的 Δ 反解成"被截断掉的概率质量", 看是否落在合理区间。

    若 artifact 成立, M = exp(d) 应该全部落在 (0, 1], 且分布集中在接近 1 处
    并带一条向下的尾巴。若有相当比例的 M > 1, artifact 解释不了那部分。
    """
    M = np.exp(d)
    return {
        "implied_topk_mass_p50": float(np.percentile(M, 50)),
        "implied_topk_mass_p01": float(np.percentile(M, 1)),
        "frac_mass_above_1": float((M > 1).mean()),   # artifact 无法解释的部分
        "note": f"若为 top-{k}/{vocab} 重归一化, 全部质量应 <=1",
    }


def tail_vs_confidence(lp_sample: np.ndarray, d: np.ndarray) -> dict:
    """重尾是不是集中在模型不确定的位置?

    低概率 token (logp 很负) 处, 截断掉的质量更大, artifact 更明显。
    kernel 差异则没有理由和 token 概率相关。
    """
    absd = np.abs(d)
    order = np.argsort(lp_sample)          # 从最不可能到最可能
    n = len(order)
    q = n // 5
    out = {}
    for i, name in enumerate(["p_lowest20", "p_20_40", "p_40_60", "p_60_80", "p_highest20"]):
        seg = order[i * q : (i + 1) * q] if i < 4 else order[4 * q :]
        out[name] = float(absd[seg].mean())
    lo, hi = out["p_lowest20"], out["p_highest20"]
    out["low_over_high"] = float(lo / hi) if hi > 0 else float("inf")
    out["corr_logp_absdiff"] = (
        float(np.corrcoef(lp_sample, absd)[0, 1])
        if np.std(lp_sample) > 0 and np.std(absd) > 0 else float("nan")
    )
    return out


def bootstrap_ci(values_per_seq: list[np.ndarray], n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """对 abs_diff_mean 做序列级 bootstrap。

    *** 为什么必须做 ***
    重尾分布下均值极不稳定 —— 少数 token 主导。你现在所有跨配置比较
    (tp1=0.0082 vs tp4=0.0116) 都没有误差棒, 而且不同配置采样出的轨迹
    本来就不同 (只有 8-23/512 条 token 序列相同), 差异里混着轨迹抽样噪声。
    按序列重采样才能把这部分噪声估出来。
    """
    rng = np.random.default_rng(seed)
    n = len(values_per_seq)
    sums = np.array([v.sum() for v in values_per_seq])
    counts = np.array([len(v) for v in values_per_seq])
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = counts[idx].sum()
        means.append(sums[idx].sum() / c if c else np.nan)
    means = np.array(means)
    return {
        "abs_diff_mean": float(sums.sum() / counts.sum()),
        "ci95_lo": float(np.percentile(means, 2.5)),
        "ci95_hi": float(np.percentile(means, 97.5)),
        "boot_std": float(means.std()),
    }


def length_variance_check(mask: np.ndarray) -> dict:
    """len_corr 接近 0 可能只是因为长度本身没有方差 (全被截断)。

    截断率 99.6% 时长度全挤在 max_new 附近, 相关系数无意义 —— 这是量程限制,
    不是"不存在长程累积"的证据。这个区分很重要, 别把它写进论文当负结果。
    """
    lens = mask.sum(axis=1)
    cv = float(lens.std() / lens.mean()) if lens.mean() > 0 else 0.0
    return {
        "len_p10": float(np.percentile(lens, 10)),
        "len_p90": float(np.percentile(lens, 90)),
        "len_cv": cv,
        "len_corr_interpretable": bool(cv > 0.15),
        "note": "cv < 0.15 时 len_corr 不可解读, 需要更长的 max_new 或关闭思考模式",
    }


# ===========================================================================

def run(sample_path: str, recompute_path: str) -> dict:
    lp_s, lp_r, mask = load_pair(sample_path, recompute_path)
    m = mask.astype(bool)
    d = (lp_r - lp_s)[m]

    per_seq = [np.abs(lp_r[i] - lp_s[i])[m[i]] for i in range(lp_s.shape[0]) if m[i].any()]

    return {
        "sign": sign_asymmetry(d),
        "topk_hypothesis": topk_renorm_prediction(d),
        "tail_vs_confidence": tail_vs_confidence(lp_s[m], d),
        "bootstrap": bootstrap_ci(per_seq),
        "length": length_variance_check(mask),
    }


def verdict(r: dict) -> list[str]:
    out = []
    s, t = r["sign"], r["tail_vs_confidence"]
    if s["frac_negative_nonzero"] > 0.9 and s["onesidedness"] > 0.7:
        out.append("[!!] 失配几乎完全单侧 -> 强烈指向 top-k 重归一化 artifact, "
                   "而非 kernel 数值差异。修掉 top_k 后重跑, 现有结论全部作废。")
    elif s["frac_negative_nonzero"] > 0.65:
        out.append("[!] 存在明显单侧成分 -> artifact 与真实差异混合。"
                   "必须修掉 top_k 才能分离两者。")
    else:
        out.append("[ok] 失配基本对称 -> 符合 kernel/归约顺序差异, "
                   "不是重归一化 artifact。")

    if r["topk_hypothesis"]["frac_mass_above_1"] < 0.05:
        out.append("[!] 几乎全部 Δ<=0, 与截断质量的解释一致。")

    if t["low_over_high"] > 3:
        out.append(f"[!] 重尾集中在低概率 token ({t['low_over_high']:.1f}x), "
                   "这也是 artifact 的特征。")

    b = r["bootstrap"]
    out.append(f"[ci] abs_diff_mean = {b['abs_diff_mean']:.5f} "
               f"[{b['ci95_lo']:.5f}, {b['ci95_hi']:.5f}] —— "
               "跨配置差异要超出这个区间才算真实。")

    if not r["length"]["len_corr_interpretable"]:
        out.append(f"[!] 长度变异系数仅 {r['length']['len_cv']:.3f}, "
                   "len_corr 不可解读。别把它当作'无长程累积'的证据。")
    return out


def selftest():
    rng = np.random.default_rng(0)
    n, t = 256, 400
    mask = np.ones((n, t))
    lp_s = -rng.exponential(1.5, size=(n, t))

    # 场景 1: 纯 artifact (单侧, 且与 token 概率相关)
    mass = np.clip(1 - 0.02 * np.exp(lp_s * -0.5) * rng.random((n, t)), 0.5, 1.0)
    d1 = np.log(mass)
    r1 = {"sign": sign_asymmetry(d1.ravel()),
          "tail_vs_confidence": tail_vs_confidence(lp_s.ravel(), d1.ravel())}
    assert r1["sign"]["frac_negative_nonzero"] > 0.95, r1["sign"]
    assert r1["sign"]["onesidedness"] > 0.9
    print(f"✓ 场景1 纯 artifact: 单侧率 {r1['sign']['frac_negative_nonzero']:.3f}, "
          f"单侧度 {r1['sign']['onesidedness']:.3f}")

    # 场景 2: 纯 kernel 噪声 (对称)
    d2 = rng.normal(0, 0.01, size=(n, t))
    r2 = sign_asymmetry(d2.ravel())
    assert 0.45 < r2["frac_negative_nonzero"] < 0.55
    assert r2["onesidedness"] < 0.1
    print(f"✓ 场景2 纯 kernel 噪声: 单侧率 {r2['frac_negative_nonzero']:.3f}, "
          f"单侧度 {r2['onesidedness']:.3f}")

    # 场景 3: 混合
    d3 = d1 * 0.5 + d2
    r3 = sign_asymmetry(d3.ravel())
    assert 0.6 < r3["frac_negative_nonzero"] < 0.95
    print(f"✓ 场景3 混合: 单侧率 {r3['frac_negative_nonzero']:.3f} (落在中间区)")

    # bootstrap
    per_seq = [np.abs(d2[i]) for i in range(n)]
    b = bootstrap_ci(per_seq, n_boot=500)
    assert b["ci95_lo"] < b["abs_diff_mean"] < b["ci95_hi"]
    print(f"✓ bootstrap: {b['abs_diff_mean']:.5f} "
          f"[{b['ci95_lo']:.5f}, {b['ci95_hi']:.5f}]")

    # 长度量程限制
    narrow = np.zeros((n, t))
    for i in range(n):
        narrow[i, : rng.integers(380, 400)] = 1
    lc = length_variance_check(narrow)
    assert not lc["len_corr_interpretable"]
    print(f"✓ 长度检查: cv={lc['len_cv']:.4f}, 正确判定为不可解读")
    print("\n诊断层可以信任了。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample")
    ap.add_argument("--recompute")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    r = run(args.sample, args.recompute)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print("\n" + "\n".join(verdict(r)))


if __name__ == "__main__":
    main()
