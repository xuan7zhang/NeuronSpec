"""
验证-回滚方案的开销估计。在现有 sample/recompute 数据上跑, 不需要新采样。

    python rejection_analysis.py --sample <s.jsonl> --recompute <r.jsonl>
    python rejection_analysis.py --selftest

回答: 如果用 speculative 接受规则把 rollout 拉到训练图上, 要付多少代价?
输出: 每 token 拒绝率、每序列期望回滚数、不同分块长度下的开销表。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from analyze import load_pair


def rejection_probs(lp_sample: np.ndarray, lp_recompute: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    """ρ_t = max(0, 1 − p_t(a_t)/q_t(a_t)) = max(0, 1 − exp(Δ_t))。

    q 是采样侧 (decode), p 是训练图 (prefill)。
    Δ = logp − logq。Δ<0 时 p<q, 该 token 有被拒绝的概率。
    """
    d = lp_recompute - lp_sample
    rho = np.clip(1.0 - np.exp(d), 0.0, 1.0)
    return rho * mask


def per_sequence_stats(rho: np.ndarray, mask: np.ndarray) -> dict:
    lens = mask.sum(axis=1)
    exp_rej = rho.sum(axis=1)                       # 每序列期望拒绝数
    # 一个都不拒绝的概率 = Π(1−ρ_t)
    with np.errstate(divide="ignore"):
        log_acc = np.where(mask > 0, np.log1p(-np.clip(rho, 0, 1 - 1e-12)), 0.0)
    p_clean = np.exp(log_acc.sum(axis=1))
    # 首次拒绝的期望位置 (只在有拒绝的序列上有意义)
    first = _expected_first_rejection(rho, mask)
    return {
        "lambda_per_token": float(rho.sum() / mask.sum()),
        "expected_rejections_per_seq": float(exp_rej.mean()),
        "p_seq_no_rejection": float(p_clean.mean()),
        "expected_first_rejection_pos": float(first),
        "mean_len": float(lens.mean()),
        "rho_p50": float(np.percentile(rho[mask > 0], 50)),
        "rho_p99": float(np.percentile(rho[mask > 0], 99)),
        "rho_max": float(rho.max()),
    }


def _expected_first_rejection(rho, mask):
    """E[首次拒绝位置 | 至少一次拒绝]。P(首次在 t) = Π_{s<t}(1−ρ_s)·ρ_t"""
    n, t = rho.shape
    surv = np.cumprod(np.where(mask > 0, 1 - rho, 1.0), axis=1)
    surv_prev = np.concatenate([np.ones((n, 1)), surv[:, :-1]], axis=1)
    p_first = surv_prev * rho
    pos = np.arange(t)[None, :]
    num = (p_first * pos).sum()
    den = p_first.sum()
    return num / den if den > 0 else float("nan")


def overhead_table(lam: float, T: float, prefill_cost: float = 0.15,
                   chunks=(32, 64, 128, 256, None)) -> list[dict]:
    """不同分块长度 C 下的开销估计。

    模型 (§4 of VERIFY_ROLLBACK.md):
      回滚次数        ≈ λ·T
      每次回滚浪费    ≈ C/2 个 decode token    (C=None 时为 T/2)
      验证 prefill 数 ≈ T/C + λ·T               (C=None 时为 1 + λ·T)
    prefill_cost: 一次全长 prefill 相对于全长 decode 的成本, 你测的是 0.1-0.2。

    这是一阶估计。回滚后的重采部分自身也会有拒绝, 这里忽略 (二阶小量)。
    """
    rows = []
    n_rej = lam * T
    for C in chunks:
        c_eff = T if C is None else min(C, T)
        wasted_decode = n_rej * c_eff / 2
        n_prefill = (1 + n_rej) if C is None else (T / c_eff + n_rej)
        # 分块验证时每次 prefill 只到当前块尾, 平均长度 T/2 (H_prefix 成立时)
        prefill_len_frac = 1.0 if C is None else 0.5
        decode_overhead = wasted_decode / T
        prefill_overhead = n_prefill * prefill_len_frac * prefill_cost
        rows.append({
            "chunk": "all" if C is None else C,
            "rollbacks": round(n_rej, 2),
            "decode_overhead": round(decode_overhead, 3),
            "prefill_overhead": round(prefill_overhead, 3),
            "total_overhead": round(decode_overhead + prefill_overhead, 3),
        })
    return rows


def run(sample_path, recompute_path, prefill_cost):
    lp_s, lp_r, mask = load_pair(sample_path, recompute_path)
    rho = rejection_probs(lp_s, lp_r, mask)
    stats = per_sequence_stats(rho, mask)
    table = overhead_table(stats["lambda_per_token"], stats["mean_len"], prefill_cost)
    return stats, table, rho, mask


def print_report(stats, table):
    print(json.dumps(stats, indent=2))
    print("\n开销估计 (相对于纯 decode 成本, 1.0 = 翻倍):")
    print(f"  {'块长':>6} {'回滚数':>8} {'decode':>8} {'prefill':>8} {'总计':>8}")
    for r in table:
        print(f"  {str(r['chunk']):>6} {r['rollbacks']:>8} "
              f"{r['decode_overhead']:>8} {r['prefill_overhead']:>8} "
              f"{r['total_overhead']:>8}")
    best = min(table, key=lambda r: r["total_overhead"])
    print(f"\n最优块长 {best['chunk']}, 总开销 {best['total_overhead']:.1%}")
    print("参照: TBIK 56-135%, SGLang 确定性 ~34%, TML 61.5%")
    if stats["p_seq_no_rejection"] > 0.9:
        print("\n[ok] 超过 90% 的序列零回滚, 方案几乎免费")
    elif stats["expected_rejections_per_seq"] > 5:
        print("\n[!] 每序列超过 5 次回滚。若 top_k 未修, 这个数被高估; 先修再算")


def selftest():
    rng = np.random.default_rng(0)
    n, t = 200, 512
    mask = np.ones((n, t))
    lq = -rng.exponential(1.5, size=(n, t))

    # 场景 A: 完全一致 -> 零拒绝
    rho = rejection_probs(lq, lq.copy(), mask)
    s = per_sequence_stats(rho, mask)
    assert s["lambda_per_token"] == 0.0 and s["p_seq_no_rejection"] == 1.0
    print("✓ 场景 A 一致: 零拒绝")

    # 场景 B: 对称小噪声, Δ ~ N(0, 0.01)
    lp = lq + rng.normal(0, 0.01, size=(n, t))
    rho = rejection_probs(lq, lp, mask)
    s = per_sequence_stats(rho, mask)
    # E[max(0, 1−e^Δ)] ≈ E[max(0,−Δ)] = σ/√(2π) ≈ 0.004
    assert 0.003 < s["lambda_per_token"] < 0.005, s["lambda_per_token"]
    print(f"✓ 场景 B 对称噪声: λ={s['lambda_per_token']:.4f}, "
          f"每序列 {s['expected_rejections_per_seq']:.2f} 次回滚")

    # 场景 C: 单侧 (artifact 型), Δ ≤ 0 恒成立 -> 每个 token 都有拒绝概率
    lp = lq + np.log(np.clip(1 - rng.exponential(0.01, size=(n, t)), 0.5, 1))
    rho = rejection_probs(lq, lp, mask)
    s = per_sequence_stats(rho, mask)
    assert s["lambda_per_token"] > 0.008
    print(f"✓ 场景 C 单侧: λ={s['lambda_per_token']:.4f} (明显高于对称同幅度), "
          f"零回滚序列占比 {s['p_seq_no_rejection']:.3f}")

    # 开销表单调性: 块越小 decode 浪费越少
    tab = overhead_table(0.005, 512)
    dec = [r["decode_overhead"] for r in tab]
    assert dec == sorted(dec), dec
    print(f"✓ 开销表: 块 32→all 的 decode 开销 {dec[0]:.3f}→{dec[-1]:.3f}, 单调")

    # 首次拒绝位置: 均匀 ρ 时应接近几何分布均值
    rho_u = np.full((n, t), 0.01) * mask
    fp = _expected_first_rejection(rho_u, mask)
    assert 80 < fp < 110, fp   # 1/0.01 = 100, 被 T 截断略小
    print(f"✓ 首次拒绝位置: 均匀 ρ=0.01 时 {fp:.0f} (理论≈100)")
    print("\n分析层可以信任了。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample")
    ap.add_argument("--recompute")
    ap.add_argument("--prefill-cost", type=float, default=0.15,
                    help="一次全长 prefill / 一次全长 decode 的成本比")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    stats, table, _, _ = run(args.sample, args.recompute, args.prefill_cost)
    print_report(stats, table)


if __name__ == "__main__":
    main()
