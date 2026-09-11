"""
入口 3: 分析。纯 CPU, 不依赖 Neuron, 可以在本地跑。

用法:
    python analyze.py --sample results/samples/<uid>.jsonl \
                      --recompute results/recompute/<uid>.jsonl \
                      --out results/analysis/<uid>
    python analyze.py --selftest        # 用合成数据验证分析逻辑

输出:
    stats.json   全部标量指标
    figs/*.png   四张图
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ===========================================================================
# 指标
# ===========================================================================

def compute_metrics(lp_sample: np.ndarray, lp_recompute: np.ndarray,
                    mask: np.ndarray) -> dict:
    """核心指标。

    Args:
        lp_sample:    [N, T] 推理侧(NEFF 图)采样时返回的 logprob
        lp_recompute: [N, T] 训练侧(XLA 图)对同一序列重算的 logprob
        mask:         [N, T] 1 表示有效 token

    两侧用的是完全相同的权重和完全相同的 token 序列。任何非零差异都来自
    编译路径、tiling、归约顺序、padding —— 也就是我们要归因的东西。
    """
    m = mask.astype(bool)
    d = (lp_recompute - lp_sample)[m]          # 有效 token 的 logprob 差
    log_r = d                                   # log(π_learner / π_sampler)
    r = np.exp(log_r)                           # 重要性比

    # 序列级: 每条序列的 log 比之和 (这是序列级 IS 权重的对数)
    seq_log_r = np.where(m, lp_recompute - lp_sample, 0.0).sum(axis=1)
    seq_len = m.sum(axis=1)

    def q(x, ps=(50, 90, 99, 99.9)):
        return {f"p{p}": float(np.percentile(x, p)) for p in ps}

    return {
        "n_sequences": int(lp_sample.shape[0]),
        "n_tokens": int(m.sum()),

        # --- token 级绝对差 ---
        "abs_diff_mean": float(np.abs(d).mean()),
        "abs_diff_max": float(np.abs(d).max()),
        **{f"abs_diff_{k}": v for k, v in q(np.abs(d)).items()},

        # --- 重要性比 ---
        # 完美一致时 r 恒为 1。偏离程度直接决定 TIS 之类的补偿是否必要。
        "ratio_mean": float(r.mean()),
        "ratio_std": float(r.std()),
        **{f"ratio_{k}": v for k, v in q(r).items()},
        # r 超出 [0.8, 1.25] 的比例: 这个范围外 TIS 的裁剪会真正生效
        "ratio_outside_tis_band": float(((r < 0.8) | (r > 1.25)).mean()),

        # --- 序列级 ---
        # 长程累积是关键: token 级微小差异沿生成长度复合, 长序列偏离更大
        "seq_log_ratio_mean": float(seq_log_r.mean()),
        "seq_log_ratio_std": float(seq_log_r.std()),
        "seq_log_ratio_absmax": float(np.abs(seq_log_r).max()),

        # --- KL 估计 (k3, 与训练里用的同一个估计量) ---
        "kl_k3": float((np.exp(-d) + d - 1.0).mean()),

        # --- 相关性: 差异是否随位置累积 ---
        "len_corr": _safe_corr(seq_len, np.abs(seq_log_r)),
    }


def _safe_corr(x, y) -> float:
    """零方差时 corrcoef 会除零报警。完全一致的场景下 y 恒为 0, 这是正常的。"""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def positional_profile(lp_sample, lp_recompute, mask, n_bins=32):
    """|Δlogp| 随 token 位置的变化。

    如果差异在序列后段显著增大, 说明是累积效应 (KV cache 的数值漂移);
    如果全程平坦, 说明是 kernel 的固定偏差。两者的应对方式不同, 所以
    这张图对归因很关键。
    """
    n, t = lp_sample.shape
    d = np.abs(lp_recompute - lp_sample)
    edges = np.linspace(0, t, n_bins + 1).astype(int)
    means, counts = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg_m = mask[:, a:b].astype(bool)
        means.append(float(d[:, a:b][seg_m].mean()) if seg_m.any() else np.nan)
        counts.append(int(seg_m.sum()))
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, np.array(means), np.array(counts)


# ===========================================================================
# 绘图
# ===========================================================================

def make_figures(lp_sample, lp_recompute, mask, out_dir: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    m = mask.astype(bool)
    d = (lp_recompute - lp_sample)[m]
    r = np.exp(d)

    # 图 1: token 级绝对差的分布 (log 横轴, 差异跨好几个数量级)
    fig, ax = plt.subplots(figsize=(6, 4))
    ad = np.abs(d)
    ad = ad[ad > 0]
    if len(ad):
        ax.hist(np.log10(ad), bins=80, color="#4C72B0")
    ax.set_xlabel("log10 |Δ logp|")
    ax.set_ylabel("tokens")
    ax.set_title(f"Token-level mismatch\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig1_abs_diff_hist.png", dpi=140)
    plt.close(fig)

    # 图 2: 重要性比分布 + TIS 裁剪带
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.clip(r, 0.5, 1.5), bins=80, color="#DD8452")
    for x in (0.8, 1.25):
        ax.axvline(x, ls="--", c="k", lw=1)
    ax.axvline(1.0, ls="-", c="k", lw=1)
    ax.set_xlabel("importance ratio  π_learner / π_sampler")
    ax.set_ylabel("tokens")
    ax.set_title(f"Importance ratio (dashed = TIS clip band)\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig2_ratio_hist.png", dpi=140)
    plt.close(fig)

    # 图 3: 差异随位置的变化 —— 区分累积效应和固定偏差
    centers, means, counts = positional_profile(lp_sample, lp_recompute, mask)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, means, marker="o", ms=3, color="#55A868")
    ax.set_xlabel("token position")
    ax.set_ylabel("mean |d logp|")
    ax.set_title(f"Positional profile (rising = accumulation)\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig3_positional.png", dpi=140)
    plt.close(fig)

    # 图 4: 序列长度 vs 序列级偏离 —— 长程累积的直接证据
    seq_log_r = np.where(mask.astype(bool), lp_recompute - lp_sample, 0.0).sum(1)
    seq_len = mask.sum(1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(seq_len, np.abs(seq_log_r), s=6, alpha=0.4, color="#C44E52")
    ax.set_xlabel("sequence length (valid tokens)")
    ax.set_ylabel("|Σ Δ logp|")
    ax.set_title(f"Sequence-level drift vs length\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig4_seq_vs_len.png", dpi=140)
    plt.close(fig)


# ===========================================================================
# IO
# ===========================================================================

def load_pair(sample_path: str, recompute_path: str):
    """载入两侧结果并按 seq_id 对齐。

    *** 对齐是这里唯一会静默出错的地方 ***
    两个文件的行顺序不保证一致 (recompute 可能按桶重排过)。必须按 seq_id
    显式对齐, 不能靠行号。对齐错了会得到一个看起来很大很像样的"失配",
    实际是在比不同的序列。
    """
    def load(p):
        out = {}
        with open(p) as f:
            for line in f:
                o = json.loads(line)
                if "_meta" in o:      # sample.py / recompute.py 的首行是 meta, 不是序列
                    continue
                out[o["seq_id"]] = o
        return out

    a, b = load(sample_path), load(recompute_path)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("两个文件没有共同的 seq_id, 检查是否来自同一次采样")
    if len(ids) < len(a):
        print(f"[warn] {len(a) - len(ids)} 条序列只在 sample 侧出现, 已忽略")

    # 完整性校验: 同一 seq_id 的 token 序列必须逐位相同, 否则不是同一条序列
    for i in ids[: min(20, len(ids))]:
        if a[i]["token_ids"] != b[i]["token_ids"]:
            raise ValueError(f"seq_id={i} 两侧 token 序列不一致, 无法比较")

    t = max(len(a[i]["logprobs"]) for i in ids)
    n = len(ids)
    lp_s = np.zeros((n, t), dtype=np.float64)
    lp_r = np.zeros((n, t), dtype=np.float64)
    mask = np.zeros((n, t), dtype=np.float64)
    for row, i in enumerate(ids):
        ls, lr = a[i]["logprobs"], b[i]["logprobs"]
        k = min(len(ls), len(lr))
        lp_s[row, :k] = ls[:k]
        lp_r[row, :k] = lr[:k]
        mask[row, :k] = 1.0
    return lp_s, lp_r, mask


# ===========================================================================

def selftest():
    """用合成数据验证分析逻辑。上机前先跑这个。"""
    rng = np.random.default_rng(0)
    n, t = 256, 512
    lens = rng.integers(64, t, size=n)
    mask = (np.arange(t)[None, :] < lens[:, None]).astype(np.float64)
    lp_s = -rng.exponential(1.5, size=(n, t))

    # 场景 A: 完全一致
    mA = compute_metrics(lp_s, lp_s.copy(), mask)
    assert mA["abs_diff_max"] == 0.0
    assert abs(mA["ratio_mean"] - 1.0) < 1e-12
    assert mA["ratio_outside_tis_band"] == 0.0
    assert np.isnan(mA["len_corr"])   # 零方差时应返回 nan 而非报警
    print("✓ 场景 A 完全一致: ratio=1, 无 TIS 触发")

    # 场景 B: 固定的小噪声 (模拟 kernel 差异)
    lp_B = lp_s + rng.normal(0, 0.002, size=(n, t))
    mB = compute_metrics(lp_s, lp_B, mask)
    assert 0.001 < mB["abs_diff_mean"] < 0.003
    assert mB["ratio_outside_tis_band"] < 0.01
    print(f"✓ 场景 B 小噪声: abs_diff_mean={mB['abs_diff_mean']:.4f}, "
          f"TIS 触发率={mB['ratio_outside_tis_band']:.4f}")

    # 场景 C: 随位置累积 (模拟 KV cache 漂移)
    ramp = np.linspace(0, 0.05, t)[None, :]
    lp_C = lp_s + rng.normal(0, 1, size=(n, t)) * ramp
    mC = compute_metrics(lp_s, lp_C, mask)
    _, means, _ = positional_profile(lp_s, lp_C, mask)
    valid = means[~np.isnan(means)]
    assert valid[-1] > valid[0] * 3, "位置剖面应该显著上升"
    assert mC["len_corr"] > 0.3, f"长度相关性应为正, 得到 {mC['len_corr']}"
    print(f"✓ 场景 C 累积效应: 位置剖面上升 {valid[-1]/valid[0]:.1f}x, "
          f"长度相关 ={mC['len_corr']:.3f}")

    # 场景 D: 大失配 (会让 TIS 真正生效)
    lp_D = lp_s + rng.normal(0, 0.15, size=(n, t))
    mD = compute_metrics(lp_s, lp_D, mask)
    assert mD["ratio_outside_tis_band"] > 0.1
    print(f"✓ 场景 D 大失配: TIS 触发率={mD['ratio_outside_tis_band']:.3f}, "
          f"kl_k3={mD['kl_k3']:.5f}")

    # 对齐保护
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with open(f"{td}/a.jsonl", "w") as f:
            f.write(json.dumps({"seq_id": "x", "token_ids": [1, 2], "logprobs": [-1.0, -2.0]}) + "\n")
        with open(f"{td}/b.jsonl", "w") as f:
            f.write(json.dumps({"seq_id": "x", "token_ids": [1, 3], "logprobs": [-1.0, -2.0]}) + "\n")
        try:
            load_pair(f"{td}/a.jsonl", f"{td}/b.jsonl")
            raise AssertionError("应该检测到 token 序列不一致")
        except ValueError as e:
            assert "token 序列不一致" in str(e)
    print("✓ 对齐校验: token 序列不一致时正确报错")

    make_figures(lp_s, lp_C, mask, "/tmp/selftest_figs", title="selftest scenario C")
    n_figs = len(os.listdir("/tmp/selftest_figs"))
    assert n_figs == 4, n_figs
    print(f"✓ 出图: 4 张已生成于 /tmp/selftest_figs")
    print("\n分析层可以信任了。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample")
    ap.add_argument("--recompute")
    ap.add_argument("--out", default="./results/analysis/run")
    ap.add_argument("--title", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    lp_s, lp_r, mask = load_pair(args.sample, args.recompute)
    stats = compute_metrics(lp_s, lp_r, mask)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    make_figures(lp_s, lp_r, mask, f"{args.out}/figs", args.title or args.out)

    print(json.dumps(stats, indent=2))
    print(f"\n→ {args.out}/")
    # 结论提示
    if stats["ratio_outside_tis_band"] > 0.05:
        print("\n[!] 超过 5% 的 token 落在 TIS 裁剪带外 —— 失配显著, H1 有支撑")
    elif stats["abs_diff_mean"] < 1e-4:
        print("\n[!] 失配极小 —— H1 可能被证伪, 考虑把重心转向 RQ3/RQ4")


if __name__ == "__main__":
    main()
