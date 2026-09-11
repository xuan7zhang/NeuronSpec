# 失配测量实验 Runbook

## 三个入口

```
prepare_data.py  →  sample.py  →  recompute.py  →  analyze.py
   (无需加速器)      (推理侧)       (训练侧)        (纯 CPU)
```

中间结果全部落盘。任一步崩了只需重跑那一步。

## 等机器的两天：GPU 全流程演练

GPU 侧数据不是练手，是论文的 control 组。你要论证 Trainium 失配更大，就必须有这组数字。

```bash
python analyze.py --selftest          # 先验证分析层（已通过）
python prepare_data.py                # 冻结 512 条 prompt

python sample.py --backend gpu --model Qwen/Qwen3-1.7B --tp 1
python recompute.py --backend gpu --sample results/samples/gpu_Qwen3-1.7B_tp1_lnc2_b1024.jsonl
python analyze.py --sample <上面那个> --recompute <上面那个> --out results/analysis/gpu_baseline
```

一张 24GB 显存的卡就够跑 1.7B。没有本地 GPU 就开个 g5.xlarge，一小时一美元。

**GPU 侧的预期结果**：abs_diff_mean 在 1e-3 量级，TIS 触发率接近 0。
文献里 dense 模型的 per-token 失配约 1.0±0.1，此时 TIS 基本是 no-op。
如果你的 GPU 结果和这个对不上，说明流水线有问题，**必须在上 Trainium 前查清楚**。

## Trainium 5 天计划

| 天 | 做什么 |
|---|---|
| 1 | 环境、`NEURON_COMPILE_CACHE_URL` 指 S3、跑通 vLLM-Neuron 采样 |
| 2 | 跑通 recompute（首次编译要等）、出第一张 NEFF vs XLA 图 |
| 3 | span 扫描（3 个配置） |
| 4 | LNC 扫描（×2 = 6 个配置） |
| 5 | 缓冲 / 8B / 补跑 |

时间不够时按 `config.py` 里的 TIER 顺序砍，别砍流程。

## 三个静默出错点

1. **环境变量设晚了**：`NEURON_RT_VISIBLE_CORES` 和 LNC 必须在 import 之前设，晚了不报错，只是静默用默认值。
2. **对齐错位**：analyze 按 seq_id 对齐并逐位校验 token 序列，不靠行号。报"token 序列不一致"就是这里出了问题。
3. **采样参数被改**：`top_p=1.0, top_k=-1` 是硬约束，任何截断都让两侧不可比。

## 判读

- `ratio_outside_tis_band > 0.05` → 失配显著，H1 有支撑
- `abs_diff_mean < 1e-4` → H1 可能被证伪，重心转向 RQ3/RQ4
- fig3 位置剖面上升 → 累积效应（KV cache 漂移）；平坦 → kernel 固定偏差

两者应对方式不同，这张图对归因最关键。
