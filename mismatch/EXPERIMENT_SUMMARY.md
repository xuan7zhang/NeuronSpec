# Trainium 训练/推理数值失配测量 + 验证-回滚采样 — 实验总结

日期：2026-09-08 → 09-11。机器：trn2.3xlarge（1× Trainium2，8 NeuronCore-v3，96 GiB HBM，LNC=2 → 4 逻辑核）。
模型：Qwen3-1.7B（主）、Qwen3-8B（泛化）。数据：GSM8K test 512 条冻结 prompt（`data/prompts.jsonl`，checksum `fb8e2ab826e12dfe`）。

推理侧：vLLM 0.11.0 + vllm-neuron 0.2.2+lts + NxDI 0.7.15063（NEFF 图）。训练侧：HF transformers 4.56.2 + torch_xla 2.8.1 lazy tensor（XLA 图）。两侧同一个 neuronx-cc 2.22.12471、runtime 2.29.40、torch 2.8.0。

---

## 1. 一句话结论

**Trainium 上的训练/推理失配不比 GPU 大，只是分布得更均匀；它来自两个编译器各自生成全部代码的逐层独立舍入，调不掉；但正因为均匀（无重尾），验证-回滚可以用 <20% 的 wall-clock 代价把它精确消掉。**

## 2. 失配测量（`analyze.py`，`results/analysis/`）

### 2.1 主结果（tp4 / lnc2 / 桶 1024，262k token）

| 指标 | Qwen3-1.7B | Qwen3-8B | H100 参照（另一侧提供） |
|---|---|---|---|
| Δ 恰好 = 0 | 0.00% | 0.00% | 23.5% |
| \|Δ\| < 1e-5 | 39.4% | 48.7% | ~50% |
| p50 | 1.8e-4 | 1.5e-5 | 3.75e-5 |
| p90 | 0.038 | 0.031 | 0.044 |
| p99 | 0.131 | 0.133 | 0.162 |
| p99.9 | 0.265 | 0.266 | 0.293 |
| max | 2.65 | 3.65 | 1.41 |
| abs_diff_mean | 0.0117 | 0.0099 | 0.0128 |
| ratio_outside_tis_band [0.8,1.25] | 0.18% | 0.20% | 0.35% |
| 序列级 log 比 std | 0.66 | 0.68 | 0.81 |

- **形状不同**：H100 双峰（¼ token 逐位相同 + 长尾，p99/p50 ≈ 4300×）；Trainium 弥散（无一 token 逐位相同，p99/p50 ≈ 750×）。原因：GPU 两侧共用 cuBLAS 等 kernel，差异集中在 attention；Trainium 的 NEFF 与 XLA 是两个编译器各自生成全部代码。
- **对 RL 的干扰更小**：TIS 裁剪带外和序列级 std 都低于 H100。
- **8B 泛化**：分布主体更紧，尾部不变，λ 降 15%。
- 两侧各自逐位确定（同 seed 重跑 Δ ≡ 0），失配 100% 来自编译路径。
- 采样侧 13% 的 logprob 精确为 0（p≈1 的 fp32 舍入），训练侧同位置 ~1e-9，进入 "<1e-5" 而非 "=0"；比较零值比例时要注意这一点。

### 2.2 归因（全部实测，`results/analysis/SUMMARY.md`、`CTE_O2.md`、`PREFIX_SCALING.md`）

| 自变量 | 结果 | 结论 |
|---|---|---|
| **TP** | tp1 0.0082 → tp2 0.0117，tp2/4/8 一致（bootstrap CI 宽 3e-4） | **阶跃**：代价来自 all-reduce 的存在（分片先舍到 bf16 再求和），不来自规模。占总失配 ~30% |
| LNC 1 vs 2 | 1.00× | 无影响 |
| 训练侧桶长 1024 vs 2048 | 1.00× | padding 无影响 |
| 序列位置 | pos≥64 完全平坦；前缀扫描指数 k ≤ 0.5；lag-1 自相关 ≈ 0 | **无长程累积**，误差逐 token 独立，√T 就是 CLT |
| 序列头部 | pos0 \|Δ\| = 0.046（8B 0.061），是稳态的 4–6× | context-encoding 图相对训练图偏差更大 |
| CTE 编译等级 -O1→-O2 | 0.0457 → 0.0426 | 不是原因 |
| CTE flash attention kernel 关闭 | 0.0468 | 不是原因（494/512 prompt 本就不走 kernel） |
| CTE 桶 128→1024 | 0.0410 | 不是原因 |
| 训练侧 `--auto-cast=none` | 与默认图 100% 逐位相同 | 精度策略不是原因 |
| top-k 截断 | onesidedness 0.03–0.05（对称） | 不是 artifact |

尚未归因的：CTE 图 4–6× 尖峰的具体算子来源（需逐层 dump HLO/hidden state）。

### 2.3 指标稳健性
`kl_k3` 对重尾极不稳健：单个 Δ=−8.8 的 token 把它从 0.0005 抬到 0.025，而 abs_diff_mean / p99.9 / λ 不动。报 KL 必须配尾部分位数。

## 3. 验证-回滚采样（`verify_rollback.py`，`results/vr/`）

算法：Leviathan 2023 接受规则，草稿 = 推理引擎 q，目标 = 训练图 p。r_t = p/q，以 min(1,r) 接受；拒绝时从残差 (p−q)⁺ 采（q 在 top-256 外视为 0），之后 token 作废，从 [prefix, a'] 重新 decode。验证器就是 `recompute.py` 的训练图。

### 3.1 前提（全部通过）
- top-k 无污染（§2.2）。
- **H_prefix bit-exact**：同桶长下前缀 256/512/全长 prefill，重叠位置 100% 逐位相同 → 可分块验证。
- 训练图行提取（残差需要 p 整行）与默认图逐位相同（one-hot 掩码求和；直接 gather 触发 neuronx-cc 内部错误 NCC_ITEN404）。
- 融合图（一次 prefill 同时给标量 + 图内首拒绝位置 + 整行）与默认图逐位相同。

### 3.2 正确性
8 个 run、200 万 token：接受率经验值 = E[min(1,r)] 理论值到 1e-4；final.jsonl 用未改动的默认训练图重算 **Δ ≡ 0**（含残差替换位置）；同 seed 重跑 512/512 轨迹逐位相同。近似误差（q 在 top-256 外质量）中位数 1e-5、p99 ~7e-3。

### 3.3 开销（Qwen3-1.7B，T=512，`results/vr/SUMMARY.md`，`overhead_vs_lambda.png`）

| run | tp | C | 验证器 | 融合图 | λ | prefill/序列 | 资源口径 | **wall-clock** |
|---|---|---|---|---|---|---|---|---|
| tp2_CT | 2 | T | 1 | – | 0.0058 | 7.0 | 235% | 236% |
| tp2_C128 | 2 | 128 | 1 | – | 0.0056 | 9.1 | 103% | 103% |
| tp2_C64 | 2 | 64 | 1 | – | 0.0057 | 12.7 | 91% | 92% |
| tp1_C64 | 1 | 64 | 1 | – | 0.0039 | 11.5 | 60% | 61% |
| tp2_C128_v2 | 2 | 128 | 2 | ✓ | 0.0057 | 6.3 | 81% | **41%** |
| tp2_C64_v2 | 2 | 64 | 2 | ✓ | 0.0056 | 10.1 | 78% | **19%** |
| tp2_C32_v2 | 2 | 32 | 2 | ✓ | 0.0056 | 17.9 | 104% | **9%** |
| tp1_C64_v2 | 1 | 64 | 3 | ✓ | 0.0039 | 9.5 | 54% | **13%** |
| tp2_C64_T1024_v2 | 2 | 64 | 2 | ✓ | 0.0052 | 19.3 | 118% | **18%** |

- **资源口径**（忙时之和）：地板由 T/C 次全桶 prefill 决定——Trainium 静态形状让每次验证 prefill 都是整桶成本（§4 模型的 len_frac 要从 0.5 改成 1.0）；验证器 batch>1 被编译器 HBM 估算挡死（B=2 → 56 GB；分块 log_softmax 反而 93 GB）。
- **wall-clock 口径**：DP 验证器 + 两条 lane 流水线 + 融合图把验证藏进 decode，9–19%，低于 SGLang 确定性推理的 ~34%，远低于 TBIK 56–135%；T 翻倍不变。
- 开销是 λ 的函数（两个 λ 点落在同一条修正模型曲线上）。

## 4. 环境事实（踩过的坑，都在 `tools/`、`hooks/` 有对应脚本）
- 只有 `aws_neuronx_venv_pytorch_inference_vllm` 可用；三个 torch-2.9 venv 因 glibc 2.34 < 2.35 无法 import torch_xla。
- LNC=2 → 4 逻辑核 → TP ≤ 4；TP=8 需 LNC=1。
- vllm-neuron on-device sampler 把 top_k 截到 256，全词表采样必须 `on_device_sampling_config=None`（CPU 采样，也是唯一返回 logprob 的路径）；`enable_prefix_caching=False`。
- XLA 路径不把 LNC 传给编译器，必须显式 `--lnc=N`，否则 LNC=1 加载 LNC=2 的 NEFF 报 "NEFF is invalid"。
- torch_xla 2.8.1 在 LNC=1 且只有 1 个可见核时 PJRT 崩溃，需 `NEURON_RT_VISIBLE_CORES=0-1`。
- 训练侧 B≥2 被 HBM 估算拒绝（与 padding、T 无关）。
- `prefill_cost` 实测 0.04–0.05，不是脚本默认 0.15。
- 冷编译两侧各 ~4 分钟；CPU 全词表采样 512 条 ~18 分钟。
- vllm-neuron 会把整份 HF 权重复制到 `local-models/`，8B 需 ~30 GB；已把 `local-models` 和 HF cache 移到 instance store `/scratch`（stop 即清）。

## 5. 未做 / 需决策
- 训练侧真 TP>1（需 NxD 并行版 Qwen3）——三行分解的第二行缺失。
- CTE 尖峰的最终归因（逐层 dump）。
- prompt 格式：裸 completion 让 Qwen3 不发 EOS，截断 99%；换 chat template = 换数据集。
- GPU 对照的 `flash_attention_2` 判别实验（GPU 侧）。

## 6. 复现
```
source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate
python analyze.py --selftest && python diagnose.py --selftest && python rejection_analysis.py --selftest
python prepare_data.py                                     # data/prompts.jsonl, checksum fb8e2ab826e12dfe
python sample.py --backend neuron --tp 4 --lnc 2           # 推理侧
python recompute.py --backend neuron --sample results/samples/<s>.jsonl --bucket-len 1024 --lnc 2 --tp 1 --batch-size 1
python analyze.py --sample <s> --recompute <r> --out results/analysis/<name>
python verify_rollback.py run --run tp2_C64_v2 --tp 2 --lnc 2 --chunk 64 --n 512 --verifiers 2 --lanes 2 --fused
python tools/vr_summary.py > results/vr/SUMMARY.md
```
原始 jsonl（samples / recompute / vr final，约 350 MB）未入库，在 trn2 机器 `~/mismatch/results/` 下。
