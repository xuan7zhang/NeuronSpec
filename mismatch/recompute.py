"""
入口 2: 训练侧重算 logprob。

    # GPU 对照
    python recompute.py --backend gpu --sample results/samples/gpu_xxx.jsonl

    # Trainium
    python recompute.py --backend neuron --sample results/samples/neuron_xxx.jsonl \
                        --bucket-len 1024 --span 0.75 --lnc 2 --tp 1
    # neuron 后端目前只支持 --tp 1 (单进程单核 XLA), 见 build_model 注释。

这一步用 *训练侧* 的栈 (GPU: HF transformers; Neuron: NxD + XLA) 对采样阶段
产生的完全相同的 token 序列重新计算 logprob。权重相同、输入相同, 任何差异都
来自编译路径、tiling、归约顺序和 padding —— 正是我们要归因的对象。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

import config as C


# ---------------------------------------------------------------------------

def setup_neuron_env(lnc: int, cores: str | None):
    """必须在 import torch_xla / 建模型之前调用。"""
    # 已核对 (Neuron SDK 2.27 / NxD 0.16): 变量名正确, env 优先级最高。
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    # 单进程只用 xla:0 一个 core。LNC=2 时可见 "0" 即可; LNC=1 时 torch_xla 2.8.1 的
    # PJRT 插件在只有 1 个可见核时设备枚举崩溃 (pjrt_c_api_client.h:380 Check failed),
    # 可见 2 个物理核就正常, 模型仍只跑在 xla:0 上。
    os.environ["NEURON_RT_VISIBLE_CORES"] = cores or ("0" if lnc == 2 else "0-1")
    # 根盘是 EBS, stop/start 不丢; /tmp 是 tmpfs 重启即清空。
    os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/home/ec2-user/neuron-cache/train")
    # *** --lnc 必须显式传给编译器 ***
    # NEURON_LOGICAL_NC_CONFIG 只作用于运行时; torch_xla/libneuronxla 路径不会把它
    # 转成编译 flag (缓存里 compile_flags.json 只有 --target=trn2 --model-type=transformer),
    # neuronx-cc 对 trn2 默认 --lnc=2, 于是 LNC=1 运行时会拿到 LNC=2 的 NEFF,
    # 加载时报 "NEFF is invalid"。显式传 --lnc 也让两种 LNC 的编译缓存分开。
    # MISMATCH_CC_EXTRA: 实验用额外编译 flag (如 --auto-cast=none), 追加在固定 flag 之后
    os.environ["NEURON_CC_FLAGS"] = f"--model-type=transformer --lnc={lnc} " + os.environ.get("MISMATCH_CC_EXTRA", "")
    # 不设 XLA_USE_BF16: torch_xla 2.9 已移除该变量 (源码里只剩 experimental/ 一处引用),
    # 设了也无效。且它的语义是把所有 f32 强转 bf16, 会把 gather_logprobs 里
    # 刻意的 .float() log_softmax 也变成 bf16, 和 GPU 对照组不对等。


def load_sample(path: str):
    with open(path) as f:
        meta = json.loads(f.readline())["_meta"]
        rows = [json.loads(l) for l in f if l.strip()]
    return rows, meta


def select_by_span(rows, bucket_len: int, span: float, tol: float = 0.08):
    """筛出实际长度接近 span × bucket_len 的序列。

    *** 为什么筛选而不是截断 ***
    人为截断会改变序列内容, 两侧就不是同一条序列了。筛选保持内容不变,
    只改变"这条序列在桶里占多满", 这才是我们要研究的自变量 (H1b)。

    代价是每个 span 的样本数不同。analyze 侧按实际数量报告, 不要跨 span
    比较绝对 token 数。
    """
    target = span * bucket_len
    lo, hi = target * (1 - tol), target * (1 + tol)
    keep = [r for r in rows if lo <= len(r["token_ids"]) <= hi]
    if len(keep) < 32:
        raise SystemExit(
            f"span={span} 下只筛到 {len(keep)} 条 (需要 >=32)。"
            f"目标长度 {target:.0f}, 放宽 --tol 或换 bucket_len。"
        )
    return keep


def pad_batch(rows, bucket_len: int, pad_id: int):
    """pad 到固定的桶长度。这是 XLA 静态形状要求, 也是失配的怀疑来源之一。"""
    n = len(rows)
    ids = torch.full((n, bucket_len), pad_id, dtype=torch.long)
    attn = torch.zeros((n, bucket_len), dtype=torch.long)
    for i, r in enumerate(rows):
        t = r["token_ids"][:bucket_len]
        ids[i, : len(t)] = torch.tensor(t)
        attn[i, : len(t)] = 1
    return ids, attn


def gather_logprobs(logits, input_ids):
    """位置 t 的输出预测 token t+1。与 grpo_core 里的实现保持一致。"""
    lp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    return torch.gather(lp, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def forward_scalar(model, ids, attn):
    """默认验证器图: 所有位置的 logprob 标量。ids/attn 已在 device 上。"""
    logits = model(input_ids=ids, attention_mask=attn).logits
    return gather_logprobs(logits, ids)                                   # [B, L-1]


def forward_rows(model, ids, attn, idx, ROW_K):
    """行图: 同一个 log_softmax 张量既取标量又取 idx 处的整行 (one-hot 掩码求和, 见 main 里的注释)。
    与 forward_scalar 的标量逐位相同 (tools/test_3c_rows.py 验证)。"""
    logits = model(input_ids=ids, attention_mask=attn).logits
    lp_full = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)      # [1, L-1, V]
    lp = torch.gather(lp_full, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    onehot = torch.zeros(ROW_K, lp_full.shape[1], dtype=torch.float32)
    onehot[torch.arange(ROW_K), torch.tensor(idx)] = 1.0
    onehot = onehot.to(lp_full.device)
    rows_dev = (onehot.unsqueeze(-1) * lp_full[0].unsqueeze(0)).sum(dim=1)   # [ROW_K, V]
    return lp, rows_dev


def forward_scalar_fused(model, ids, attn, q_lp, u, seg_mask):
    """融合图 (验证-回滚用): 一次 prefill 同时得到 (a) 所有位置的 logprob 标量, (b) 图内按接受规则
    u < min(1, exp(lp - q_lp)) 算出的首个拒绝位置 idx (无拒绝时 = L-1), (c) 该位置的完整 log_softmax 行。
    q_lp / u / seg_mask: [L-1], 段外 u=-1 且 seg_mask=False (永不拒绝)。静态形状。
    标量是否与 forward_scalar 逐位相同由 tools/test_fused_row.py 验证。"""
    logits = model(input_ids=ids, attention_mask=attn).logits
    lp_full = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)      # [1, L-1, V]
    lp = torch.gather(lp_full, 2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    r = torch.exp(lp[0] - q_lp)
    reject = (u >= torch.clamp(r, max=1.0)) & seg_mask
    n = lp_full.shape[1]
    ar = torch.arange(n, device=lp_full.device)
    idx = torch.where(reject, ar, torch.full_like(ar, n)).min()
    onehot = (ar == torch.clamp(idx, max=n - 1)).to(torch.float32)
    row = (onehot.unsqueeze(-1) * lp_full[0]).sum(dim=0)                 # [V]
    return lp, idx, row


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpu", "neuron"], required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--model", default=None, help="默认取 sample 文件里的 model")
    ap.add_argument("--bucket-len", type=int, default=1024)
    ap.add_argument("--span", type=float, default=None, help="不给则不筛选")
    ap.add_argument("--tol", type=float, default=0.08)
    ap.add_argument("--lnc", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--cores", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--logits-positions", default=None,
                    help="json {seq_id: [completion 位置, ...]}: 额外输出这些位置的完整 log_softmax 行到 <out>.rows.npz (残差采样用)")
    args = ap.parse_args()

    if args.backend == "neuron":
        setup_neuron_env(args.lnc, args.cores)

    rows, smeta = load_sample(args.sample)
    model_path = args.model or smeta["model"]
    print(f"[in] {len(rows)} 条 from {args.sample}")

    if args.span is not None:
        rows = select_by_span(rows, args.bucket_len, args.span, args.tol)
        print(f"[span={args.span}] 筛后 {len(rows)} 条")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model, device = build_model(args.backend, model_path, args.tp)
    model.eval()

    want_rows = None
    if args.logits_positions:
        want_rows = {k: sorted(set(int(x) for x in v)) for k, v in json.load(open(args.logits_positions)).items()}
        rows = [r for r in rows if r["seq_id"] in want_rows]
        if args.batch_size != 1:
            raise SystemExit("--logits-positions 只支持 --batch-size 1")
        # 固定 K 保证 XLA 静态形状 (一个 K 一张图); 不足的位置用最后一个位置重复填充
        ROW_K = max(len(v) for v in want_rows.values())
        row_buf = []   # (seq_id, positions, rows[n, V])
        print(f"[rows] {len(rows)} 条, 每条最多 {ROW_K} 行")

    t0 = time.perf_counter()
    results = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        ids, attn = pad_batch(chunk, args.bucket_len, pad_id)
        ids, attn = ids.to(device), attn.to(device)

        with torch.no_grad():
            if want_rows is None:
                lp = forward_scalar(model, ids, attn)      # [B, bucket_len-1]  (默认路径, 图与之前逐位相同)
            else:
                # 同一个 log_softmax 张量既取标量 (与默认路径同公式) 又取整行。注意这会换一张 XLA 图;
                # 行/标量是否与默认图逐位一致由 tools/test_rows.py 验证, 不能假定。
                p0 = chunk[0]["prompt_len"]
                pos_list = want_rows[chunk[0]["seq_id"]]
                idx = [p0 - 1 + q for q in pos_list] + [p0 - 1 + pos_list[-1]] * (ROW_K - len(pos_list))
                # 不能用 lp_full[0, idx, :] (gather): neuronx-cc 2.22 对 [1023,151936] 的行 gather 报
                # NCC_ITEN404 内部错误 (DataLocalityOpt)。改用 one-hot 掩码 + 求和: 每个输出元素 = x*1 + Σ(y*0),
                # fp32 下加 0 精确, 与 gather 逐位等价 (由 tools/test_3c_rows.py 验证)。
                lp, rows_dev = forward_rows(model, ids, attn, idx, ROW_K)
            if args.backend == "neuron":
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            lp = lp.float().cpu()
            if want_rows is not None:
                row_buf.append((chunk[0]["seq_id"], pos_list, rows_dev.float().cpu().numpy()[: len(pos_list)]))

        for j, r in enumerate(chunk):
            # 只取 completion 段, 与采样侧的 logprobs 对齐。
            # 采样侧第 k 个 logprob 对应绝对位置 prompt_len + k,
            # 由位置 prompt_len + k - 1 的输出预测 -> lp 的下标 prompt_len + k - 1
            p = r["prompt_len"]
            n_comp = len(r["logprobs"])
            seg = lp[j, p - 1 : p - 1 + n_comp].tolist()
            results.append({
                "seq_id": r["seq_id"],
                "token_ids": r["token_ids"],   # 原样带出, analyze 会逐位校验
                "logprobs": seg,
            })

        if i % (args.batch_size * 8) == 0:
            print(f"  {i + len(chunk)}/{len(rows)}")

    elapsed = time.perf_counter() - t0
    tag = args.tag or (
        f"{args.backend}_{model_path.split('/')[-1]}_b{args.bucket_len}"
        f"_s{args.span}_lnc{args.lnc}_tp{args.tp}"
    )
    path = f"{C.RECOMPUTE_DIR}/{tag}.jsonl"
    os.makedirs(C.RECOMPUTE_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_meta": {
            "backend": args.backend, "model": model_path,
            "bucket_len": args.bucket_len, "span": args.span,
            "lnc": args.lnc, "tp": args.tp, "elapsed_s": elapsed,
            "from_sample": args.sample,
        }}) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")

    if want_rows is not None:
        import numpy as np
        npz = f"{C.RECOMPUTE_DIR}/{tag}.rows.npz"
        np.savez_compressed(npz,
            seq_ids=np.array([b[0] for b in row_buf for _ in b[1]]),
            pos=np.array([q for b in row_buf for q in b[1]], dtype=np.int32),
            rows=np.concatenate([b[2] for b in row_buf]).astype(np.float32))
        print(f"[rows] → {npz}")
    print(f"[done] {elapsed:.1f}s → {path}")
    print(f"\n下一步:\n  python analyze.py --sample {args.sample} "
          f"--recompute {path} --out results/analysis/{tag}")


def build_model(backend: str, model_path: str, tp: int):
    if backend == "gpu":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa",
        )
        return m, "cuda"

    # --- Neuron ---
    # 已核对 (NxD 0.16.25997): 这里加载的是普通 HF 模型, 没有 NxD 并行层, TP 对它
    # 不起作用; 而 parallel_state.initialize_model_parallel() 第 548 行
    # assert torch.distributed.is_initialized(), 单进程直接调用会断言失败。
    # 所以训练侧当前是: 单进程、单逻辑 core、TP=1 的 XLA 前向。真正的 TP>1
    # 需要 torchrun 多进程 + 把 Qwen3 改写成 NxD 并行层, 是另一份代码。
    # 首次运行会触发编译, 之后走 NEURON_COMPILE_CACHE_URL 里的缓存。
    if tp != 1:
        raise SystemExit(
            f"neuron 后端目前只支持 --tp 1 (普通 HF 模型 + 单核 XLA), 收到 --tp {tp}。"
            "TP>1 需要 NxD 并行层实现, 见 build_model 注释。"
        )
    import torch_xla.core.xla_model as xm
    from transformers import AutoModelForCausalLM

    device = xm.xla_device()
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    return m.to(device), device


if __name__ == "__main__":
    main()
