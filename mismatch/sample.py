"""
入口 1: 采样。用推理引擎生成 completion 并记录采样时的 logprob。

    # GPU 对照 (现在就能跑, 也是论文的 control 组)
    python sample.py --backend gpu --model Qwen/Qwen3-1.7B --tp 1

    # Trainium
    python sample.py --backend neuron --model Qwen/Qwen3-1.7B --tp 4 --lnc 2
    # trn2.3xlarge: LNC=2 时只有 4 个逻辑 core, TP 最大 4; TP=8 必须配 LNC=1。

两个后端走同一份 vLLM 代码, 差别只在 device 和环境变量。这不是为了省事 ——
是为了让 GPU 和 Trainium 的对比中, 采样这一侧的逻辑严格一致, 唯一变量是硬件。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import config as C
from prepare_data import load_prompts


def setup_neuron_env(tp: int, lnc: int, cores: str | None):
    """*** 必须在 import vllm 之前调用 ***

    Neuron runtime 的这些设置在初始化后就不可变。import 顺序错了不会报错,
    只会静默使用默认值 —— 你以为在测 LNC=1, 实际跑的是默认配置。
    """
    # 已核对 (Neuron SDK 2.27 / NxDI 0.7.15063): 变量名正确。优先级
    # NEURON_LOGICAL_NC_CONFIG > logical_nc_config kwarg > 平台默认 (trn2=2),
    # 不一致时只发 UserWarning, 所以只走 env、不传 kwarg。
    # LNC 决定可见逻辑 core 数: trn2.3xlarge 上 LNC=1 -> 8 核, LNC=2 -> 4 核。
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)
    os.environ["NEURON_RT_VISIBLE_CORES"] = cores or ("0" if tp == 1 else f"0-{tp - 1}")
    # 编译缓存: 根盘是 EBS, stop/start 不丢; /tmp 是 tmpfs, 重启即清空。
    os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "/home/ec2-user/neuron-cache/infer")
    # 不设 NEURON_CC_FLAGS: NxDI 自己会传 --model-type=transformer --lnc=N,
    # libneuronxla 会把 NEURON_CC_FLAGS 追加上去, 重复传参没有意义。
    # 不设 VLLM_NEURON_FRAMEWORK: vllm-neuron 0.2.x 插件不再读这个变量。


def build_llm(backend: str, model: str, tp: int, max_len: int, max_logprobs: int = 20, neuron_override: dict | None = None):
    from vllm import LLM
    common = dict(
        model=model,
        tensor_parallel_size=tp,
        max_model_len=max_len,
        dtype="bfloat16",
        seed=C.SEED,
        # vLLM 0.11: ModelConfig.max_logprobs 默认 20, SamplingParams(logprobs=256) 会被拒。
        max_logprobs=max_logprobs,
    )
    if backend == "neuron":
        # 已核对 (vllm 0.11.0 + vllm-neuron 0.2.2): device= 已废弃且不接受 "neuron",
        # 平台由 vllm_neuron 插件检测到 /dev/neuron* 后自动注册。
        # Neuron 专属配置走 additional_config["override_neuron_config"]。
        #
        # *** on_device_sampling_config=None 是硬约束的一部分 ***
        # 插件默认开 on-device sampling, 其 global_topk 默认 256 且运行时硬上限
        # _MAX_NEURON_SAMPLING_TOP_K=256: top_k=-1 会被静默改成 top-256, 这就是
        # 截断。设为 None 后 NEFF 图输出完整 logits, 由 vLLM 标准 CPU sampler 做
        # 全词表采样并返回 logprob (on-device 路径根本不返回 logprobs)。
        return LLM(
            max_num_seqs=32,
            # vLLM V1 默认开 prefix caching, 插件此时要求显式 block_size; 关掉后
            # 插件自动令 block_size = max_model_len。实验也不该跨 prompt 复用 KV。
            enable_prefix_caching=False,
            additional_config={
                # 额外的 NxDI neuron_config 覆盖 (实验用, 默认空), 不允许覆盖 on_device_sampling_config
                "override_neuron_config": {**(neuron_override or {}), "on_device_sampling_config": None},
            },
            **common,
        )
    return LLM(gpu_memory_utilization=0.85, enforce_eager=False, **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpu", "neuron"], required=True)
    ap.add_argument("--model", default=C.MODELS[0])
    ap.add_argument("--prompts", default=f"{C.DATA_DIR}/prompts.jsonl")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--lnc", type=int, default=2, help="仅 neuron")
    ap.add_argument("--cores", default=None, help="仅 neuron, 如 '0-7'")
    ap.add_argument("--bucket-len", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--tag", default=None, help="输出文件名, 默认自动生成")
    ap.add_argument("--topk-logprobs", type=int, default=0,
                    help="额外记录每个位置的 top-K logprob 到 <out>.topK.npz (验证-回滚的残差采样用)。0=不记录")
    ap.add_argument("--neuron-override", default=None,
                    help='JSON, 追加到 override_neuron_config (仅 neuron, 实验用), 如 \'{"attn_kernel_enabled": false}\'')
    ap.add_argument("--prefix-file", default=None,
                    help="jsonl, 每行 {seq_id, token_ids}: 以给定 token 前缀 (可含已生成 token) 继续 decode, 代替文本 prompt")
    args = ap.parse_args()

    if args.backend == "neuron":
        setup_neuron_env(args.tp, args.lnc, args.cores)

    rows, meta = load_prompts(args.prompts)
    print(f"[data] {len(rows)} 条, checksum {meta['checksum']}")
    prefixes = None
    if args.prefix_file:
        # 前缀续写模式: 只跑 prefix 文件里列出的 seq_id, prompt 换成 token 前缀。
        # 前缀里已生成的 token 对 vLLM 只是更长的 prompt (走 context-encoding 图),
        # 输出行的 prompt_len = 前缀长度, recompute/analyze 的对齐逻辑不变。
        prefixes = {}
        with open(args.prefix_file) as f:
            for l in f:
                if l.strip():
                    o = json.loads(l); prefixes[o["seq_id"]] = list(o["token_ids"])
        rows = [r for r in rows if r["seq_id"] in prefixes]
        print(f"[prefix] {len(rows)} 条来自 {args.prefix_file}")

    from vllm import SamplingParams
    llm = build_llm(args.backend, args.model, args.tp, args.bucket_len,
                    max_logprobs=max(20, args.topk_logprobs),
                    neuron_override=json.loads(args.neuron_override) if args.neuron_override else None)

    # *** 采样参数不可更改 ***
    # 任何 top_p/top_k 截断都会改变分布, 使两侧 logprob 不可比。
    # temperature=1.0 保证采样自模型的真实分布。
    sp = SamplingParams(
        n=1,
        max_tokens=args.max_new,
        # 0: 只要被采样 token 自身的 logprob。K>0: 采样 token + top-K (vLLM 返回 K+1 项)。
        # 这只影响返回什么, 不影响采样分布 (采样仍是全词表)。
        logprobs=args.topk_logprobs,
        seed=C.SEED,
        **C.SAMPLING,
    )

    t0 = time.perf_counter()
    if prefixes is None:
        outs = llm.generate([r["prompt"] for r in rows], sp)
    else:
        from vllm.inputs import TokensPrompt
        outs = llm.generate([TokensPrompt(prompt_token_ids=prefixes[r["seq_id"]]) for r in rows], sp)
    elapsed = time.perf_counter() - t0

    tag = args.tag or f"{args.backend}_{args.model.split('/')[-1]}_tp{args.tp}_lnc{args.lnc}_b{args.bucket_len}"
    path = f"{C.SAMPLE_DIR}/{tag}.jsonl"
    os.makedirs(C.SAMPLE_DIR, exist_ok=True)

    n_trunc = 0
    with open(path, "w") as f:
        f.write(json.dumps({"_meta": {
            "backend": args.backend, "model": args.model, "tp": args.tp,
            "lnc": args.lnc, "bucket_len": args.bucket_len,
            "prompt_checksum": meta["checksum"], "elapsed_s": elapsed,
            "sampling": C.SAMPLING, "topk_logprobs": args.topk_logprobs,
            "prefix_file": args.prefix_file, "neuron_override": args.neuron_override,
        }}) + "\n")
        topk_buf = []   # (seq_idx, pos, ids[K+1], lps[K+1])

        for row, out in zip(rows, outs):
            comp = out.outputs[0]
            lps = extract_logprobs(comp)
            if args.topk_logprobs > 0:
                topk_buf.append(extract_topk(comp, args.topk_logprobs))
            if comp.finish_reason != "stop":
                n_trunc += 1
            f.write(json.dumps({
                "seq_id": row["seq_id"],
                # token_ids 是完整序列: prompt + completion。
                # recompute 侧必须喂完全相同的这一串, analyze 会逐位校验。
                "token_ids": list(out.prompt_token_ids) + list(comp.token_ids),
                "prompt_len": len(out.prompt_token_ids),
                "logprobs": lps,          # 只覆盖 completion 段
                "text": comp.text,
                "finish_reason": comp.finish_reason,
            }) + "\n")

    if args.topk_logprobs > 0:
        import numpy as np
        K = args.topk_logprobs
        seq_idx = np.concatenate([np.full(len(b[0]), i, dtype=np.int32) for i, b in enumerate(topk_buf)])
        pos = np.concatenate([np.arange(len(b[0]), dtype=np.int32) for b in topk_buf])
        ids = np.concatenate([b[0] for b in topk_buf]); lp = np.concatenate([b[1] for b in topk_buf])
        npz = f"{C.SAMPLE_DIR}/{tag}.top{K}.npz"
        # 列 0 = 被采样 token (与 jsonl 的 logprobs 逐位相同), 列 1.. = 其余 top-K 按 logprob 降序; 不足补 (-1, -inf)
        np.savez_compressed(npz, seq_idx=seq_idx, pos=pos, ids=ids, logprobs=lp,
                            seq_ids=np.array([r["seq_id"] for r in rows]))
        print(f"[topk] {ids.shape} → {npz}")
    print(f"[done] {elapsed:.1f}s, 截断 {n_trunc}/{len(rows)} → {path}")
    if n_trunc > len(rows) * 0.3:
        print("[warn] 截断率过高, 考虑增大 --max-new; 截断序列的末端 logprob "
              "不代表自然结束, 会影响位置剖面的解读")


def extract_logprobs(comp) -> list[float]:
    """vLLM 的 logprobs 结构在版本间有差异, 做防御性提取。"""
    if comp.logprobs is None:
        raise RuntimeError("引擎没有返回 logprobs, 检查 SamplingParams(logprobs=0)")
    out = []
    for tok_id, d in zip(comp.token_ids, comp.logprobs):
        e = d[tok_id]
        out.append(float(e.logprob) if hasattr(e, "logprob") else float(e))
    return out


def extract_topk(comp, K: int):
    """每个位置: 采样 token 放列 0, 其余按 logprob 降序放列 1..K。
    vLLM v1 的 dict 含采样 token + top-K, 采样 token 若本身在 top-K 里则总数为 K, 否则 K+1。"""
    import numpy as np
    n = len(comp.token_ids)
    ids = np.full((n, K + 1), -1, dtype=np.int32)
    lps = np.full((n, K + 1), -np.inf, dtype=np.float32)
    for t, (tok_id, d) in enumerate(zip(comp.token_ids, comp.logprobs)):
        val = lambda e: float(e.logprob) if hasattr(e, "logprob") else float(e)
        ids[t, 0] = tok_id; lps[t, 0] = val(d[tok_id])
        rest = sorted(((val(e), k) for k, e in d.items() if k != tok_id), reverse=True)[:K]
        for j, (v, k) in enumerate(rest, start=1):
            ids[t, j] = k; lps[t, j] = v
    return ids, lps


if __name__ == "__main__":
    main()
