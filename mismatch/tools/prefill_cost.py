"""实测 prefill_cost = 训练图一次 prefill 的成本 / 推理侧 decode 一条序列的成本。
两侧口径不同 (训练图 1 逻辑核 B=1; NEFF tp4 4 逻辑核), 同时报 wall-clock 和 core-seconds 两种比值。
用法: python tools/prefill_cost.py train | python tools/prefill_cost.py decode"""
import sys, os, time, json
MODE = sys.argv[1]
S = "/home/ec2-user/mismatch/results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"
rows = [json.loads(l) for l in open(S)][1:9]        # 8 条
out = {}
if MODE == "train":
    sys.path.insert(0, "/home/ec2-user/mismatch"); import recompute as R
    R.setup_neuron_env(lnc=2, cores=None)
    import torch, torch_xla.core.xla_model as xm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); pad = tok.pad_token_id or tok.eos_token_id
    model, dev = R.build_model("neuron", "Qwen/Qwen3-1.7B", 1); model.eval()
    def one(r):
        ids, attn = R.pad_batch([r], 1024, pad); ids, attn = ids.to(dev), attn.to(dev)
        with torch.no_grad():
            lp = R.gather_logprobs(model(input_ids=ids, attention_mask=attn).logits, ids); xm.mark_step()
            return lp.float().cpu()
    one(rows[0]); one(rows[1])                      # warmup (编译/缓存加载)
    ts = []
    for r in rows * 3:
        t0 = time.perf_counter(); one(r); ts.append(time.perf_counter() - t0)
    out = {"train_prefill_s_per_seq_B1_T1024": sum(ts)/len(ts), "n": len(ts), "cores": 1, "min": min(ts), "max": max(ts)}
else:
    sys.path.insert(0, "/home/ec2-user/mismatch"); import sample as SM, config as C
    SM.setup_neuron_env(tp=4, lnc=2, cores=None)
    from vllm import SamplingParams
    llm = SM.build_llm("neuron", "Qwen/Qwen3-1.7B", 4, 1024)
    sp = SamplingParams(n=1, max_tokens=512, logprobs=0, seed=C.SEED, **C.SAMPLING)
    prompts = [tok["text"] if False else None for tok in rows]
    from vllm.inputs import TokensPrompt
    P = [TokensPrompt(prompt_token_ids=r["token_ids"][:r["prompt_len"]]) for r in rows]
    llm.generate(P[:1], sp)                          # warmup
    # (a) 单序列 decode 延迟: 逐条跑, max_num_seqs 已是 32 但队列里只有 1 条
    ts, ntok = [], []
    for p in P:
        t0 = time.perf_counter(); o = llm.generate([p], sp); ts.append(time.perf_counter()-t0); ntok.append(len(o[0].outputs[0].token_ids))
    out["decode_single_s_per_seq"] = sum(ts)/len(ts); out["decode_single_tokens_mean"] = sum(ntok)/len(ntok)
    out["decode_single_ms_per_token"] = 1000*sum(ts)/sum(ntok)
    # (b) batch=8 吞吐口径
    t0 = time.perf_counter(); o = llm.generate(P, sp); tb = time.perf_counter()-t0
    out["decode_batch8_s_total"] = tb; out["decode_batch8_s_per_seq"] = tb/len(P); out["decode_batch8_tokens"] = sum(len(x.outputs[0].token_ids) for x in o)
    out["cores"] = 4
print("RESULT " + json.dumps(out))
