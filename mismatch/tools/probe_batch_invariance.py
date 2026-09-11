"""Trainium 的 NEFF decode/prefill 是否 batch-invariant?
这是 GPU 侧确定性推理文献(TML/SGLang batch-invariant kernels)要解决的核心问题。
同一批前缀, 三种提交方式, 比较 top-256 logprob 是否逐位相同:
  A) 一次提交 32 条   B) 一条一条提交   C) 一次提交 32 条但顺序反转 (测槽位依赖)
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/ec2-user/mismatch")
import sample as SM
SM.setup_neuron_env(tp=4, lnc=2, cores=None)
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
import config as C
llm = SM.build_llm("neuron", "Qwen/Qwen3-1.7B", 4, 1024, max_logprobs=256)
S = "results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"
rows = [json.loads(l) for l in open(S)][1:33]
P = [TokensPrompt(prompt_token_ids=r["token_ids"][: r["prompt_len"] + 128]) for r in rows]
sp = SamplingParams(n=1, max_tokens=1, logprobs=256, seed=C.SEED, **C.SAMPLING)

def rows_of(outs):
    out = []
    for o in outs:
        ids, lps = SM.extract_topk(o.outputs[0], 256)
        order = np.argsort(ids[0]); out.append((ids[0][order], lps[0][order]))   # 按 token id 排序便于比较
    return out

t0 = time.time(); A = rows_of(llm.generate(P, sp, use_tqdm=False)); tA = time.time()-t0
t0 = time.time(); B = [rows_of(llm.generate([p], sp, use_tqdm=False))[0] for p in P]; tB = time.time()-t0
t0 = time.time(); Crev = rows_of(llm.generate(P[::-1], sp, use_tqdm=False))[::-1]; tC = time.time()-t0
t0 = time.time(); A2 = rows_of(llm.generate(P, sp, use_tqdm=False)); tA2 = time.time()-t0

def cmp(X, Y, name):
    eq = tot = 0; mx = 0.0; idmismatch = 0
    for (ix, lx), (iy, ly) in zip(X, Y):
        if not np.array_equal(ix, iy): idmismatch += 1; continue
        eq += int((lx == ly).sum()); tot += len(lx); mx = max(mx, float(np.abs(lx.astype(np.float64)-ly.astype(np.float64)).max()))
    print(f"RESULT {name}: top-256 集合不同的序列 {idmismatch}/{len(X)}; 其余 logprob 逐位相同 {eq}/{tot} ({100*eq/max(1,tot):.3f}%), max|Δ| {mx:.3e}")
print(f"[timing] batch32 {tA:.1f}s, one-by-one {tB:.1f}s, reversed {tC:.1f}s, batch32 again {tA2:.1f}s")
cmp(A, A2,   "A vs A2   (同样提交方式, 重跑)      [对照: 应完全相同]")
cmp(A, B,    "A vs B    (批 32  vs  批 1)        [batch-invariance]")
cmp(A, Crev, "A vs C    (批 32, 顺序反转)         [槽位依赖]")
