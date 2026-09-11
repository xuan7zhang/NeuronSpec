"""补测 decode (TKG) 图的 batch-invariance: 连续生成 64 token, 批 32 vs 批 1 vs 顺序反转。
连续批处理下每步的活跃序列数不同, 这是 GPU 侧问题最严重的路径。"""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/ec2-user/mismatch")
import sample as SM
SM.setup_neuron_env(tp=4, lnc=2, cores=None)
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
import config as C
llm = SM.build_llm("neuron", "Qwen/Qwen3-1.7B", 4, 1024, max_logprobs=20)
rows = [json.loads(l) for l in open("results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl")][1:33]
P = [TokensPrompt(prompt_token_ids=r["token_ids"][: r["prompt_len"]]) for r in rows]
# 每条固定 seed, 分布相同则采样结果也相同; 轨迹分叉本身就是不变性被破坏的信号
sps = [SamplingParams(n=1, max_tokens=64, logprobs=0, seed=C.SEED + i, **C.SAMPLING) for i in range(len(P))]
def run(idx):
    outs = llm.generate([P[i] for i in idx], [sps[i] for i in idx], use_tqdm=False)
    d = {}
    for j, o in zip(idx, outs):
        c = o.outputs[0]; d[j] = (list(c.token_ids), SM.extract_logprobs(c))
    return d
t0=time.time(); A = run(range(32)); tA=time.time()-t0
t0=time.time(); B = {}; [B.update(run([i])) for i in range(32)]; tB=time.time()-t0
t0=time.time(); Cr = run(list(reversed(range(32)))); tC=time.time()-t0
print(f"[timing] batch32 {tA:.1f}s, one-by-one {tB:.1f}s, reversed {tC:.1f}s")
def cmp(X, Y, name):
    same_tok = sum(X[i][0] == Y[i][0] for i in X)
    eq = tot = 0; mx = 0.0
    for i in X:
        if X[i][0] != Y[i][0]: continue
        a, b = np.array(X[i][1]), np.array(Y[i][1]); eq += int((a == b).sum()); tot += len(a); mx = max(mx, float(np.abs(a-b).max()))
    print(f"RESULT {name}: token 序列相同 {same_tok}/{len(X)}; 相同序列上 logprob 逐位相同 {eq}/{tot} ({100*eq/max(1,tot):.3f}%), max|Δ| {mx:.3e}")
cmp(A, Cr, "decode 批32 顺序反转 [槽位依赖]")
cmp(A, B,  "decode 批32 vs 批1   [batch-invariance]")
