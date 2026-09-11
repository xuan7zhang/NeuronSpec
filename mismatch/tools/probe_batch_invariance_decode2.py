"""decode (TKG) 图的 batch-invariance —— 用贪心解码隔离采样 RNG。
温度=0 时无随机数, 轨迹分叉只可能来自前向数值。另报带 seed 时首次分叉的位置, 以确认是 RNG 而非数值。"""
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
def run(idx, sps):
    outs = llm.generate([P[i] for i in idx], [sps[i] for i in idx], use_tqdm=False)
    return {j: (list(o.outputs[0].token_ids), SM.extract_logprobs(o.outputs[0])) for j, o in zip(idx, outs)}
def cmp(X, Y, name):
    same = sum(X[i][0] == Y[i][0] for i in X); eq=tot=0; mx=0.0; firsts=[]
    for i in X:
        a,b = X[i][0], Y[i][0]
        k = next((t for t in range(min(len(a),len(b))) if a[t]!=b[t]), None)
        if k is not None: firsts.append(k)
        n = k if k is not None else min(len(a),len(b))
        la,lb = np.array(X[i][1][:n]), np.array(Y[i][1][:n]); eq += int((la==lb).sum()); tot += n; mx = max(mx, float(np.abs(la-lb).max()) if n else 0.0)
    fs = f", 首次分叉位置 中位数 {int(np.median(firsts))} 最小 {min(firsts)}" if firsts else ""
    print(f"RESULT {name}: token 序列相同 {same}/{len(X)}; 分叉前 logprob 逐位相同 {eq}/{tot} ({100*eq/max(1,tot):.3f}%), max|Δ| {mx:.3e}{fs}")

print("--- 贪心 (temperature=0, 无 RNG): 前向路径的 batch-invariance ---")
g = [SamplingParams(n=1, max_tokens=64, logprobs=0, temperature=0.0) for _ in P]
t0=time.time(); A=run(range(32), g); print(f"[timing] batch32 {time.time()-t0:.1f}s")
Cr=run(list(reversed(range(32))), g); B={}; [B.update(run([i], g)) for i in range(32)]
cmp(A, Cr, "贪心 批32 顺序反转 [槽位依赖]")
cmp(A, B,  "贪心 批32 vs 批1   [batch-invariance]")
print("\n--- 带 seed 的采样 (C.SAMPLING): 分叉是否来自 RNG ---")
s = [SamplingParams(n=1, max_tokens=64, logprobs=0, seed=C.SEED+i, **C.SAMPLING) for i in range(len(P))]
A2=run(range(32), s); B2={}; [B2.update(run([i], s)) for i in range(32)]
cmp(A2, B2, "采样 批32 vs 批1")
