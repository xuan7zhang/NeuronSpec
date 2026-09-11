"""验证器 batch>1 探针: 图内按位置分块做 log_softmax (绕开 [B*T,V] 的 HBM 估算),
对比默认 B=1 验证器图的标量是否逐位相同, 并测每序列 prefill 时间。
    python tools/probe_batched_verifier.py <B> <chunk>"""
import sys, os, json, time, numpy as np
sys.path.insert(0, "/home/ec2-user/mismatch")
B, CH = int(sys.argv[1]), int(sys.argv[2])
import recompute as R
R.setup_neuron_env(lnc=2, cores=None)
os.environ["NEURON_COMPILE_CACHE_URL"] = "/home/ec2-user/neuron-cache/train"
import torch, torch_xla.core.xla_model as xm
from transformers import AutoTokenizer

def forward_scalar_chunked(model, ids, attn, chunk):
    logits = model(input_ids=ids, attention_mask=attn).logits          # [B, L, V] bf16
    L = ids.shape[1]; outs = []
    for a in range(0, L - 1, chunk):
        b = min(a + chunk, L - 1)
        lp = torch.log_softmax(logits[:, a:b, :].float(), dim=-1)
        outs.append(torch.gather(lp, 2, ids[:, a + 1:b + 1].unsqueeze(-1)).squeeze(-1))
    return torch.cat(outs, dim=1)                                        # [B, L-1]

S = "results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"; RJ = "results/recompute/neuron_Qwen3-1.7B_b1024_sNone_lnc2_tp1.jsonl"
rows = [json.loads(l) for l in open(S)][1:17]; ref = {json.loads(l)["seq_id"]: json.loads(l) for l in open(RJ) if "_meta" not in l}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model, dev = R.build_model("neuron", "Qwen/Qwen3-1.7B", 1); model.eval()
ts, eq, mx, n = [], 0, 0.0, 0
for i in range(0, 16, B):
    chunk = rows[i:i + B]
    if len(chunk) < B: break
    ids, attn = R.pad_batch(chunk, 1024, pad); ids, attn = ids.to(dev), attn.to(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        lp = forward_scalar_chunked(model, ids, attn, CH); xm.mark_step(); lp = lp.float().cpu().numpy()
    ts.append(time.perf_counter() - t0)
    for j, r in enumerate(chunk):
        p, m = r["prompt_len"], len(r["logprobs"])
        a = lp[j, p - 1:p - 1 + m]; b = np.array(ref[r["seq_id"]]["logprobs"], dtype=np.float32)
        eq += int((a == b).sum()); n += m; mx = max(mx, float(np.abs(a - b).max()))
print(f"RESULT B={B} chunk={CH}: bitwise_equal {eq}/{n} ({100*eq/max(1,n):.3f}%), max|Δ| vs default graph {mx:.3e}, "
      f"time/batch first {ts[0]:.2f}s (含编译/加载) steady {np.mean(ts[1:]) if len(ts)>1 else float('nan'):.3f}s → per-seq {np.mean(ts[1:])/B if len(ts)>1 else float('nan'):.4f}s")
