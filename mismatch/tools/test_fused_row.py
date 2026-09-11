"""融合图验证: 标量 vs 默认图逐位相同? 首拒绝位置 vs CPU 决策一致? 行[token] vs 标量一致?"""
import sys, json, time, numpy as np
sys.path.insert(0, "/home/ec2-user/mismatch")
import recompute as R
R.setup_neuron_env(lnc=2, cores=None)
import torch, torch_xla.core.xla_model as xm
from transformers import AutoTokenizer
S = "results/samples/neuron_Qwen3-1.7B_tp4_lnc2_b1024.jsonl"; RJ = "results/recompute/neuron_Qwen3-1.7B_b1024_sNone_lnc2_tp1.jsonl"
rows = [json.loads(l) for l in open(S)][1:13]; ref = {json.loads(l)["seq_id"]: json.loads(l) for l in open(RJ) if "_meta" not in l}
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model, dev = R.build_model("neuron", "Qwen/Qwen3-1.7B", 1); model.eval()
rng = np.random.default_rng(0); L = 1024
eq = tot = 0; mx = 0.0; idx_ok = 0; row_ok = 0; n_rej = 0; ts = []
for r in rows:
    ids, attn = R.pad_batch([r], L, pad)
    p0, m = r["prompt_len"], len(r["logprobs"])
    q = np.full(L - 1, 0.0, np.float32); u = np.full(L - 1, -1.0, np.float32); seg = np.zeros(L - 1, bool)
    q[p0 - 1:p0 - 1 + m] = r["logprobs"]; u_seg = rng.random(m).astype(np.float32); u[p0 - 1:p0 - 1 + m] = u_seg; seg[p0 - 1:p0 - 1 + m] = True
    t0 = time.perf_counter()
    with torch.no_grad():
        lp, idx, row = R.forward_scalar_fused(model, ids.to(dev), attn.to(dev), torch.tensor(q).to(dev), torch.tensor(u).to(dev), torch.tensor(seg).to(dev))
        xm.mark_step(); lp = lp[0].float().cpu().numpy(); idx = int(idx.cpu()); row = row.float().cpu().numpy()
    ts.append(time.perf_counter() - t0)
    a = lp[p0 - 1:p0 - 1 + m]; b = np.array(ref[r["seq_id"]]["logprobs"], np.float32)
    eq += int((a == b).sum()); tot += m; mx = max(mx, float(np.abs(a - b).max()))
    # CPU 决策 (与 verify_worker 相同: float64)
    rr = np.exp(a.astype(np.float64) - q[p0 - 1:p0 - 1 + m].astype(np.float64)); acc = u_seg.astype(np.float64) < np.minimum(1.0, rr)
    cpu_idx = (p0 - 1 + int(np.argmin(acc))) if not acc.all() else L - 1
    idx_ok += int(cpu_idx == idx)
    if idx < L - 1:
        n_rej += 1; tokp = r["token_ids"][idx + 1]; row_ok += int(float(row[tokp]) == float(lp[idx]))
print(f"RESULT scalar bitwise {eq}/{tot} ({100*eq/tot:.3f}%) max|Δ| {mx:.3e} | first-reject idx device==cpu {idx_ok}/{len(rows)} | row[token]==scalar {row_ok}/{n_rej} (rejections {n_rej}) | time first {ts[0]:.1f}s steady {np.mean(ts[1:]):.3f}s")
