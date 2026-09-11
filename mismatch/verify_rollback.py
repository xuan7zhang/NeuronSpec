"""
验证-回滚采样主循环 (VERIFY_ROLLBACK.md §2/§3/§7-5)。

进程: 1 个 decode worker (vLLM, core 0..tp-1) + W 个验证 worker (训练图, core tp..tp+W-1) + 编排器 (CPU), 文件 IPC。
    python verify_rollback.py run --run R --tp 2 --lnc 2 --chunk 64 --n 512 --verifiers 2 --lanes 2

工程项 (相对 v1):
  --verifiers W   数据并行验证器: 每个逻辑核一个 B=1 训练图进程 (batch>1 被编译器 HBM 估算挡住)。
  --lanes K       流水线: 序列分成 K 条 lane, 一条在验证时其它 lane 在 decode。
  --fused         融合图: 一次 prefill 同时给出标量、图内首拒绝位置和该位置的整行 (省掉拒绝时的第二次 prefill)。
                  CPU 仍用 float64 做接受判定; 设备给的位置与 CPU 不一致时回退到行 prefill (计数 n_fallback)。

接受规则: r_t = p_t(a_t)/q_t(a_t), 以 min(1, r_t) 接受; 拒绝时从残差 (p_t − q_t)⁺ 采 a'_t
(q_t 在 top-256 外视为 0, 误差 = q 在 top-256 外的质量, 按位置记录), 之后的 token 作废,
从 [prefix, a'_t] 重新 decode。验证器就是 recompute.py 的训练图。

产出 results/vr/<run>/final.jsonl (与 sample.py 同格式, logprobs = 验证器给的 p_t(a_t)), stats.json, events.jsonl。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import config as C

TOPK = 256
IPC_POLL = 0.2


# ============================================================================
# IPC 小工具
# ============================================================================

def _wait_for(path: str, stop_path: str | None = None):
    while not os.path.exists(path):
        if stop_path and os.path.exists(stop_path):
            return False
        time.sleep(IPC_POLL)
    return True


def _atomic_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _atomic_npz(path: str, **arrs):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


# ============================================================================
# decode worker (vLLM-Neuron)
# ============================================================================

def decode_worker(args):
    ipc = f"{args.root}/ipc"
    import sample as SM
    SM.setup_neuron_env(tp=args.tp, lnc=args.lnc, cores=None)      # cores 0..tp-1, 必须在 import vllm 前
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    llm = SM.build_llm("neuron", args.model, args.tp, args.bucket_len, max_logprobs=TOPK)
    _atomic_json(f"{ipc}/decode_ready.json", {"pid": os.getpid()})
    n = 0
    while True:
        req_path = f"{ipc}/dec_req_{n}.json"
        if not _wait_for(req_path, f"{ipc}/stop"):
            break
        req = json.load(open(req_path))
        t0 = time.perf_counter()
        try:
            prompts = [TokensPrompt(prompt_token_ids=r["prefix"]) for r in req["items"]]
            # *** 采样参数不可更改 *** (C.SAMPLING); 每个请求换 seed, 续写是新鲜的 q 采样
            sps = [SamplingParams(n=1, max_tokens=r["max_tokens"], logprobs=TOPK,
                                  seed=C.SEED * 1000 + n * 7919 + i, **C.SAMPLING)
                   for i, r in enumerate(req["items"])]
            outs = llm.generate(prompts, sps, use_tqdm=False)
            items, ids_all, lp_all, seg = [], [], [], []
            for r, o in zip(req["items"], outs):
                comp = o.outputs[0]
                ids, lps = SM.extract_topk(comp, TOPK)
                items.append({"seq_id": r["seq_id"], "tokens": list(comp.token_ids),
                              "q_lp": [float(x) for x in lps[:, 0]], "finish_reason": comp.finish_reason,
                              "off": int(sum(seg)), "n": len(comp.token_ids)})
                seg.append(len(comp.token_ids)); ids_all.append(ids); lp_all.append(lps)
            _atomic_npz(f"{ipc}/dec_resp_{n}.npz",
                        ids=np.concatenate(ids_all) if ids_all else np.zeros((0, TOPK + 1), np.int32),
                        lps=np.concatenate(lp_all) if lp_all else np.zeros((0, TOPK + 1), np.float32))
            _atomic_json(f"{ipc}/dec_resp_{n}.json",
                         {"items": items, "elapsed": time.perf_counter() - t0, "n_tokens": int(sum(seg))})
        except Exception:  # noqa
            import traceback
            _atomic_json(f"{ipc}/dec_resp_{n}.json", {"error": traceback.format_exc()})
        n += 1


# ============================================================================
# verify worker (训练图)
# ============================================================================

def verify_worker(args):
    ipc = f"{args.root}/ipc"; wid = args.worker_id
    import recompute as R
    R.setup_neuron_env(lnc=args.lnc, cores=str(args.tp + wid))      # decode 占 0..tp-1, 验证器 wid 用 core tp+wid
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model, device = R.build_model("neuron", args.model, 1)
    model.eval()
    L = args.bucket_len
    vocab = None

    def _inputs(token_ids):
        ids, attn = R.pad_batch([{"token_ids": token_ids}], L, pad_id)
        return ids.to(device), attn.to(device)

    def prefill_scalar(token_ids):
        ids, attn = _inputs(token_ids)
        with torch.no_grad():
            lp = R.forward_scalar(model, ids, attn); xm.mark_step()
            return lp[0].float().cpu().numpy()                          # [L-1]

    def prefill_row(token_ids, lp_index):
        ids, attn = _inputs(token_ids)
        with torch.no_grad():
            lp, rows = R.forward_rows(model, ids, attn, [lp_index], 1); xm.mark_step()
            return lp[0].float().cpu().numpy(), rows[0].float().cpu().numpy().astype(np.float64)

    def prefill_fused(token_ids, q_vec, u_vec, seg_vec):
        ids, attn = _inputs(token_ids)
        with torch.no_grad():
            lp, idx, row = R.forward_scalar_fused(model, ids, attn, torch.tensor(q_vec).to(device),
                                                  torch.tensor(u_vec).to(device), torch.tensor(seg_vec).to(device))
            xm.mark_step()
            return lp[0].float().cpu().numpy(), int(idx.cpu()), row.float().cpu().numpy().astype(np.float64)

    # 预热: 图都编译/加载好, 编译时间不算进开销
    warm = [1] * 64
    if args.fused:
        prefill_fused(warm, np.zeros(L - 1, np.float32), np.full(L - 1, -1.0, np.float32), np.zeros(L - 1, bool))
    else:
        prefill_scalar(warm)
    prefill_row(warm, 10)
    _atomic_json(f"{ipc}/verify_ready_{wid}.json", {"pid": os.getpid()})

    n = 0
    while True:
        req_path = f"{ipc}/ver_req_{wid}_{n}.json"
        if not _wait_for(req_path, f"{ipc}/stop"):
            break
        req = json.load(open(req_path))
        z = np.load(f"{ipc}/ver_req_{wid}_{n}.npz")
        t0 = time.perf_counter()
        try:
            results = []; cnt = {"n_scalar": 0, "n_row": 0, "n_fused": 0, "n_fallback": 0, "t_scalar": 0.0, "t_row": 0.0, "t_fused": 0.0}
            for it in req["items"]:
                rng = np.random.default_rng([C.SEED, int(it["rng_key"]), int(it["dec_n"])])
                full = it["prefix"] + it["tokens"]
                p0, k0 = it["prompt_len"], len(it["prefix"]) - it["prompt_len"]   # k0: 新段在 completion 里的起点
                m = len(it["tokens"]); lo = p0 - 1 + k0
                q_lp = np.array(it["q_lp"], dtype=np.float64)
                u = rng.random(m)
                dev_idx = row = None
                if args.fused:
                    q_vec = np.zeros(L - 1, np.float32); u_vec = np.full(L - 1, -1.0, np.float32); seg = np.zeros(L - 1, bool)
                    q_vec[lo:lo + m] = q_lp; u_vec[lo:lo + m] = u; seg[lo:lo + m] = True
                    ts = time.perf_counter(); lp, dev_idx, row = prefill_fused(full, q_vec, u_vec, seg)
                    cnt["t_fused"] += time.perf_counter() - ts; cnt["n_fused"] += 1
                else:
                    ts = time.perf_counter(); lp = prefill_scalar(full); cnt["t_scalar"] += time.perf_counter() - ts; cnt["n_scalar"] += 1
                p_lp = lp[lo:lo + m].astype(np.float64)
                r = np.exp(p_lp - q_lp)
                acc = u < np.minimum(1.0, r)                       # CPU float64 决策是权威
                rej = int(np.argmin(acc)) if not acc.all() else -1
                out = {"seq_id": it["seq_id"], "p_lp": p_lp.tolist(), "r": r.tolist(), "u": u.tolist(),
                       "n_accept": m if rej < 0 else rej, "rejected_at": rej}
                if rej >= 0:
                    lp_index = lo + rej
                    if not (args.fused and dev_idx == lp_index):   # 非融合, 或设备的 fp32 决策与 CPU 不一致 -> 行 prefill
                        if args.fused:
                            cnt["n_fallback"] += 1
                        ts = time.perf_counter(); lp2, row = prefill_row(full, lp_index); cnt["t_row"] += time.perf_counter() - ts; cnt["n_row"] += 1
                        assert float(lp2[lp_index]) == float(lp[lp_index]), "行图与标量图不一致"
                    assert float(row[it["tokens"][rej]]) == float(lp[lp_index]), "行[token] 与标量不一致"
                    if vocab is None:
                        vocab = row.shape[0]
                    j = it["off"] + rej
                    q_ids, q_lps = z["ids"][j], z["lps"][j].astype(np.float64)
                    valid = q_ids >= 0
                    q_full = np.zeros(vocab); q_full[q_ids[valid]] = np.exp(q_lps[valid])   # top-256 外视为 0
                    p_full = np.exp(row)
                    resid = np.maximum(0.0, p_full - q_full)
                    Z = resid.sum()
                    a_new = int(rng.choice(vocab, p=resid / Z))
                    out.update({"a_new": a_new, "a_new_p_lp": float(row[a_new]),
                                "q_out_mass": float(1.0 - q_full.sum()),        # 近似误差 (规格 §2)
                                "tv_at_reject": float(Z),                          # Σ(p−q)⁺ = TV(p,q)
                                "a_new_in_top256": bool(((q_ids == a_new) & valid).any()),
                                "resid_mass_outside_top256": float(resid[q_full == 0].sum() / Z)})
                results.append(out)
            _atomic_json(f"{ipc}/ver_resp_{wid}_{n}.json", {"items": results, "elapsed": time.perf_counter() - t0, **cnt})
        except Exception:  # noqa
            import traceback
            _atomic_json(f"{ipc}/ver_resp_{wid}_{n}.json", {"error": traceback.format_exc()})
        n += 1


# ============================================================================
# orchestrator: K 条 lane 的状态机, 单线程轮询
# ============================================================================

def orchestrate(args):
    from prepare_data import load_prompts
    from transformers import AutoTokenizer
    ipc = f"{args.root}/ipc"
    os.makedirs(ipc, exist_ok=True)
    rows, meta = load_prompts(args.prompts)
    rows = rows[: args.n]
    tok = AutoTokenizer.from_pretrained(args.model)
    T = args.max_new
    Cchunk = args.chunk if args.chunk > 0 else T
    W, K = args.verifiers, args.lanes
    print(f"[orch] {len(rows)} 条, T={T}, C={Cchunk}, verifiers={W}, lanes={K}, fused={args.fused}; 等 worker 就绪...", flush=True)
    _wait_for(f"{ipc}/decode_ready.json")
    for w in range(W):
        _wait_for(f"{ipc}/verify_ready_{w}.json")
    print("[orch] workers ready", flush=True)

    S = {}
    for i, r in enumerate(rows):
        S[r["seq_id"]] = {"idx": i, "prompt": tok(r["prompt"], add_special_tokens=False)["input_ids"], "acc": [], "p_lp": [],
                          "src": [], "done": False, "finish": None, "n_rej": 0, "wasted": 0, "generated": 0, "lane": i % K}
    ev = open(f"{args.root}/events.jsonl", "w")
    tot = {"dec_calls": 0, "dec_tokens": 0, "dec_elapsed": 0.0, "ver_calls": 0, "ver_elapsed": 0.0,
           "n_scalar": 0, "n_row": 0, "n_fused": 0, "n_fallback": 0, "t_scalar": 0.0, "t_row": 0.0, "t_fused": 0.0,
           "tested": 0, "accepted": 0, "sum_min1r": 0.0, "sum_rho": 0.0,
           "rejections": 0, "q_out_mass": [], "tv": [], "resid_out": [], "a_new_in_top256": 0}
    dec_n = 0; ver_n = [0] * W
    lanes = [{"id": k, "phase": "idle", "dec": None, "ver": [], "vitems": None, "rounds": 0} for k in range(K)]
    t_start = time.perf_counter()

    def issue_decode(lane):
        nonlocal dec_n
        active = [sid for sid, s in S.items() if s["lane"] == lane["id"] and not s["done"]]
        if not active:
            lane["phase"] = "finished"; return
        items = [{"seq_id": sid, "prefix": S[sid]["prompt"] + S[sid]["acc"], "max_tokens": int(min(Cchunk, T - len(S[sid]["acc"])))} for sid in active]
        _atomic_json(f"{ipc}/dec_req_{dec_n}.json", {"lane": lane["id"], "items": items})
        lane["phase"] = "decoding"; lane["dec"] = dec_n; dec_n += 1

    def on_decode_done(lane):
        n = lane["dec"]
        resp = json.load(open(f"{ipc}/dec_resp_{n}.json"))
        if "error" in resp:
            raise SystemExit("decode worker error:\n" + resp["error"])
        z = np.load(f"{ipc}/dec_resp_{n}.npz")
        tot["dec_calls"] += 1; tot["dec_tokens"] += resp["n_tokens"]; tot["dec_elapsed"] += resp["elapsed"]
        vitems = []
        for it in resp["items"]:
            s = S[it["seq_id"]]; s["generated"] += it["n"]
            if it["n"] == 0:
                s["done"] = True; s["finish"] = "empty"; continue
            vitems.append({"seq_id": it["seq_id"], "prefix": s["prompt"] + s["acc"], "prompt_len": len(s["prompt"]),
                           "tokens": it["tokens"], "q_lp": it["q_lp"], "off": it["off"], "n": it["n"], "rng_key": s["idx"],
                           "dec_n": n, "finish_reason": it["finish_reason"]})
        lane["vitems"] = {v["seq_id"]: v for v in vitems}
        if not vitems:
            issue_decode(lane); return
        # 拆到 W 个验证器: 按 item 轮转, 每个 worker 一份 npz 切片 (重排 off)
        lane["ver"] = []
        for w in range(W):
            mine = vitems[w::W]
            if not mine:
                continue
            ids_w, lps_w, off = [], [], 0; sub = []
            for v in mine:
                ids_w.append(z["ids"][v["off"]:v["off"] + v["n"]]); lps_w.append(z["lps"][v["off"]:v["off"] + v["n"]])
                sub.append({**v, "off": off}); off += v["n"]
            _atomic_npz(f"{ipc}/ver_req_{w}_{ver_n[w]}.npz", ids=np.concatenate(ids_w), lps=np.concatenate(lps_w))
            _atomic_json(f"{ipc}/ver_req_{w}_{ver_n[w]}.json", {"items": sub})
            lane["ver"].append((w, ver_n[w])); ver_n[w] += 1
        lane["phase"] = "verifying"

    def on_verify_done(lane):
        for w, n in lane["ver"]:
            vresp = json.load(open(f"{ipc}/ver_resp_{w}_{n}.json"))
            if "error" in vresp:
                raise SystemExit(f"verify worker {w} error:\n" + vresp["error"])
            for k in ("n_scalar", "n_row", "n_fused", "n_fallback", "t_scalar", "t_row", "t_fused"):
                tot[k] += vresp[k]
            tot["ver_calls"] += 1; tot["ver_elapsed"] += vresp["elapsed"]
            for o in vresp["items"]:
                s = S[o["seq_id"]]; v = lane["vitems"][o["seq_id"]]; na = o["n_accept"]; m = len(o["r"])
                tested = m if o["rejected_at"] < 0 else o["rejected_at"] + 1
                tot["tested"] += tested; tot["accepted"] += na
                r = np.array(o["r"][:tested]); tot["sum_min1r"] += float(np.minimum(1, r).sum()); tot["sum_rho"] += float(np.maximum(0, 1 - r).sum())
                s["acc"] += v["tokens"][:na]; s["p_lp"] += o["p_lp"][:na]; s["src"] += [0] * na
                if o["rejected_at"] >= 0:
                    s["n_rej"] += 1; tot["rejections"] += 1
                    s["wasted"] += m - na - 1
                    s["acc"].append(o["a_new"]); s["p_lp"].append(o["a_new_p_lp"]); s["src"].append(1)
                    tot["q_out_mass"].append(o["q_out_mass"]); tot["tv"].append(o["tv_at_reject"])
                    tot["resid_out"].append(o["resid_mass_outside_top256"]); tot["a_new_in_top256"] += int(o["a_new_in_top256"])
                    ev.write(json.dumps({"dec_n": v["dec_n"], "seq_id": o["seq_id"], "pos": len(s["acc"]) - 1, "r": o["r"][o["rejected_at"]],
                                         "u": o["u"][o["rejected_at"]], "a_old": v["tokens"][o["rejected_at"]], "a_new": o["a_new"],
                                         "tv": o["tv_at_reject"], "q_out_mass": o["q_out_mass"]}) + "\n")
                    if o["a_new"] == tok.eos_token_id:
                        s["done"] = True; s["finish"] = "stop"
                elif v["finish_reason"] == "stop":
                    s["done"] = True; s["finish"] = "stop"
                if len(s["acc"]) >= T:
                    s["done"] = True; s["finish"] = s["finish"] or "length"
        lane["rounds"] += 1
        n_done = sum(s["done"] for s in S.values())
        print(f"[orch] lane {lane['id']} round {lane['rounds']}: done {n_done}/{len(S)}, dec_calls {tot['dec_calls']}, "
              f"rejections {tot['rejections']}, fallback {tot['n_fallback']}, t={time.perf_counter()-t_start:.0f}s", flush=True)
        issue_decode(lane)

    for lane in lanes:
        issue_decode(lane)
    while any(l["phase"] != "finished" for l in lanes):
        progressed = False
        for lane in lanes:
            if lane["phase"] == "decoding" and os.path.exists(f"{ipc}/dec_resp_{lane['dec']}.json"):
                on_decode_done(lane); progressed = True
            elif lane["phase"] == "verifying" and all(os.path.exists(f"{ipc}/ver_resp_{w}_{n}.json") for w, n in lane["ver"]):
                on_verify_done(lane); progressed = True
        if not progressed:
            time.sleep(IPC_POLL)
        if sum(l["rounds"] for l in lanes) > args.max_rounds * K:
            print("[orch] max_rounds reached", flush=True); break
    ev.close()
    open(f"{ipc}/stop", "w").close()
    wall = time.perf_counter() - t_start

    with open(f"{args.root}/final.jsonl", "w") as f:
        f.write(json.dumps({"_meta": {"backend": "neuron-verify-rollback", "model": args.model, "tp": args.tp, "lnc": args.lnc,
                                      "bucket_len": args.bucket_len, "chunk": Cchunk, "prompt_checksum": meta["checksum"],
                                      "sampling": C.SAMPLING, "elapsed_s": wall, "verifiers": W, "lanes": K, "fused": args.fused}}) + "\n")
        for sid, s in S.items():
            f.write(json.dumps({"seq_id": sid, "token_ids": s["prompt"] + s["acc"], "prompt_len": len(s["prompt"]),
                                "logprobs": s["p_lp"], "src": s["src"], "text": tok.decode(s["acc"]),
                                "finish_reason": s["finish"] or "length", "n_rejections": s["n_rej"],
                                "wasted_decode_tokens": s["wasted"], "generated_tokens": s["generated"]}) + "\n")
    acc_len = sum(len(s["acc"]) for s in S.values())
    ideal_dec = tot["dec_elapsed"] * acc_len / max(1, tot["dec_tokens"])     # 纯 decode 只生成被接受 token 的忙时
    stats = {
        "n_sequences": len(S), "T": T, "chunk": Cchunk, "tp": args.tp, "lnc": args.lnc,
        "verifiers": W, "lanes": K, "fused": args.fused,
        "rounds": sum(l["rounds"] for l in lanes), "wall_s": wall,
        "tested_tokens": tot["tested"], "accept_rate_empirical": tot["accepted"] / max(1, tot["tested"]),
        "accept_rate_theoretical": tot["sum_min1r"] / max(1, tot["tested"]),
        "lambda_empirical": tot["sum_rho"] / max(1, tot["tested"]),
        "rejections_total": tot["rejections"], "rejections_per_seq": tot["rejections"] / len(S),
        "seqs_zero_rejection": sum(s["n_rej"] == 0 for s in S.values()) / len(S),
        "accepted_tokens_total": acc_len, "decode_tokens_total": tot["dec_tokens"],
        "decode_overhead": (tot["dec_tokens"] - acc_len) / acc_len,
        "prefills_scalar": tot["n_scalar"], "prefills_row": tot["n_row"], "prefills_fused": tot["n_fused"], "fused_fallbacks": tot["n_fallback"],
        "prefills_per_seq_total": (tot["n_scalar"] + tot["n_row"] + tot["n_fused"]) / len(S),
        "decode_busy_s": tot["dec_elapsed"], "verify_busy_s": tot["ver_elapsed"],
        "verify_scalar_s": tot["t_scalar"], "verify_row_s": tot["t_row"], "verify_fused_s": tot["t_fused"],
        # 资源口径 (忙时之和) 与延迟口径 (端到端 wall) 两个开销
        "prefill_overhead_measured_wall": tot["ver_elapsed"] / max(1e-9, ideal_dec),
        "total_overhead_measured_wall": (tot["dec_elapsed"] + tot["ver_elapsed"]) / max(1e-9, ideal_dec) - 1,
        "total_overhead_wallclock": wall / max(1e-9, ideal_dec) - 1,
        "q_out_mass_at_rejections": {k: float(v) for k, v in zip(("mean", "p50", "p99", "max"),
                                     (np.mean(tot["q_out_mass"]), np.percentile(tot["q_out_mass"], 50), np.percentile(tot["q_out_mass"], 99), np.max(tot["q_out_mass"])))} if tot["q_out_mass"] else None,
        "resid_mass_outside_top256_mean": float(np.mean(tot["resid_out"])) if tot["resid_out"] else None,
        "tv_at_rejections_mean": float(np.mean(tot["tv"])) if tot["tv"] else None,
        "a_new_in_top256_frac": tot["a_new_in_top256"] / max(1, tot["rejections"]),
        "finish": {k: sum(s["finish"] == k for s in S.values()) for k in ("stop", "length", "empty", None)},
    }
    json.dump(stats, open(f"{args.root}/stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"\n→ {args.root}/final.jsonl", flush=True)


# ============================================================================

def run_all(args):
    os.makedirs(f"{args.root}/ipc", exist_ok=True); os.makedirs(f"{args.root}/logs", exist_ok=True)
    common = ["--run", args.run, "--tp", str(args.tp), "--lnc", str(args.lnc), "--model", args.model,
              "--bucket-len", str(args.bucket_len)] + (["--fused"] if args.fused else [])
    procs = [subprocess.Popen([sys.executable, "-u", __file__, "decode-worker"] + common,
                              stdout=open(f"{args.root}/logs/decode-worker.log", "w"), stderr=subprocess.STDOUT)]
    for w in range(args.verifiers):
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, "verify-worker", "--worker-id", str(w)] + common,
                                      stdout=open(f"{args.root}/logs/verify-worker-{w}.log", "w"), stderr=subprocess.STDOUT))
    try:
        orchestrate(args)
    finally:
        open(f"{args.root}/ipc/stop", "w").close()
        for p in procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "orchestrate", "decode-worker", "verify-worker"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=C.MODELS[0])
    ap.add_argument("--prompts", default=f"{C.DATA_DIR}/prompts.jsonl")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--lnc", type=int, default=2)
    ap.add_argument("--bucket-len", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--chunk", type=int, default=0, help="块长 C; 0 = C=T (采完再验)")
    ap.add_argument("--n", type=int, default=C.N_PROMPTS)
    ap.add_argument("--max-rounds", type=int, default=400)
    ap.add_argument("--verifiers", type=int, default=1, help="数据并行验证器数 (各占一个逻辑核 tp..tp+W-1)")
    ap.add_argument("--lanes", type=int, default=1, help="流水线 lane 数")
    ap.add_argument("--fused", action="store_true", help="融合图: 标量+首拒绝位置+整行 一次 prefill")
    ap.add_argument("--worker-id", type=int, default=0)
    args = ap.parse_args()
    args.root = f"results/vr/{args.run}"
    {"run": run_all, "orchestrate": orchestrate, "decode-worker": decode_worker, "verify-worker": verify_worker}[args.mode](args)


if __name__ == "__main__":
    main()
