"""Prototype: CORRECT batched fused-speculation host loop (b>1).

Hypothesis: the compiled fused-spec graph is already ragged-correct — it
returns per-request next_input_ids / next_attn_mask / next_pos_ids
(fused_outputs[1..3], all shaped (b, .)) and per-request acceptance is
recoverable as n_i = next_pos_ids[i] - position_ids[i]. NxDI's stock host
loop collapses n to a scalar (batch-1 assumption); here we keep it ragged.

Correctness gate: each request's greedy output must exactly match the
b=1 AR reference (speculative decoding is lossless under greedy).

Usage (inside container):
  python batched_spec_loop.py --spec-trace /home/ubuntu/traced_model/p0_spec \
      --ar-trace /home/ubuntu/traced_model/rq1_ar_b1 \
      [--json-out results/raw/batched_loop_g4b4.json]
"""
import argparse
import json
import sys
import time
import types

_stub = types.ModuleType("neuronx_distributed_inference.models.qwen3.modeling_qwen3")
_stub.NeuronQwen3ForCausalLM = None
sys.modules["neuronx_distributed_inference.models.qwen3.modeling_qwen3"] = _stub

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.random import set_random_seed

PROMPTS = [
    "I believe the meaning of life is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain step by step how to make a cup of tea:\nStep 1:",
    'Generate a JSON list of users: [{"name": "Alice", "age": 30},',
]


def load(trace, model_path=None):
    cfg_cls = NeuronLlamaForCausalLM.get_config_cls()
    config = cfg_cls.load(trace)
    mp = model_path or config._name_or_path
    model = NeuronLlamaForCausalLM(mp, config)
    model.load(trace)
    return model, mp


def batched_spec_generate(model, adapter, tokenizer, prompts, max_new):
    """Ragged-aware batched fused-speculation loop."""
    nc = model.config.neuron_config
    bs = nc.batch_size
    assert len(prompts) == bs
    spec_len = nc.speculation_length

    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    model.reset()
    sampling_params = prepare_sampling_params(
        batch_size=bs, top_k=[1], top_p=[1.0], temperature=[1.0])
    kwargs = {"attention_mask": enc.attention_mask, "sampling_params": sampling_params}
    model_inputs = adapter.prepare_inputs_for_generation(enc.input_ids, **kwargs)

    t0 = time.perf_counter()
    outputs = adapter(**model_inputs)
    ctx_ms = (time.perf_counter() - t0) * 1000

    # first target token after context encoding, per request
    committed = [[int(outputs.fused_outputs[0][i, 0])] for i in range(bs)]
    rounds = []
    eos = set(model.config.eos_token_id if isinstance(model.config.eos_token_id, list)
              else [model.config.eos_token_id])
    finished = [False] * bs

    # per-request count of valid KV positions: prompt tokens (committed
    # tokens enter KV as they are fed back / verified)
    valid = enc.attention_mask.sum(dim=1).clone()  # (b,)

    while not all(finished) and max(len(c) for c in committed) < max_new:
        # ragged attention mask: request i attends to its first valid_i slots
        L = int(valid.max())
        mask = torch.zeros((bs, L), dtype=torch.int32)
        for i in range(bs):
            mask[i, : int(valid[i])] = 1

        # per-request input token & position come from the device outputs
        step_inputs = {
            "input_ids": outputs.fused_outputs[1],       # (b,1)
            "attention_mask": mask,                      # (b, L) ragged
            "position_ids": outputs.fused_outputs[3],    # (b,1)
            "sampling_params": sampling_params,
        }
        t0 = time.perf_counter()
        outputs = model.forward(**step_inputs)
        step_ms = (time.perf_counter() - t0) * 1000

        accepted_padded = outputs.fused_outputs[0]       # (b, spec_len)
        next_pos = outputs.fused_outputs[3]              # (b,1)
        n = (next_pos - step_inputs["position_ids"]).view(-1)  # per-request accepts
        rounds.append({"n_per_req": [int(x) for x in n], "ms": round(step_ms, 3)})

        valid += n  # per-request KV growth — ragged, no uniform slice
        for i in range(bs):
            if finished[i]:
                continue
            toks = [int(t) for t in accepted_padded[i, : int(n[i])]]
            for t in toks:
                committed[i].append(t)
                if t in eos or len(committed[i]) >= max_new:
                    finished[i] = True
                    break

    return {
        "ctx_ms": round(ctx_ms, 2),
        "rounds": rounds,
        "committed_tokens": committed,
        "texts": [tokenizer.decode(c, skip_special_tokens=True) for c in committed],
    }


def batched_spec_generate_v2(model, adapter, tokenizer, prompts, max_new):
    """Vectorized ragged loop: fixed-bucket mask, no per-token python work.

    v1 costs per round: mask rebuild via python row loop, per-token int()
    conversions, growing mask length (causes a bucket switch mid-run).
    v2: preallocated arange-comparison mask pinned to the large KV bucket,
    tensorized accept/EOS handling, per-round host/forward time split.
    """
    nc = model.config.neuron_config
    bs = nc.batch_size
    spec_len = nc.speculation_length
    # pin to the largest KV bucket for every step: constant executable,
    # constant latency (Phase-1: kv128 vs kv256 step latency ~equal)
    mask_len = max(nc.seq_len - spec_len - 2, nc.max_context_length)

    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    model.reset()
    sampling_params = prepare_sampling_params(
        batch_size=bs, top_k=[1], top_p=[1.0], temperature=[1.0])
    kwargs = {"attention_mask": enc.attention_mask, "sampling_params": sampling_params}
    model_inputs = adapter.prepare_inputs_for_generation(enc.input_ids, **kwargs)

    t0 = time.perf_counter()
    outputs = adapter(**model_inputs)
    ctx_ms = (time.perf_counter() - t0) * 1000

    eos_list = (model.config.eos_token_id if isinstance(model.config.eos_token_id, list)
                else [model.config.eos_token_id])
    eos_t = torch.tensor(eos_list, dtype=torch.int64)

    committed_buf = torch.full((bs, max_new + spec_len), -1, dtype=torch.int64)
    committed_buf[:, 0] = outputs.fused_outputs[0][:, 0]
    lens = torch.ones(bs, dtype=torch.int64)
    valid = enc.attention_mask.sum(dim=1).to(torch.int64)  # per-request KV len
    alive = torch.ones(bs, dtype=torch.bool)
    positions = torch.arange(mask_len)

    rounds = []
    while bool(alive.any()):  # per-request stopping lives in `alive`
        t_host0 = time.perf_counter()
        mask = (positions < valid.unsqueeze(1)).to(torch.int32)  # (bs, mask_len)
        step_inputs = {
            "input_ids": outputs.fused_outputs[1],
            "attention_mask": mask,
            "position_ids": outputs.fused_outputs[3],
            "sampling_params": sampling_params,
        }
        t_fwd0 = time.perf_counter()
        outputs = model.forward(**step_inputs)
        t_fwd = time.perf_counter() - t_fwd0

        accepted = outputs.fused_outputs[0].to(torch.int64)      # (bs, spec_len)
        n = (outputs.fused_outputs[3] - step_inputs["position_ids"]).view(-1).to(torch.int64)

        # accept-window mask: position j accepted iff j < n_i (and request alive)
        win = torch.arange(spec_len).unsqueeze(0) < n.unsqueeze(1)  # (bs, spec_len)
        win &= alive.unsqueeze(1)
        # EOS: cut each row's window at first EOS (inclusive)
        is_eos = (accepted.unsqueeze(-1) == eos_t).any(-1) & win
        first_eos = torch.where(is_eos.any(1), is_eos.int().argmax(1),
                                torch.full((bs,), spec_len))
        win &= torch.arange(spec_len).unsqueeze(0) <= first_eos.unsqueeze(1)
        n_eff = win.sum(1)

        # scatter accepted tokens into per-request committed buffers
        dest = lens.unsqueeze(1) + torch.arange(spec_len).unsqueeze(0)  # (bs, spec_len)
        flat_rows = torch.arange(bs).unsqueeze(1).expand_as(dest)[win]
        committed_buf[flat_rows, dest[win]] = accepted[win]
        lens += n_eff
        valid += n * alive.to(torch.int64)  # KV grows by device-side n for alive reqs
        alive &= ~is_eos.any(1)
        alive &= lens < max_new

        t_total = time.perf_counter() - t_host0
        rounds.append({"n_per_req": n.tolist(), "ms": round(t_total * 1000, 3),
                       "fwd_ms": round(t_fwd * 1000, 3),
                       "host_ms": round((t_total - t_fwd) * 1000, 3)})

    committed = [committed_buf[i, : int(lens[i])].tolist() for i in range(bs)]
    return {
        "ctx_ms": round(ctx_ms, 2),
        "rounds": rounds,
        "committed_tokens": committed,
        "texts": [tokenizer.decode(c, skip_special_tokens=True) for c in committed],
    }


REFILL_QUEUE = [  # 12 requests cycling the workload spectrum
    "I believe the meaning of life is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain step by step how to make a cup of tea:\nStep 1:",
    'Generate a JSON list of users: [{"name": "Alice", "age": 30},',
    "Count from 1 to 100: 1, 2, 3, 4, 5, 6, 7, 8,",
    "Write a short story about a dragon who is afraid of heights.",
    "The three most important principles of good software design are:",
    "SELECT name, age FROM users WHERE",
    "Once upon a time in a small village by the sea,",
    "The capital of France is Paris. The capital of Japan is",
    "import numpy as np\n\ndef softmax(x):\n",
    "To reverse a linked list, you first",
]


def refill_generate(model, tokenizer, queue, max_new):
    """Continuous-batching fused speculation: freed slots are refilled from
    the queue via single-slot context encoding (ctx_batch_size=1 trace)."""
    nc = model.config.neuron_config
    bs = nc.batch_size
    spec_len = nc.speculation_length
    mask_len = max(nc.seq_len - spec_len - 2, nc.max_context_length)
    positions_row = torch.arange(mask_len)
    sampling_1 = prepare_sampling_params(batch_size=1, top_k=[1], top_p=[1.0],
                                         temperature=[1.0])
    sampling_b = prepare_sampling_params(batch_size=bs, top_k=[1], top_p=[1.0],
                                         temperature=[1.0])
    eos_list = (model.config.eos_token_id if isinstance(model.config.eos_token_id, list)
                else [model.config.eos_token_id])
    eos = set(eos_list)

    model.reset()
    t_start = time.perf_counter()

    # per-slot state
    slot_req = [-1] * bs          # queue index served by slot
    slot_tokens = [None] * bs     # committed token list of active request
    valid = torch.zeros(bs, dtype=torch.int64)
    cur_input = torch.zeros((bs, 1), dtype=torch.int32)
    cur_pos = torch.zeros((bs, 1), dtype=torch.int32)
    next_req = 0
    done = []                     # (req_idx, tokens, ttft_ms, e2e_ms)
    req_t0 = [0.0] * bs
    ctx_calls = 0
    ctx_ms_total = 0.0

    def encode_into(slot, req_idx):
        nonlocal next_req, ctx_calls, ctx_ms_total
        enc = tokenizer([queue[req_idx]], return_tensors="pt")
        L = enc.input_ids.shape[1]
        t0 = time.perf_counter()
        # call the submodule directly: app-level routing (_is_prefill checks
        # position_ids.min()==0) misroutes mixed batches
        out = model.context_encoding_model(
            enc.input_ids.to(torch.int32), enc.attention_mask.to(torch.int32),
            torch.arange(L).unsqueeze(0).to(torch.int32),
            torch.tensor([slot], dtype=torch.int32),
            sampling_1, None, None)
        dt = (time.perf_counter() - t0) * 1000
        ctx_calls += 1
        ctx_ms_total += dt
        first = int(out[0][0, 0])
        slot_req[slot] = req_idx
        slot_tokens[slot] = [first]
        valid[slot] = L
        cur_input[slot, 0] = out[1][0, 0]
        cur_pos[slot, 0] = out[3][0, 0]
        req_t0[slot] = t0
        return dt

    for s in range(min(bs, len(queue))):
        encode_into(s, next_req)
        next_req += 1

    rounds = 0
    fwd_ms_total = 0.0
    while any(r >= 0 for r in slot_req):
        mask = (positions_row < valid.unsqueeze(1)).to(torch.int32)
        t0 = time.perf_counter()
        out = model.fused_spec_model(
            cur_input, mask, cur_pos,
            torch.arange(bs, dtype=torch.int32),
            sampling_b, None, None)
        fwd_ms_total += (time.perf_counter() - t0) * 1000
        rounds += 1

        accepted = out[0].to(torch.int64)
        n = (out[3] - cur_pos).view(-1).to(torch.int64)
        cur_input = out[1].clone()
        cur_pos = out[3].clone()

        for s in range(bs):
            if slot_req[s] < 0:
                continue
            finished = False
            for t in accepted[s, : int(n[s])].tolist():
                slot_tokens[s].append(t)
                if t in eos or len(slot_tokens[s]) >= max_new:
                    finished = True
                    break
            valid[s] += int(n[s])
            if finished:
                e2e = (time.perf_counter() - req_t0[s]) * 1000
                done.append((slot_req[s], slot_tokens[s], round(e2e, 1)))
                slot_req[s] = -1
                if next_req < len(queue):
                    encode_into(s, next_req)
                    next_req += 1
                else:
                    valid[s] = 1  # park the dead slot on a harmless state
                    cur_pos[s, 0] = 0
                    cur_input[s, 0] = 0

    wall_ms = (time.perf_counter() - t_start) * 1000
    total_tokens = sum(len(t) for _, t, _ in done)
    return {
        "requests": len(done),
        "total_new_tokens": total_tokens,
        "wall_ms": round(wall_ms, 1),
        "agg_tok_per_s": round(total_tokens / (wall_ms / 1000), 1),
        "rounds": rounds,
        "mean_fwd_ms": round(fwd_ms_total / max(rounds, 1), 2),
        "ctx_calls": ctx_calls,
        "mean_ctx_ms": round(ctx_ms_total / max(ctx_calls, 1), 2),
        "per_request": [
            {"req": r, "tokens": len(t), "e2e_ms": e,
             "text_preview": tokenizer.decode(t, skip_special_tokens=True)[:80]}
            for r, t, e in sorted(done)],
        "committed_tokens": [t for _, t, _ in sorted(done)],
    }


def ar_reference(model, adapter, tokenizer, prompt, max_new):
    """b=1 greedy AR reference output tokens."""
    enc = tokenizer([prompt], return_tensors="pt")
    model.reset()
    gen_cfg = GenerationConfig(max_new_tokens=max_new, do_sample=False, top_k=1,
                               pad_token_id=tokenizer.pad_token_id or 2)
    out = adapter.generate(enc.input_ids, attention_mask=enc.attention_mask,
                           generation_config=gen_cfg)
    return [int(t) for t in out[0][enc.input_ids.shape[1]:]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["spec", "ar", "compare", "refill"], default="spec")
    p.add_argument("--spec-trace")
    p.add_argument("--ar-trace")
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--json-out", default=None)
    p.add_argument("--spec-json", default=None, help="compare mode: spec results file")
    p.add_argument("--ar-json", default=None, help="compare mode: AR reference file")
    p.add_argument("--loop", choices=["v1", "v2"], default="v1")
    p.add_argument("--queue-indices", default=None,
                   help="refill mode: comma-separated REFILL_QUEUE indices to serve")
    args = p.parse_args()

    set_random_seed(0)

    # NxDI leaves global state behind after loading a fused-spec model, which
    # breaks loading a plain AR model in the same process — hence split modes.
    if args.mode == "spec":
        spec_model, mp = load(args.spec_trace.rstrip("/") + "/")
        tokenizer = AutoTokenizer.from_pretrained(mp, padding_side="right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        spec_adapter = HuggingFaceGenerationAdapter(spec_model)

        print(f"== batched fused-spec loop ({args.loop}) ==")
        gen_fn = batched_spec_generate_v2 if args.loop == "v2" else batched_spec_generate
        res = gen_fn(spec_model, spec_adapter, tokenizer, PROMPTS, args.max_new)
        n_rounds = len(res["rounds"])
        total_new = sum(len(c) for c in res["committed_tokens"])
        total_ms = res["ctx_ms"] + sum(r["ms"] for r in res["rounds"])
        print(f"rounds={n_rounds} new_tokens={total_new} "
              f"{total_new / (total_ms / 1000):.1f} tok/s (batch aggregate)")
        if res["rounds"] and "fwd_ms" in res["rounds"][0]:
            fwd = sum(r["fwd_ms"] for r in res["rounds"]) / n_rounds
            host = sum(r["host_ms"] for r in res["rounds"]) / n_rounds
            print(f"per-round: fwd={fwd:.2f}ms host={host:.2f}ms")
        for i, t in enumerate(res["texts"]):
            print(f"--- req[{i}] ({len(res['committed_tokens'][i])} toks) ---\n{t[:160]}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"spec_trace": args.spec_trace, "batched": res}, f, indent=2)
            print("wrote", args.json_out)

    elif args.mode == "refill":
        spec_model, mp = load(args.spec_trace.rstrip("/") + "/")
        tokenizer = AutoTokenizer.from_pretrained(mp, padding_side="right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        queue = REFILL_QUEUE
        if args.queue_indices:
            idx = [int(i) for i in args.queue_indices.split(",")]
            queue = [REFILL_QUEUE[i] for i in idx]
        print(f"== continuous-batching refill loop ({len(queue)} reqs) ==")
        res = refill_generate(spec_model, tokenizer, queue, args.max_new)
        print(f"requests={res['requests']} tokens={res['total_new_tokens']} "
              f"wall={res['wall_ms']}ms agg={res['agg_tok_per_s']} tok/s")
        print(f"rounds={res['rounds']} fwd={res['mean_fwd_ms']}ms "
              f"ctx_calls={res['ctx_calls']} ctx={res['mean_ctx_ms']}ms")
        for r in res["per_request"]:
            print(f"req[{r['req']:2d}] {r['tokens']:3d} toks e2e={r['e2e_ms']:7.1f}ms "
                  f"| {r['text_preview']!r}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"spec_trace": args.spec_trace, "refill": res}, f, indent=2)
            print("wrote", args.json_out)

    elif args.mode == "ar":
        ar_model, mp = load(args.ar_trace.rstrip("/") + "/")
        tokenizer = AutoTokenizer.from_pretrained(mp, padding_side="right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        ar_adapter = HuggingFaceGenerationAdapter(ar_model)
        nc = ar_model.config.neuron_config
        if nc.batch_size == len(PROMPTS):
            # batched AR control: same prompts, same padding as the spec loop
            enc = tokenizer(PROMPTS, return_tensors="pt", padding=True)
            gen_cfg = GenerationConfig(max_new_tokens=args.max_new, do_sample=False,
                                       top_k=1, pad_token_id=tokenizer.pad_token_id)
            out = ar_adapter.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                      generation_config=gen_cfg)
            refs = [[int(t) for t in row[enc.input_ids.shape[1]:]] for row in out]
        else:
            refs = []
            for prompt in PROMPTS:
                refs.append(ar_reference(ar_model, ar_adapter, tokenizer, prompt, args.max_new))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"ar_trace": args.ar_trace, "references": refs}, f, indent=2)
            print("wrote", args.json_out)

    else:  # compare
        spec = json.load(open(args.spec_json))["batched"]["committed_tokens"]
        refs = json.load(open(args.ar_json))["references"]
        for i, (got, ref) in enumerate(zip(spec, refs)):
            L = min(len(ref), len(got))
            first_div = next((j for j in range(L) if ref[j] != got[j]), None)
            status = "EXACT MATCH" if first_div is None else f"DIVERGES at token {first_div}"
            print(f"req[{i}]: {status} (compared {L} tokens)")
            if first_div is not None:
                lo = max(0, first_div - 3)
                print(f"   ref[{lo}:{first_div+3}] = {ref[lo:first_div+3]}")
                print(f"   got[{lo}:{first_div+3}] = {got[lo:first_div+3]}")


if __name__ == "__main__":
    main()
