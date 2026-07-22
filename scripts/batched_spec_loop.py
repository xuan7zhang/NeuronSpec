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
    p.add_argument("--mode", choices=["spec", "ar", "compare"], default="spec")
    p.add_argument("--spec-trace")
    p.add_argument("--ar-trace")
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--json-out", default=None)
    p.add_argument("--spec-json", default=None, help="compare mode: spec results file")
    p.add_argument("--ar-json", default=None, help="compare mode: AR reference file")
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

        print("== batched fused-spec loop ==")
        res = batched_spec_generate(spec_model, spec_adapter, tokenizer,
                                    PROMPTS, args.max_new)
        n_rounds = len(res["rounds"])
        total_new = sum(len(c) for c in res["committed_tokens"])
        total_ms = res["ctx_ms"] + sum(r["ms"] for r in res["rounds"])
        print(f"rounds={n_rounds} new_tokens={total_new} "
              f"{total_new / (total_ms / 1000):.1f} tok/s (batch aggregate)")
        for i, t in enumerate(res["texts"]):
            print(f"--- req[{i}] ({len(res['committed_tokens'][i])} toks) ---\n{t[:160]}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"spec_trace": args.spec_trace, "batched": res}, f, indent=2)
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
