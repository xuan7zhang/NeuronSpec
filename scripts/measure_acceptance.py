"""Measure per-round acceptance (n_matches) and latency for a b=1 fused-spec trace.

Replicates NxDI's HuggingFaceGenerationAdapter._fused_assisted_decoding loop
with instrumentation: per speculation round we record how many draft tokens the
target accepted (n_matches in [1, gamma]) and the round latency. This turns the
break-even analysis from an upper bound into measured E[m] per workload type.

Usage (inside container):
  python measure_acceptance.py --compiled-model-path /home/ubuntu/traced_model/p0_spec_b1 \
      [--json-out results/raw/acceptance_g4.json]
"""
import argparse
import json
import sys
import time
import types

# Qwen3 import shim (see run_spec.py)
_stub = types.ModuleType("neuronx_distributed_inference.models.qwen3.modeling_qwen3")
_stub.NeuronQwen3ForCausalLM = None
sys.modules["neuronx_distributed_inference.models.qwen3.modeling_qwen3"] = _stub

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.random import set_random_seed

# Workloads across the expected acceptance spectrum
PROMPTS = {
    "repetitive": "Count from 1 to 100: 1, 2, 3, 4, 5, 6, 7, 8,",
    "structured_json": (
        'Generate a JSON list of 10 users with fields name, age, city. '
        '[{"name": "Alice", "age": 30, "city": "Paris"}, {"name": "Bob",'
    ),
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "instruction": "Explain step by step how to make a cup of tea:\nStep 1:",
    "open_ended": "I believe the meaning of life is",
    "creative": "Write a short story about a dragon who is afraid of heights.",
}


def run_prompt(model, adapter, tokenizer, prompt, max_len):
    nc = model.config.neuron_config
    spec_len = nc.speculation_length
    inputs = tokenizer([prompt], return_tensors="pt")
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    model.reset()
    sampling_params = prepare_sampling_params(
        batch_size=1, top_k=[1], top_p=[1.0], temperature=[1.0])
    kwargs = {
        "attention_mask": inputs.attention_mask,
        "sampling_params": sampling_params,
    }
    model_inputs = adapter.prepare_inputs_for_generation(input_ids, **kwargs)

    t0 = time.perf_counter()
    outputs = adapter(**model_inputs)
    ctx_ms = (time.perf_counter() - t0) * 1000

    new_token = outputs.fused_outputs[0][:, 0].view(1, 1)
    returned_ids = new_token
    incremental_len = 0
    rounds = []
    eos_ids = set(
        model.config.eos_token_id if isinstance(model.config.eos_token_id, list)
        else [model.config.eos_token_id])

    while prompt_len + returned_ids.shape[1] < max_len - spec_len:
        kwargs = adapter._update_model_kwargs_for_fused_generation(
            outputs, kwargs, incremental_len)
        model_inputs = adapter.prepare_inputs_for_generation(returned_ids, **kwargs)

        t0 = time.perf_counter()
        outputs = adapter(**model_inputs)
        step_ms = (time.perf_counter() - t0) * 1000

        accepted_padded = outputs.fused_outputs[0]
        next_pos_ids = outputs.fused_outputs[3]
        n_matches = int(next_pos_ids - model_inputs["position_ids"])
        incremental_len = n_matches
        accepted = accepted_padded[:, :n_matches]
        rounds.append({"n_matches": n_matches, "ms": round(step_ms, 3)})

        hit_eos = any(int(t) in eos_ids for t in accepted[0])
        returned_ids = torch.cat((returned_ids, accepted), dim=1)
        if hit_eos:
            break

    text = tokenizer.decode(returned_ids[0], skip_special_tokens=True)
    total_new = int(returned_ids.shape[1])
    total_ms = ctx_ms + sum(r["ms"] for r in rounds)
    n_rounds = len(rounds)
    return {
        "prompt_len": prompt_len,
        "new_tokens": total_new,
        "rounds": n_rounds,
        "ctx_ms": round(ctx_ms, 2),
        "mean_step_ms": round(sum(r["ms"] for r in rounds) / max(n_rounds, 1), 3),
        "mean_accept": round(sum(r["n_matches"] for r in rounds) / max(n_rounds, 1), 3),
        "accept_hist": {str(k): sum(1 for r in rounds if r["n_matches"] == k)
                        for k in range(1, model.config.neuron_config.speculation_length + 1)},
        "tok_per_s": round(total_new / (total_ms / 1000), 1),
        "round_detail": rounds,
        "output_preview": text[-200:],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compiled-model-path", required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    set_random_seed(0)
    compiled = args.compiled_model_path.rstrip("/") + "/"
    config_cls = NeuronLlamaForCausalLM.get_config_cls()
    config = config_cls.load(compiled)
    nc = config.neuron_config
    assert nc.batch_size == 1 and nc.enable_fused_speculation, \
        "acceptance measurement requires a b=1 fused-spec trace"

    model_path = args.model_path or config._name_or_path
    model = NeuronLlamaForCausalLM(model_path, config)
    print("loading...")
    model.load(compiled)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter = HuggingFaceGenerationAdapter(model)

    results = {"trace": compiled, "gamma": nc.speculation_length,
               "seq_len": nc.seq_len, "workloads": {}}
    for name, prompt in PROMPTS.items():
        r = run_prompt(model, adapter, tokenizer, prompt, nc.seq_len)
        results["workloads"][name] = r
        print(f"{name:16s} E[m]={r['mean_accept']:.2f}/{nc.speculation_length} "
              f"step={r['mean_step_ms']:.1f}ms rounds={r['rounds']} "
              f"{r['tok_per_s']:.1f} tok/s")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
