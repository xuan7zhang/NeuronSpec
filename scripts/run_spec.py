"""Load a compiled NxDI trace (AR or fused-speculation) and run generation + benchmark.

Rebuilt runner for the NeuronSpec experiments. Works around the broken
`inference_demo` CLI in the DLC (transformers 4.48 lacks Qwen3, which
inference_demo imports at module level) by importing only the llama stack.

Usage (inside neuronmm:repro container):
  python run_spec.py --compiled-model-path /home/ubuntu/traced_model/sw_g2_b1 \
      [--model-path /home/ubuntu/models/llama-3.2-3b] [--prompt "..."] \
      [--benchmark] [--max-new-tokens N]

The config (batch size, speculation length, buckets, ...) is reconstructed
entirely from the trace's neuron_config.json — no need to re-specify shapes.
"""
import argparse
import json
import time

import sys
import types

import torch
from transformers import AutoTokenizer, GenerationConfig

# Shim: the DLC ships transformers 4.48 (no Qwen3), but NxDI's utils/constants
# imports NeuronQwen3ForCausalLM at module level, which then fails on
# `from transformers import Qwen3ForCausalLM`. Pre-seed a stub module so the
# import never happens (we only use llama).
_qwen3_stub = types.ModuleType("neuronx_distributed_inference.models.qwen3.modeling_qwen3")
_qwen3_stub.NeuronQwen3ForCausalLM = None
sys.modules["neuronx_distributed_inference.models.qwen3.modeling_qwen3"] = _qwen3_stub

from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.random import set_random_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-model-path", required=True)
    parser.add_argument("--model-path", default=None,
                        help="HF checkpoint dir; defaults to _name_or_path from the saved config")
    parser.add_argument("--prompt", dest="prompts", action="append", default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="write results as JSON to this path")
    args = parser.parse_args()

    set_random_seed(args.seed)

    compiled = args.compiled_model_path.rstrip("/") + "/"
    config_cls = NeuronLlamaForCausalLM.get_config_cls()
    config = config_cls.load(compiled)
    nc = config.neuron_config

    model_path = args.model_path or config._name_or_path
    print(f"trace           : {compiled}")
    print(f"target model    : {model_path}")
    print(f"batch_size      : {nc.batch_size}")
    print(f"ctx/seq         : {nc.max_context_length}/{nc.seq_len}")
    print(f"speculation_len : {nc.speculation_length}")
    print(f"fused_spec      : {nc.enable_fused_speculation}")
    print(f"tp_degree       : {nc.tp_degree}")

    model = NeuronLlamaForCausalLM(model_path, config)
    t0 = time.monotonic()
    print("\nLoading model to Neuron...")
    model.load(compiled)
    print(f"load time: {time.monotonic() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = False
    generation_config.pad_token_id = tokenizer.pad_token_id
    max_new = args.max_new_tokens or (nc.seq_len - nc.max_context_length)
    generation_config.max_new_tokens = max_new

    # ---- generation (correctness eyeball check) ----
    # HF's generate() refuses assisted decoding with batch>1 — the exact
    # batch-size-1 limitation of the official speculation path. For those
    # traces we can only benchmark raw forwards (benchmark_sampling).
    skip_generate = nc.enable_fused_speculation and nc.batch_size > 1
    dt, new_tokens = None, None
    if skip_generate:
        print("\nNOTE: fused spec with batch>1 — HF assisted generate unsupported, "
              "skipping generation, benchmark only")
    prompts = args.prompts or ["I believe the meaning of life is"]
    # replicate prompts to fill the compiled batch size
    prompts = (prompts * nc.batch_size)[: nc.batch_size]
    gen_model = HuggingFaceGenerationAdapter(model)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    if not skip_generate:
        gen_kwargs = {}
        if nc.enable_fused_speculation:
            # Triggers the adapter's _fused_assisted_decoding path (see NxDI utils/accuracy.py)
            gen_kwargs["prompt_lookup_num_tokens"] = nc.speculation_length

        t0 = time.monotonic()
        outputs = gen_model.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask,
            generation_config=generation_config, **gen_kwargs,
        )
        dt = time.monotonic() - t0
        new_tokens = int((outputs.shape[1] - inputs.input_ids.shape[1]) * outputs.shape[0])
        print(f"\ngenerate: {dt*1000:.1f} ms, {new_tokens} new tokens, "
              f"{new_tokens/dt:.1f} tok/s (batch={outputs.shape[0]})")
        for i, text in enumerate(tokenizer.batch_decode(outputs, skip_special_tokens=True)):
            print(f"--- output[{i}] ---\n{text}")

    # ---- benchmark ----
    report = None
    if args.benchmark:
        from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
        print("\nBenchmarking...")
        report = benchmark_sampling(model, None, generation_config)
        print(json.dumps(report, indent=2, default=str))

    if args.json_out:
        result = {
            "trace": compiled,
            "batch_size": nc.batch_size,
            "max_context_length": nc.max_context_length,
            "seq_len": nc.seq_len,
            "speculation_length": nc.speculation_length,
            "fused_spec": nc.enable_fused_speculation,
            "tp_degree": nc.tp_degree,
            "generate_ms": dt * 1000 if dt else None,
            "new_tokens": new_tokens,
            "tok_per_s": new_tokens / dt if dt else None,
            "hf_generate_supported": not skip_generate,
            "benchmark": report,
        }
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
