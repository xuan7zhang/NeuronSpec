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

import torch
from transformers import AutoTokenizer, GenerationConfig

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

    tokenizer = AutoTokenizer.from_pretrained(compiled, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = False
    generation_config.pad_token_id = tokenizer.pad_token_id
    max_new = args.max_new_tokens or (nc.seq_len - nc.max_context_length)
    generation_config.max_new_tokens = max_new

    # ---- generation (correctness eyeball check) ----
    prompts = args.prompts or ["I believe the meaning of life is"]
    # replicate prompts to fill the compiled batch size
    prompts = (prompts * nc.batch_size)[: nc.batch_size]
    gen_model = HuggingFaceGenerationAdapter(model)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    t0 = time.monotonic()
    outputs = gen_model.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask,
        generation_config=generation_config,
    )
    dt = time.monotonic() - t0
    new_tokens = int((outputs.shape[1] - inputs.input_ids.shape[1]) * outputs.shape[0])
    print(f"\ngenerate: {dt*1000:.1f} ms, {new_tokens} new tokens, "
          f"{new_tokens/dt:.1f} tok/s (batch={outputs.shape[0]})")
    for i, text in enumerate(tokenizer.batch_decode(outputs, skip_special_tokens=True)):
        print(f"--- output[{i}] ---\n{text}")

    # ---- benchmark ----
    if args.benchmark:
        from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
        print("\nBenchmarking...")
        report = benchmark_sampling(model, None, generation_config)
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
