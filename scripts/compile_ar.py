"""Compile an AR (non-speculative) NxDI trace, reusing a saved config as template.

Usage:
  python compile_ar.py --template /home/ubuntu/traced_model/rq1_ar \
      --output /home/ubuntu/traced_model/rq1_ar_b1 --batch-size 1 --seq-len 256
"""
import argparse
import json
import sys
import types

# Qwen3 import shim (see run_spec.py)
_stub = types.ModuleType("neuronx_distributed_inference.models.qwen3.modeling_qwen3")
_stub.NeuronQwen3ForCausalLM = None
sys.modules["neuronx_distributed_inference.models.qwen3.modeling_qwen3"] = _stub

from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, help="dir with a neuron_config.json to reuse")
    p.add_argument("--output", required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--max-context-length", type=int, default=None)
    args = p.parse_args()

    saved = json.load(open(args.template.rstrip("/") + "/neuron_config.json"))
    nc_kwargs = saved["neuron_config"]
    # strip derived/stale fields; NeuronConfig recomputes them
    for k in ["max_batch_size", "ctx_batch_size", "tkg_batch_size", "max_length",
              "buckets", "n_active_tokens", "bucket_n_active_tokens",
              "kv_cache_batch_size", "spec_batch_size", "pa_num_blocks"]:
        nc_kwargs.pop(k, None)
    if args.batch_size is not None:
        nc_kwargs["batch_size"] = args.batch_size
    if args.seq_len is not None:
        nc_kwargs["seq_len"] = args.seq_len
    if args.max_context_length is not None:
        nc_kwargs["max_context_length"] = args.max_context_length

    model_path = args.model_path or saved["_name_or_path"]
    config_cls = NeuronLlamaForCausalLM.get_config_cls()
    neuron_config = config_cls.get_neuron_config_cls()(**nc_kwargs)
    config = config_cls(neuron_config, load_config=load_pretrained_config(model_path))

    print(f"model={model_path} b={neuron_config.batch_size} "
          f"ctx={neuron_config.max_context_length} seq={neuron_config.seq_len}")
    model = NeuronLlamaForCausalLM(model_path, config)
    print("compiling ->", args.output)
    model.compile(args.output)
    print("done")


if __name__ == "__main__":
    main()
