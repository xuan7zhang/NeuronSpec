"""Compile a fused-speculation trace with continuous batching enabled.

ctx_batch_size=1 (single-slot context encoding, seq_ids-addressed KV) +
spec/tkg batch 4 — the executable pair needed for slot refill.

Usage:
  python compile_cb.py --template /home/ubuntu/traced_model/p0_spec \
      --output /home/ubuntu/traced_model/cb_g4b4
"""
import argparse
import json
import sys
import types

_stub = types.ModuleType("neuronx_distributed_inference.models.qwen3.modeling_qwen3")
_stub.NeuronQwen3ForCausalLM = None
sys.modules["neuronx_distributed_inference.models.qwen3.modeling_qwen3"] = _stub

from neuronx_distributed_inference.models.config import FusedSpecNeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

DERIVED = ["max_batch_size", "ctx_batch_size", "tkg_batch_size", "max_length",
           "buckets", "n_active_tokens", "bucket_n_active_tokens",
           "kv_cache_batch_size", "spec_batch_size", "pa_num_blocks"]


def clean(nc):
    for k in DERIVED:
        nc.pop(k, None)
    return nc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, help="fused-spec trace dir to reuse")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--speculation-length", type=int, default=None)
    args = p.parse_args()

    saved = json.load(open(args.template.rstrip("/") + "/neuron_config.json"))
    nc_kwargs = clean(saved["neuron_config"])
    draft = saved["fused_spec_config"]["draft_config"]
    draft_nc_kwargs = clean(draft["neuron_config"])
    draft_path = saved["fused_spec_config"]["draft_model_path"]
    model_path = saved["_name_or_path"]

    if args.batch_size:
        nc_kwargs["batch_size"] = args.batch_size
        draft_nc_kwargs["batch_size"] = args.batch_size
    if args.speculation_length:
        nc_kwargs["speculation_length"] = args.speculation_length

    # continuous batching: single-slot ctx encode, batched decode/spec
    for kw in (nc_kwargs, draft_nc_kwargs):
        kw["is_continuous_batching"] = True
        kw["ctx_batch_size"] = 1

    config_cls = NeuronLlamaForCausalLM.get_config_cls()
    nc_cls = config_cls.get_neuron_config_cls()

    draft_nc = nc_cls(**draft_nc_kwargs)
    draft_nc.speculation_length = 0
    draft_nc.enable_fused_speculation = False
    draft_config = config_cls(draft_nc, load_config=load_pretrained_config(draft_path))

    neuron_config = nc_cls(**nc_kwargs)
    config = config_cls(neuron_config, load_config=load_pretrained_config(model_path))
    config.fused_spec_config = FusedSpecNeuronConfig(
        NeuronLlamaForCausalLM._model_cls,
        draft_config=draft_config,
        draft_model_path=draft_path,
    )

    print(f"target={model_path} draft={draft_path} b={neuron_config.batch_size} "
          f"ctx_b={neuron_config.ctx_batch_size} spec={neuron_config.speculation_length} "
          f"cb={neuron_config.is_continuous_batching}")
    model = NeuronLlamaForCausalLM(model_path, config)
    print("compiling ->", args.output)
    model.compile(args.output)
    print("done")


if __name__ == "__main__":
    main()
