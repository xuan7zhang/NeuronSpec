# NeuronSpec

Batched speculative decoding experiments on AWS Trainium (trn2).

- `scripts/run_spec.py` — load a compiled NxDI trace (AR or fused-spec) and run generation + benchmark
- Traces live in `/data/traced_model` (target llama-3.2-3b + draft llama-3.2-1b, tp=2, bf16, ctx128/seq256)
- Run inside `neuronmm:repro` container:
  `docker run --rm --device=/dev/neuron0 -v $PWD:/workspace/neuronspec -v /data/models:/home/ubuntu/models -v /data/traced_model:/home/ubuntu/traced_model -w /workspace/neuronspec neuronmm:repro -c "python scripts/run_spec.py --compiled-model-path /home/ubuntu/traced_model/sw_g2_b1"`
