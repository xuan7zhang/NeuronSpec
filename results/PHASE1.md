# Phase 1: Speculative Decoding Characterization on Trainium2

**Setup**: trn2.3xlarge (1 device, 4 logical cores, 96GB), tp=2, bf16.
Target llama-3.2-3B + draft llama-3.2-1B, ctx 128 / seq 256, greedy
(on-device sampling, top-k 1), NxDI fused speculation (SDK 2.25 DLC,
neuronx-cc 2.19, NxDI 0.4.7422). Date: 2026-07-22.

## 1. Baseline & step latencies

| config | ctx encode | decode / spec step | e2e generate |
|---|--:|--:|--:|
| AR b=1 | 12.9 ms | 8.63 ms/tok | 108.9 tok/s |
| AR b=4 | 27.6 ms | 8.8 ms/step | 352.0 tok/s |
| spec b=1 γ=2 | 17.7 ms | 18.3 ms | 95–109 tok/s |
| spec b=1 γ=4 | 17.7 ms | 26.4 ms | 123–150 tok/s |
| spec b=1 γ=8 | 17.7 ms | 42.4 ms | 115–186 tok/s |
| spec b=4 γ=2 | 39.0 ms | 19.9 ms (raw) | blocked (host) |
| spec b=4 γ=4 | 39.0 ms | 30.5 ms (raw) | blocked (host) |
| spec b=4 γ=8 | 39.2 ms | 54.0 ms (raw) | blocked (host) |

b=1 spec-step times measured inside the instrumented generation loop
(includes host); b=4 are raw device forwards (dummy inputs, no host loop).

## 2. Measured acceptance E[m] (b=1, greedy, per workload)

| workload | γ=2 | γ=4 | γ=8 | tok/s @ γ=2/4/8 |
|---|--:|--:|--:|---|
| repetitive | 2.00 | 4.00 | 8.00 | 108.4 / 149.9 / **186.1** |
| structured_json | 1.99 | 3.77 | 7.28 | 108.8 / 142.6 / **171.2** |
| creative | 1.98 | 3.90 | 7.68 | 108.6 / 147.5 / **179.8** |
| code | 1.93 | 3.86 | 6.75 | 95.0 / 135.7 / **144.4** |
| open_ended | 1.91 | 3.55 | 6.03 | 104.5 / 133.7 / **139.8** |
| instruction | 1.86 | 3.25 | 4.89 | 100.3 / **122.8** / 114.6 |

## 3. Break-even acceptance (E[m] required to match AR)

E[m]* = spec_step_ms / ar_decode_ms.

| | γ=2 | γ=4 | γ=8 |
|---|--:|--:|--:|
| b=1 (AR 8.63ms) | **2.12 > γ — impossible** | 3.06 (77%) | 4.91 (61%) |
| b=4 (AR 8.8ms) | **2.26 > γ — impossible** | 3.47 (87%) | 6.14 (77%) |

**Consistency check**: instruction @ b=1 γ=8 — measured E[m]=4.89 vs
required 4.91 → predicted ≈ tie; measured 114.6 vs AR 108.9–116 tok/s. ✓

## 4. Findings

1. **γ=2 can never win** on this pair/hardware: one spec step costs >2×
   an AR step, so even 100% acceptance loses. The floor comes from running
   the 1B draft γ times sequentially plus verify overhead.
2. **Batching is nearly free for AR** (8.63 → 8.8 ms going b=1→4: decode
   is memory-bound on weights), but spec-step cost grows with B
   (26.4 → 30.5 ms at γ=4). Consequently the break-even bar *rises* with
   batch size (77% → 87% at γ=4). Speculation and batching compete for
   the same slack on Trainium2.
3. **Optimal γ is workload-dependent**: γ=8 wins up to +60–70% on
   high-acceptance workloads but falls to break-even/below on
   instruction-type. γ=4 is the safest static choice; no static γ is
   optimal across the board — direct motivation for per-request
   speculation-length control (compile-aware 2D bucketing).
4. **Batched speculation is blocked in software, not hardware**: the
   device graph compiles and runs at b=4 (raw step measured), but two
   host layers assume batch=1 — transformers'
   `assisted generate is only supported for batch_size = 1` assert, and
   NxDI's `_fused_assisted_decoding` collapsing per-request acceptance
   into a scalar (`torch.ops.aten.Int(n_matches)`, uniform
   `accepted_tokens[:, :n_matches]` slicing). Correct ragged host commit
   = the NeuronSpec opportunity (RQ2/RQ3).

## 5. Caveats

- b=4 spec-step numbers exclude host-loop overhead; b=1 include it.
  Expect the b=4 e2e picture to be worse than the raw analysis until a
  batched host loop exists.
- Acceptance measured at b=1 with greedy sampling and 6 short prompts;
  E[m] is roughly request-local so should transfer to batch, but needs
  confirmation once a batched loop exists.
- Same-family draft (1B for 3B) is a favorable pairing; acceptance will
  be lower for cross-family or heavily quantized drafts.
- Single measurement run per cell; no error bars yet.

## Next (per research plan)

- Batched host loop prototype with per-request commit pointers
  (transactional KV, Phase 2) → unlock real b>1 speculation e2e.
- Repeat acceptance with sampling (T>0) and longer/realistic prompts.
- Draft/verify time split via profiler (currently fused = opaque).
