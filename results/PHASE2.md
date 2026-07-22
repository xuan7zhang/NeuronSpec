# Phase 2: Correct Batched Speculation + Continuous Batching on Trainium2

Same setup as [PHASE1.md](PHASE1.md) (trn2.3xlarge, 3B target + 1B draft,
tp=2, bf16, ctx128/seq256, greedy). Dates: 2026-07-22/23.

## What was built

1. **Correct batched fused-speculation host loop** (`batched_spec_loop.py`,
   v2): keeps per-request acceptance (no scalar collapse), ragged attention
   mask pinned to the largest KV bucket. **4/4 requests exactly match the
   AR-b4 reference token-for-token**, including an early-EOS request.
   Host cost: 0.34 ms/round (~1%) — vectorization made host overhead a
   non-issue (correcting the day-1 hypothesis).
2. **Continuous batching (slot refill)**: `compile_cb.py` builds fused-spec
   traces with `is_continuous_batching + ctx_batch_size=1`. Freed slots are
   refilled via single-slot context encode (~20 ms) addressed by `seq_ids`.
   Slot-reuse correctness verified (first-4 exact vs non-CB run; all reused
   slots coherent). Two integration traps documented: app-level
   `_is_prefill` misroutes mixed batches (call submodules directly);
   CB fused-spec warmup throws an ignorable scatter/gather warning.

## Results (12-request mixed workload)

| config | agg tok/s | E[m]_eff | step ms | note |
|---|--:|--:|--:|---|
| AR b=4 | **455** | – | 8.8 | batching ~free (memory-bound) |
| spec γ4, no refill | 314 | 3.23 (live) | 30.9 | 25% dead slots |
| spec γ4 + refill | 346 | 2.85 | 30.2 | +10%; ctx interrupts 7.5% |
| spec γ8 + refill | 270 | 3.82 | 52.7 | step cost kills it |
| **γ-routed pools** (γ4:instr, γ8:struct) | 311 | – | – | **hypothesis refuted** |

γ-routing loses to static γ4: γ8's b=4 step costs 1.72× γ4's, so it needs
E[m]>6.4; measured effective E[m] was 4.17 — b=1 acceptance (7.3–8.0)
does not transfer to b=4 refill (slot drain + queue wait). Even
device-only, γ8 (317 tok/s) < γ4 (377).

## The Phase-2 finding

**On Trainium2 with a 3B target, no speculation configuration beats plain
AR at b=4** — AR decode is memory-bound so batching is free (8.63→8.8 ms
b=1→4), while the spec step cost scales with batch (26→31 ms γ4, 42→53 ms
γ8). Speculation pays only at low batch occupancy:

| batch | best spec | AR | winner |
|--:|--:|--:|---|
| 1 | 119–186 tok/s (γ4/γ8 by workload) | 109 | **spec, +10–70%** |
| 4 | 346 (γ4+refill) | 455 | **AR, +31%** |

Runtime design implication: the profitable control knob is **phase-aware
AR↔spec switching driven by batch occupancy** (serve spec when the batch
is shallow, AR when full), with γ selection as a secondary, b=1-only
lever. This replaces the original "2D bucketing wins at any batch"
hypothesis (RQ3) with a sharper, measured claim — and it should invert
for larger targets (8B+), where AR decode is slower and the draft/target
gap wider: the b* crossover point is model-size dependent.

## Caveats

- Single run per cell; short prompts; greedy only.
- γ-routed pools executed sequentially on one device (measures γ-matching
  efficiency, not a live co-scheduler).
- 3B/1B is an unusually AR-friendly pair; conclusions about b=4 should
  not be extrapolated to 8B+ targets without measurement (next step).
