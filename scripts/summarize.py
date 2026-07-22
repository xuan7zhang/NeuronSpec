"""Summarize results/raw/*.json into the Phase-1 (B, gamma) table."""
import glob
import json
import os
import sys


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def fmt(x, nd=1):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def main(results_dir="results/raw"):
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        bench = r.get("benchmark") or {}
        # fused-spec b>1 raw reports use fused_spec_model@kv<N> keys
        step_cols = {k: get(v, "latency_ms_p50")
                     for k, v in bench.items() if k.startswith("fused_spec_model")}
        rows.append({
            "trace": name,
            "B": r.get("batch_size"),
            "gamma": r.get("speculation_length"),
            "fused": r.get("fused_spec"),
            "gen_tok_s": r.get("tok_per_s"),
            "e2e_p50_ms": get(bench, "e2e_model", "latency_ms_p50"),
            "ctx_p50_ms": get(bench, "context_encoding_model", "latency_ms_p50"),
            "decode_p50_ms": get(bench, "token_generation_model", "latency_ms_p50"),
            **step_cols,
        })

    cols = ["trace", "B", "gamma", "gen_tok_s", "e2e_p50_ms", "ctx_p50_ms", "decode_p50_ms"]
    extra = sorted({k for row in rows for k in row if k.startswith("fused_spec_model")})
    cols += extra

    widths = {c: max(len(c), *(len(fmt(row.get(c)) if not isinstance(row.get(c), str)
                                else row.get(c)) for row in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-|-".join("-" * widths[c] for c in cols))
    for row in sorted(rows, key=lambda r: (r["B"] or 0, r["gamma"] or 0)):
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append((v if isinstance(v, str) else fmt(v)).ljust(widths[c]))
        print(" | ".join(cells))


if __name__ == "__main__":
    main(*sys.argv[1:])
