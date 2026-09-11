"""
入口 0: 准备测试集。不需要任何加速器, 现在就能跑。

用法:
    python prepare_data.py --out data/prompts.jsonl

产出一个冻结的 prompt 集合。*** 这个文件一旦生成就不要再改 ***
所有配置、所有后端、GPU 和 Neuron 两侧都必须用同一份, 否则结果不可比。
脚本会在文件里写入 checksum, 后续步骤会校验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import config as C


PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)


def load_gsm8k(n: int, seed: int):
    """从 GSM8K 测试集取 n 条。

    用 datasets 库; 没有网络时可以改成读本地文件。GSM8K 很小(几 MB),
    建议提前下好放进仓库, 避免上机后还要联网。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:n])          # 排序让结果与 shuffle 实现无关

    out = []
    for i in idx:
        row = ds[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append({
            "seq_id": f"gsm8k-{i:05d}",       # 稳定 id, 跨后端对齐靠它
            "prompt": PROMPT_TEMPLATE.format(question=row["question"].strip()),
            "gold": gold,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{C.DATA_DIR}/prompts.jsonl")
    ap.add_argument("--n", type=int, default=C.N_PROMPTS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    rows = load_gsm8k(args.n, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    checksum = hashlib.sha1(body.encode()).hexdigest()[:16]

    with open(args.out, "w") as f:
        f.write(json.dumps({"_meta": {
            "checksum": checksum, "n": len(rows),
            "seed": args.seed, "source": "gsm8k/test",
        }}) + "\n")
        f.write(body + "\n")

    print(f"{len(rows)} 条 → {args.out}")
    print(f"checksum: {checksum}   (后续步骤会校验这个值)")


def load_prompts(path: str):
    """供 sample.py / recompute.py 使用。"""
    with open(path) as f:
        meta = json.loads(f.readline())["_meta"]
        rows = [json.loads(l) for l in f if l.strip()]
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    got = hashlib.sha1(body.encode()).hexdigest()[:16]
    if got != meta["checksum"]:
        raise ValueError(
            f"prompt 文件 checksum 不匹配 (期望 {meta['checksum']}, 实际 {got})。"
            "文件被改过, 结果将不可比。"
        )
    return rows, meta


if __name__ == "__main__":
    main()
