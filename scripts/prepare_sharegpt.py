"""Turn ShareGPT conversations into (prompt_len, output_len) pairs.

Same sampling rules as vLLM's ``benchmark_serving.py``: the first human turn is
the prompt, the first model turn the output; pairs with fewer than 4 tokens on
either side, prompts over 1024 tokens, or totals over 2048 tokens are dropped.
Lengths are counted with the Llama 2 tokenizer.

    curl -L -o data/ShareGPT_V3_unfiltered_cleaned_split.json \
      https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    python scripts/prepare_sharegpt.py data/ShareGPT_V3_unfiltered_cleaned_split.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--num", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "benchmarks/workloads/sharegpt_llama2.json"))
    args = ap.parse_args()

    convs = [
        c["conversations"]
        for c in json.loads(Path(args.path).read_text())
        if len(c["conversations"]) >= 2
    ]
    random.Random(args.seed).shuffle(convs)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pairs: list[tuple[int, int]] = []
    for conv in convs:
        if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        p = len(tok(conv[0]["value"]).input_ids)
        o = len(tok(conv[1]["value"]).input_ids)
        if p < 4 or o < 4 or p > 1024 or p + o > 2048:
            continue
        pairs.append((p, o))
        if len(pairs) >= args.num:
            break
    ps, os_ = [p for p, _ in pairs], [o for _, o in pairs]
    meta = {
        "source": "anon8231489123/ShareGPT_Vicuna_unfiltered (ShareGPT_V3_unfiltered_cleaned_split.json)",
        "tokenizer": args.tokenizer,
        "filter": "first human/gpt turn; 4 <= len; prompt <= 1024; prompt + output <= 2048 (vLLM benchmark rules)",
        "num": len(pairs),
        "prompt_mean": statistics.mean(ps),
        "prompt_median": statistics.median(ps),
        "output_mean": statistics.mean(os_),
        "output_median": statistics.median(os_),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"meta": meta, "pairs": pairs}, separators=(",", ":")) + "\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
