"""Per-channel key outliers: how far the largest key channel's peak sits above the median channel's.

For each layer and KV head, take each key channel's absolute maximum over a prompt and
divide the largest by the median. Reports the worst layers per model and writes
``results/key_outliers.json``.

    python scripts/key_outliers.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"]


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"
    )
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    out = {"tokens": 1024, "source": "WikiText-2 test, first 1,024 tokens", "models": {}}
    for repo in MODELS:
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
        ids = torch.tensor([tok(text).input_ids[:1024]])
        with torch.no_grad():
            cache = model(ids, use_cache=True).past_key_values
        layers = []
        for i, layer in enumerate(cache.layers):
            k = layer.keys[0].abs()  # [kv_heads, tokens, head_dim]
            ch = k.amax(dim=1)  # [kv_heads, head_dim]
            ratio = (ch.amax(-1) / ch.median(-1).values).max().item()
            layers.append({"layer": i, "max_over_median": ratio, "key_absmax": k.max().item()})
        worst = sorted(layers, key=lambda x: -x["max_over_median"])
        out["models"][repo] = {"worst": worst[:3], "all": layers}
        print(repo, [(x["layer"], round(x["max_over_median"], 1)) for x in worst[:3]])
    (ROOT / "results/key_outliers.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
