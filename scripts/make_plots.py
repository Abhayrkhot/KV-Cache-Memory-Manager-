"""Render README charts from results/*.json, in light and dark variants.

python scripts/make_plots.py   # writes docs/img/*-{light,dark}.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e0",
        "neutral": "#d4d3cd",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#2c2c2a",
        "neutral": "#45443f",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}
ALLOC_LABELS = {
    "paged-16": "Paged (16-token blocks)",
    "reserve-max": "Reserve max_model_len",
    "reserve-pow2": "Reserve next power of 2",
    "reserve-oracle": "Reserve exact length (oracle)",
}


def _axes(theme: dict, w: float, h: float):  # type: ignore[no-untyped-def]
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=9)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, theme: dict, title: str, subtitle: str, name: str) -> None:  # type: ignore[no-untyped-def]
    """Lay out title, subtitle and legend in a fixed header band, then save."""
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / h))  # tight_layout already makes room for the legend
    x0 = ax.get_position().x0
    fig.text(x0, 1 - 0.12 / h, title, fontsize=12, color=theme["text"], fontweight="bold", va="top")
    fig.text(x0, 1 - 0.42 / h, subtitle, fontsize=9, color=theme["muted"], va="top")
    fig.savefig(OUT / name, facecolor=theme["surface"])
    plt.close(fig)


def memory_breakdown(sim: dict, theme: dict, mode: str) -> None:
    sc = sim["scenarios"]["llama-2-13b@a100-40g"]
    allocs = ["paged-16", "reserve-oracle", "reserve-pow2", "reserve-max"]
    cats = [
        ("token_states", "Live token K/V", theme["series"][0]),
        ("reserved", "Reserved, not yet used", theme["series"][1]),
        ("internal_frag", "Internal fragmentation", theme["series"][2]),
        ("external_frag", "External fragmentation", theme["series"][3]),
        ("free", "Free", theme["neutral"]),
    ]
    fig, ax = _axes(theme, 8.6, 3.5)
    ys = range(len(allocs))[::-1]
    for y, a in zip(ys, allocs, strict=True):
        left = 0.0
        br = sc["allocators"][a]["breakdown"]
        for key, label, color in cats:
            v = 100 * br[key]
            ax.barh(
                y,
                v,
                left=left,
                height=0.5,
                color=color,
                edgecolor=theme["surface"],
                linewidth=2,
                label=label if y == ys[0] else None,
            )
            left += v
        ax.text(
            100 * br["token_states"] / 2,
            y,
            f"{100 * br['token_states']:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="#ffffff",
            fontweight="bold",
        )
    ax.set_yticks(list(ys), [ALLOC_LABELS[a] for a in allocs], color=theme["text"], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    leg = ax.legend(
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        frameon=False,
        fontsize=8,
        handlelength=1,
        handleheight=1,
        labelcolor=theme["muted"],
        columnspacing=1.2,
    )
    for h in leg.legend_handles:
        if h is not None:
            h.set_edgecolor(theme["surface"])
    _finish(
        fig,
        ax,
        theme,
        "Where the KV-cache memory goes",
        "Llama-2-13B on A100-40GB, 2,000 ShareGPT requests; mean share of the pool while memory-bound",
        f"memory-breakdown-{mode}.png",
    )


def batch_size(sim: dict, theme: dict, mode: str) -> None:
    scen = [
        ("llama-2-13b@a100-40g", "Llama-2-13B · A100-40GB"),
        ("llama-3.1-8b@l4-24g", "Llama-3.1-8B · L4-24GB"),
    ]
    allocs = ["paged-16", "reserve-oracle", "reserve-pow2", "reserve-max"]
    fig, ax = _axes(theme, 8.6, 3.6)
    h = 0.34
    for j, (key, label) in enumerate(scen):
        vals = [sim["scenarios"][key]["allocators"][a]["mean_batch_saturated"] for a in allocs]
        ys = [
            len(allocs) - 1 - i + (h / 2 + 0.02 if j == 0 else -(h / 2 + 0.02))
            for i in range(len(allocs))
        ]
        ax.barh(
            ys,
            vals,
            height=h,
            color=theme["series"][j],
            label=label,
            edgecolor=theme["surface"],
            linewidth=1,
        )
        for y, v in zip(ys, vals, strict=True):
            ax.text(v + 1.2, y, f"{v:.0f}", va="center", fontsize=8.5, color=theme["text"])
    top = max(
        a["mean_batch_saturated"]
        for k, _ in scen
        for a in sim["scenarios"][k]["allocators"].values()
    )
    ax.set_xlim(0, top * 1.1)
    ax.set_yticks(
        range(len(allocs))[::-1],
        [ALLOC_LABELS[a] for a in allocs],
        color=theme["text"],
        fontsize=9.5,
    )
    ax.legend(
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        frameon=False,
        fontsize=8.5,
        handlelength=1,
        handleheight=1,
        labelcolor=theme["muted"],
    )
    _finish(
        fig,
        ax,
        theme,
        "Concurrent requests the KV pool sustains",
        "Mean running batch while memory is the binding limit (higher is better)",
        f"batch-size-{mode}.png",
    )


def kv_quant(ppl: dict, theme: dict, mode: str) -> None:
    fmts = ["bf16", "fp8", "fp8-calibrated", "fp8-kchannel", "int8", "int8-kchannel"]
    labels = {
        "bf16": "BF16",
        "fp8": "FP8, scale 1.0",
        "fp8-calibrated": "FP8, calibrated per layer",
        "fp8-kchannel": "FP8, per-channel keys",
        "int8": "INT8, per-token scale",
        "int8-kchannel": "INT8, per-channel keys",
    }
    models = ppl["models"]
    fig, ax = _axes(theme, 8.6, 4.0)
    h = 0.36
    for j, m in enumerate(models):
        by = {r["kv_format"]: r["ppl_delta_pct"] for r in m["results"]}
        vals = [by[f] for f in fmts]
        ys = [
            len(fmts) - 1 - i + (h / 2 + 0.02 if j == 0 else -(h / 2 + 0.02))
            for i in range(len(fmts))
        ]
        ax.barh(
            ys,
            vals,
            height=h,
            color=theme["series"][j],
            label=m["model"].split("/")[-1],
            edgecolor=theme["surface"],
            linewidth=1,
        )
        for y, v in zip(ys, vals, strict=True):
            ax.text(max(v, 0) + 0.06, y, f"{v:+.2f}%", va="center", fontsize=8, color=theme["text"])
    ax.set_yticks(
        range(len(fmts))[::-1], [labels[f] for f in fmts], color=theme["text"], fontsize=9.5
    )
    ax.axvline(0, color=theme["muted"], linewidth=0.8)
    ax.set_xlim(0, max(r["ppl_delta_pct"] for m in models for r in m["results"]) * 1.18)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        frameon=False,
        fontsize=8.5,
        handlelength=1,
        handleheight=1,
        labelcolor=theme["muted"],
    )
    _finish(
        fig,
        ax,
        theme,
        "Perplexity cost of storing the KV cache in fewer bits",
        "WikiText-2 test, 24 x 1,024-token windows, relative to an FP32 cache (lower is better)",
        f"kv-quant-{mode}.png",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = json.loads((ROOT / "results/allocator_sim.json").read_text())
    ppl = json.loads((ROOT / "results/kv_quant_ppl.json").read_text())
    for mode, theme in THEMES.items():
        memory_breakdown(sim, theme, mode)
        batch_size(sim, theme, mode)
        kv_quant(ppl, theme, mode)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
