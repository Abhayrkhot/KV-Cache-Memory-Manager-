# KV Cache Memory Manager

## Overview

This project provides a simple **KV cache memory calculator** for
transformer‑based large language models.  During autoregressive
generation the model stores the **key** and **value** tensors for every
token it has produced so far.  This cache allows the model to reuse
previous activations instead of recomputing them, but it also causes
the memory footprint to grow linearly with the **sequence length** and
the **batch size**【459119712734931†L109-L118】.  For long context windows, the KV
cache often becomes the primary consumer of GPU VRAM, even exceeding
the memory needed to store the model weights【459119712734931†L109-L118】.

To estimate the size of the KV cache you can use the formula

```
bytes = 2 * layers * heads * head_dim * seq_len * batch_size * bytes_per_element
```

The factor of **2** accounts for storing both the key and value
tensors【459119712734931†L158-L162】.  The number of **layers**, **heads** and
`head_dim` derive from the architecture of your model.  The
`bytes_per_element` parameter depends on the numeric precision used to
store the tensors (e.g. 2 bytes for FP16, 1 byte for INT8, 0.5 bytes
for INT4).  Because LLM inference is **memory‑IO bound**, reducing the
precision can directly increase the maximum achievable batch size【555339060125421†L115-L132】.

This repository includes a Python module that implements this formula,
supports predefined specs for common models like LLaMA 7B/13B/70B and
GPT‑3, and exposes a command‑line interface to compute memory
requirements for arbitrary sequence lengths, batch sizes and dtypes.

## Why this matters

Real‑world LLM inference servers must squeeze multiple requests into
finite GPU memory.  According to Anyscale, LLM inference is
**memory‑bound**, not compute‑bound【555339060125421†L115-L132】.  The amount of GPU
memory consumed scales with both the **base model size** and the
**length of the token sequence**【555339060125421†L122-L132】.  For example, a 13B
parameter model may consume nearly **1 MB of KV cache per token**,
leaving room for only a handful of 2048‑token sequences on a 40 GB
GPU【555339060125421†L122-L132】.  Quantizing the KV cache to 8‑bit or 4‑bit can
halve or quarter the memory usage with minimal impact on model
accuracy【459119712734931†L109-L114】.  Understanding this relationship is
essential when choosing a GPU instance or designing a batching policy.

## Features

* **Memory estimation formula**: Compute KV cache size in bytes for any
  combination of layers, heads, head dimension, sequence length,
  batch size and numeric precision【459119712734931†L158-L162】.
* **Predefined model specs**: Built‑in parameters for common open
  models (LLaMA 7B/13B/70B, GPT‑3).  You can also specify your own
  architecture.
* **Multiple precisions**: Estimate memory usage for FP16 (2 bytes),
  INT8/FP8 (1 byte) and INT4/FP4 (0.5 bytes) representations.
* **Batch and sequence sweeps**: Provide comma‑separated lists of
  sequence lengths and batch sizes to see how the cache scales
  linearly with each dimension【459119712734931†L109-L118】.
* **Human‑readable or JSON output**: Print results in a human
  readable format or as JSON.
* **Test suite**: `pytest` tests verify the correctness of the
  calculations and that the per‑token estimate for LLaMA‑7B matches
  published values (~512 KB per token).

## Project structure

```text
04-kv-cache-memory-manager/
├── kv_cache.py        # Core logic and CLI
├── tests/
│   └── test_kv_cache.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

You can compute KV cache memory requirements for a predefined model:

```bash
python kv_cache.py --model llama7b --seq-lens 128,512,2048 --batch-sizes 1,8 --dtype fp16
```

Sample output:

```
seq_len=128, batch_size=1 -> 64.00 MB (67108864 bytes)
seq_len=128, batch_size=8 -> 512.00 MB (536870912 bytes)
seq_len=512, batch_size=1 -> 256.00 MB (268435456 bytes)
seq_len=512, batch_size=8 -> 2048.00 MB (2147483648 bytes)
seq_len=2048, batch_size=1 -> 1024.00 MB (1073741824 bytes)
seq_len=2048, batch_size=8 -> 8192.00 MB (8589934592 bytes)
```

Alternatively, specify the architecture manually:

```bash
python kv_cache.py --layers 48 --heads 40 --head-dim 128 \
    --seq-lens 1024 --batch-sizes 4 --dtype int8
```

And output results as JSON:

```bash
python kv_cache.py --model llama13b --seq-lens 1024 --batch-sizes 1,4,8 --dtype fp16 --json
```

## Running the tests

Run the unit tests with `pytest`:

```bash
pytest -q
```

The tests verify the correctness of the formula for a simple example
and check that the LLaMA‑7B per‑token cache size matches the
approximate 512 KB estimate.

## Resume bullet

Built a KV cache memory manager that estimates GPU memory usage for
transformer decoders using a simple formula.  Supports predefined
models and custom configurations, multiple numeric precisions and
batch/sequence sweeps, with tests validating the calculations and a
command‑line interface for interactive exploration of memory bottlenecks.
