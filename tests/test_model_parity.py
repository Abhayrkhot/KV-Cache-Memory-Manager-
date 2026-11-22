"""The paged model against Hugging Face transformers on tiny random models (float32, CPU).

Covers: prefill logits, batched greedy decoding with ragged prompts, prefix-cache
hits, chunked prefill, copy-on-write after fork, sliding-window attention, swap
out/in, and quantized KV storage.
"""

from __future__ import annotations

import logging
import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pagedkv.generate import PagedGenerator  # noqa: E402
from pagedkv.models.llama import PagedLlama  # noqa: E402

pytestmark = pytest.mark.hf
logging.getLogger("transformers").setLevel(logging.ERROR)

BS = 4
VOCAB = 97


def _tiny(model_type: str, **extra):  # type: ignore[no-untyped-def]
    common = {
        "vocab_size": VOCAB,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
    }
    common.update(extra)
    cfg = transformers.AutoConfig.for_model(model_type, **common)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = transformers.AutoModelForCausalLM.from_config(cfg).float().eval()
    return hf, PagedLlama.from_hf_model(hf)


MODELS = {
    "llama": lambda: _tiny("llama"),
    "llama3-rope": lambda: _tiny(
        "llama",
        rope_parameters={
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 64,
        },
    ),
    "qwen2-bias": lambda: _tiny("qwen2", tie_word_embeddings=True),
    "qwen3-qknorm": lambda: _tiny("qwen3", head_dim=32),
}


def _prompts(n: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, VOCAB) for _ in range(rng.randint(3, 21))] for _ in range(n)]


def _hf_greedy(hf, prompt: list[int], n: int) -> list[int]:  # type: ignore[no-untyped-def]
    out = hf.generate(
        torch.tensor([prompt]),
        max_new_tokens=n,
        do_sample=False,
        min_new_tokens=n,
        pad_token_id=0,
        eos_token_id=None,
    )
    return out[0, len(prompt) :].tolist()


@pytest.mark.parametrize("name", MODELS)
def test_prefill_logits_match(name: str) -> None:
    hf, model = MODELS[name]()
    gen = PagedGenerator(model, num_blocks=64, block_size=BS, kv_dtype="fp32")
    prompts = _prompts(3)
    rids = [gen.new_request(p)[0] for p in prompts]
    logits = gen.run(gen.build_step(rids, [len(p) for p in prompts], all_logits=True))
    start = 0
    for p in prompts:
        ref = hf(torch.tensor([p])).logits[0]
        torch.testing.assert_close(logits[start : start + len(p)], ref, atol=2e-4, rtol=2e-4)
        start += len(p)


@pytest.mark.parametrize("name", MODELS)
def test_batched_greedy_is_token_exact(name: str) -> None:
    hf, model = MODELS[name]()
    gen = PagedGenerator(model, num_blocks=128, block_size=BS, kv_dtype="fp32")
    prompts = _prompts(5, seed=1)
    outs = gen.generate(prompts, max_new_tokens=12)
    for p, o in zip(prompts, outs, strict=True):
        assert o == _hf_greedy(hf, p, 12)


def test_prefix_cache_hit_gives_same_tokens() -> None:
    hf, model = MODELS["llama"]()
    gen = PagedGenerator(model, num_blocks=128, block_size=BS, kv_dtype="fp32")
    system = list(range(1, 14))
    a, b = [*system, 50, 51, 52], [*system, 60, 61]
    gen.generate([a], 6)
    rid, hit = gen.new_request(b)
    assert hit == 12  # three full 4-token blocks of the shared system prompt
    gen.free(rid)
    assert gen.generate([b], 6) == [_hf_greedy(hf, b, 6)]
    assert gen.kv.stats().prefix_hit_tokens >= 24


def test_chunked_prefill_matches_full_prefill() -> None:
    hf, model = MODELS["qwen2-bias"]()
    tokens = _prompts(1, seed=7)[0] * 3
    full = torch.nn.functional.cross_entropy(
        hf(torch.tensor([tokens])).logits[0, :-1], torch.tensor(tokens[1:]), reduction="none"
    )
    for chunk in (1, 5, 64):
        gen = PagedGenerator(model, num_blocks=64, block_size=BS, kv_dtype="fp32")
        torch.testing.assert_close(gen.score(tokens, chunk), full, atol=2e-4, rtol=2e-4)


def test_fork_copy_on_write_keeps_branches_separate() -> None:
    hf, model = MODELS["llama"]()
    gen = PagedGenerator(model, num_blocks=64, block_size=BS, kv_dtype="fp32")
    prompt = _prompts(1, seed=3)[0]
    prompt = prompt[
        : (len(prompt) // BS) * BS + 2
    ]  # end mid-block so the fork shares a partial block
    parent, _ = gen.new_request(prompt)
    gen.run(gen.build_step([parent], [len(prompt)]))
    child = gen.fork(parent)
    gen.append(parent, 11)
    gen.append(child, 22)
    assert gen.kv.stats().cow_copies == 1
    logits = gen.run(gen.build_step([parent, child], [1, 1]))
    for row, tok in zip(logits, (11, 22), strict=True):
        ref = hf(torch.tensor([[*prompt, tok]])).logits[0, -1]
        torch.testing.assert_close(row, ref, atol=2e-4, rtol=2e-4)


def test_sliding_window_matches_mistral() -> None:
    hf, model = _tiny("mistral", sliding_window=6)
    gen = PagedGenerator(model, num_blocks=128, block_size=BS, kv_dtype="fp32")
    prompts = [p * 2 for p in _prompts(3, seed=4)]  # longer than the window
    assert gen.generate(prompts, 10) == [_hf_greedy(hf, p, 10) for p in prompts]


@pytest.mark.parametrize("evict", [False, True])
def test_swap_out_and_in_preserves_kv(evict: bool) -> None:
    hf, model = MODELS["llama"]()
    gen = PagedGenerator(model, num_blocks=32, block_size=BS, kv_dtype="fp32", num_cpu_blocks=32)
    prompt = _prompts(1, seed=5)[0] + [7, 8, 9]
    rid, _ = gen.new_request(prompt)
    gen.run(gen.build_step([rid], [len(prompt)]))
    gen.kv.commit(rid, gen.tokens[rid])
    assert gen.cpu_cache is not None
    gen.cache.transfer_to(gen.cpu_cache, gen.kv.swap_out(rid) or [])
    if evict:
        # Another request takes every GPU block and overwrites it with its own K/V,
        # evicting the swapped request's cached blocks: all of it must come from host.
        hog_prompt = [5] * (32 * BS)
        hog, _ = gen.new_request(hog_prompt)
        gen.run(gen.build_step([hog], [len(hog_prompt)]))
        gen.free(hog)
    back = gen.kv.swap_in(rid)
    assert back is not None
    hits = gen.kv.stats().swap_in_cache_hits
    assert (hits == 0) if evict else (hits == len(prompt) // BS)
    gen.cpu_cache.transfer_to(gen.cache, back)
    gen.append(rid, 42)
    logits = gen.run(gen.build_step([rid], [1]))
    ref = hf(torch.tensor([[*prompt, 42]])).logits[0, -1]
    torch.testing.assert_close(logits[0], ref, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    ("kv_dtype", "max_rel_err"), [("bf16", 0.02), ("fp8", 0.1), ("int8", 0.03)]
)
def test_quantized_kv_stays_close(kv_dtype: str, max_rel_err: float) -> None:
    _, model = MODELS["llama"]()
    tokens = _prompts(1, seed=9)[0] * 3
    exact = PagedGenerator(model, 64, BS, kv_dtype="fp32").score(tokens, 4)
    approx = PagedGenerator(model, 64, BS, kv_dtype=kv_dtype).score(tokens, 4)
    rel = ((approx - exact).abs().mean() / exact.abs().mean()).item()
    assert rel < max_rel_err
