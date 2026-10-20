import os
import threading

import numpy as np
import pytest
import torch
from transformer_observatory.analysis import fit_projection, project_trace, transform
from transformer_observatory.capture import Capture
from transformer_observatory.models import TinyModel
from transformer_observatory.schema import AnalysisRequest
from transformer_observatory.storage import Store, TraceWriter

torch.set_num_threads(1 if os.name == "nt" else 2)


def capture_model(tmp_path, model, request=None):
    request = request or AnalysisRequest(prompt="test", level=4, attention=True, device="cpu")
    store = Store(tmp_path / "data", tmp_path / "traces")
    trace_id = "1" * 32
    writer = TraceWriter(
        store,
        trace_id,
        dict(
            title="test",
            level=request.level,
            request=request.model_dump(),
            tokens=[dict(id=i, processed=True) for i in [1, 4, 7, 9]],
            model=dict(fingerprint="fixture"),
        ),
    )
    adapter = Capture(model, writer, request, threading.Event(), lambda *_: None).install()
    adapter.set_step(0, range(4))
    try:
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([[1, 4, 7, 9]]), use_cache=False)
    finally:
        adapter.close()
    return store, writer, adapter, result


def layer_norm(x, layer):
    mean, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + layer.eps) * layer.weight.detach().numpy() + layer.bias.detach().numpy()


def test_tiny_matches_independent_numpy_reference(tmp_path):
    model = TinyModel().eval()
    store, writer, adapter, output = capture_model(tmp_path, model)
    residual = [r for r in writer.manifest["tensors"] if r["kind"] == "residual"]
    x = model.model.embed_tokens.weight.detach().numpy()[[1, 4, 7, 9]].astype(np.float64)
    expected = [x.copy()]
    for block in model.model.layers:
        h = layer_norm(x, block.input_layernorm)
        q, k, v = [
            (h @ p.weight.detach().numpy().T).reshape(4, 4, 16).transpose(1, 0, 2)
            for p in [block.self_attn.q_proj, block.self_attn.k_proj, block.self_attn.v_proj]
        ]
        scores = q @ k.transpose(0, 2, 1) / 4
        scores[:, np.triu_indices(4, 1)[0], np.triu_indices(4, 1)[1]] = -np.inf
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        z = (p @ v).transpose(1, 0, 2).reshape(4, 64)
        r = x + z @ block.self_attn.o_proj.weight.detach().numpy().T
        expected.append(r.copy())
        h = layer_norm(r, block.post_attention_layernorm)
        gate, up = [h @ p.weight.detach().numpy().T for p in [block.mlp.gate_proj, block.mlp.up_proj]]
        x = r + (gate / (1 + np.exp(-gate)) * up) @ block.mlp.down_proj.weight.detach().numpy().T
        expected.append(x.copy())
    assert len(residual) == 9
    for record, ref in zip(residual, expected):
        np.testing.assert_allclose(store.load(writer.trace_id, record), ref, rtol=2e-5, atol=1e-6)
    logits = layer_norm(x, model.model.norm) @ model.lm_head.weight.detach().numpy().T
    np.testing.assert_allclose(output.logits[0].numpy(), logits, rtol=3e-5, atol=2e-6)
    assert all(c["passed"] for c in adapter.checks)


@pytest.mark.parametrize("architecture", ["qwen3", "gpt2"])
def test_huggingface_adapters_random_small_model(tmp_path, architecture):
    from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM

    if architecture == "qwen3":
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()
    else:
        config = GPT2Config(vocab_size=32, n_embd=32, n_layer=2, n_head=4, n_positions=32)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    store, writer, adapter, output = capture_model(tmp_path, model)
    assert len(writer.manifest["stages"]) == 5
    assert all(c["passed"] for c in adapter.checks)
    assert any(c["name"] == "attention_qkv_reconstruction" for c in adapter.checks)
    assert torch.isfinite(output.logits).all()


def test_shared_projection_displacement_and_metrics(tmp_path):
    store, writer, _, _ = capture_model(tmp_path, TinyModel().eval())
    project_trace(store, writer, threading.Event())
    coords, displacement, metrics = transform(store, writer.manifest, writer.manifest["projection"])
    assert coords.shape == (4, 9, 3)
    assert np.array_equal(displacement[:, 0], np.zeros((4, 3)))
    assert np.all(metrics[:, 0, 1:4] == 0)
    basis = np.array(writer.manifest["projection"]["basis"])
    for row in basis:
        assert row[np.argmax(np.abs(row))] >= 0
    again = transform(store, writer.manifest, writer.manifest["projection"])[0]
    np.testing.assert_array_equal(coords, again)


def test_incremental_projection_is_finite_and_reproducible():
    x = np.random.default_rng(7).normal(size=(1061, 16)).astype(np.float32)

    def arrays():
        for start in range(0, len(x), 7):
            yield x[start : start + 7]

    a = fit_projection(arrays, "incremental", 64)
    b = fit_projection(arrays, "incremental", 64)
    np.testing.assert_allclose(a["basis"], b["basis"])
    assert np.isfinite(a["basis"]).all()
