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


def test_selected_head_scope_preserves_all_residual_trajectories(tmp_path):
    request = AnalysisRequest(prompt="test", level=3, heads=[1], layers=[0], device="cpu")
    _, writer, adapter, _ = capture_model(tmp_path, TinyModel().eval(), request)
    assert len(writer.manifest["stages"]) == 9
    details = [r for r in writer.manifest["tensors"] if r["kind"] == "tensor"]
    assert all(r["layer"] == 0 for r in details)
    for r in details:
        if r["heads"] is not None:
            assert r["heads"] == [1]
    assert all(c["passed"] for c in adapter.checks)


def test_zero_vectors_keep_validity_separate_from_numeric_storage(tmp_path):
    request = AnalysisRequest(prompt="x")
    store = Store(tmp_path / "data", tmp_path / "traces")
    writer = TraceWriter(
        store,
        "b" * 32,
        dict(title="zero", level=2, request=request.model_dump(), tokens=[dict(id=1, processed=True)]),
    )
    writer.write(np.zeros((1, 4), dtype=np.float32), kind="residual", stage="embedding", positions=[0])
    project_trace(store, writer, threading.Event())
    r = next(r for r in writer.manifest["tensors"] if r["kind"] == "metric_validity")
    validity = store.load(writer.trace_id, r)
    assert validity[0, 0, 3] == 0
    assert validity[0, 0, 4] == 0
    assert validity[0, 0, 6] == 0


def test_tensor_scope_and_causal_mask_are_explicit(tmp_path):
    request = AnalysisRequest(
        prompt="test", level=4, layers=[0], heads=[1], tensors=["q", "attention_mask"], device="cpu"
    )
    store, writer, adapter, _ = capture_model(tmp_path, TinyModel().eval(), request)
    details = [r for r in writer.manifest["tensors"] if r["kind"] == "tensor"]
    assert {r["stage"] for r in details} == {"q", "attention_mask"}
    mask = next(r for r in details if r["stage"] == "attention_mask")
    np.testing.assert_array_equal(
        store.load(writer.trace_id, mask), np.triu(np.ones((4, 4), dtype=np.uint8), 1)
    )
    assert len(writer.manifest["stages"]) == 9
    assert all(c["passed"] for c in adapter.checks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_bfloat16_cuda_accumulation_matches_cpu_reference():
    from transformer_observatory.runtime import configure_runtime

    configure_runtime()
    rng = torch.Generator().manual_seed(1729)
    x = (torch.randn(5, 3072, generator=rng) * 8).bfloat16()
    weight = torch.randn(1024, 3072, generator=rng).bfloat16()
    cpu_result = torch.nn.functional.linear(x, weight).float()
    gpu_result = torch.nn.functional.linear(x.cuda(), weight.cuda()).cpu().float()
    torch.testing.assert_close(cpu_result, gpu_result, atol=0.05, rtol=0.04)
