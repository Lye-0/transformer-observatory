import json
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from transformer_observatory.api import create_app
from transformer_observatory.lenses import apply_lens, register_lens
from transformer_observatory.models import TinyModel, tiny_registration
from transformer_observatory.research import install_intervention
from transformer_observatory.schema import StudyRequest
from transformer_observatory.storage import Store


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("research")
    app = create_app(Store(root / "data", root / "traces"), testing=True)
    with TestClient(app) as c:
        c.headers["x-observatory-token"] = c.get("/api/session").json()["token"]
        yield c


def wait(client, url):
    for _ in range(600):
        item = client.get(url).json()
        if item["state"] in {"completed", "failed", "cancelled"}:
            return item
        time.sleep(0.025)
    pytest.fail(f"Timeout: {url}")


@pytest.fixture(scope="module")
def pair(client):
    ids = []
    for text in ["Hello", "Helpx"]:
        r = client.post("/api/analysis/jobs", json={"prompt": text, "device": "cpu", "level": 2})
        assert r.status_code == 202, r.text
        trace_id = r.json()["id"]
        assert wait(client, f"/api/analysis/jobs/{trace_id}")["state"] == "completed"
        ids.append(trace_id)
    return ids


def study(client, body):
    r = client.post("/api/research/studies", json=body)
    assert r.status_code == 202, r.text
    item = wait(client, f"/api/research/studies/{r.json()['id']}")
    assert item["state"] == "completed", item
    return item


@pytest.mark.parametrize("mode", ["residual_patch", "head_ablation", "mlp_ablation"])
def test_interventions_restore_model_and_obey_causality(client, pair, mode):
    item = study(
        client,
        dict(
            kind="intervention",
            mode=mode,
            trace_ids=pair if mode == "residual_patch" else pair[:1],
            positions=[5],
            source_positions=[5],
            readout_position=2,
            layer=0,
            head=1,
        ),
    )
    result = item["report"]["result"]
    assert result["restoration_max_logit_error"] == 0
    assert result["logit_l2"] == 0  # Later-token intervention cannot affect a causal prefix.
    assert client.get(f"/api/traces/{pair[0]}").json()["status"] == "completed"


@pytest.mark.parametrize("stage", ["embedding", "layer.0.mlp", "layer.3.mlp"])
def test_self_patch_is_identity_but_cross_prompt_patch_changes_output(client, pair, stage):
    control = study(client, dict(kind="intervention", trace_ids=[pair[0], pair[0]], stage=stage))
    assert control["report"]["result"]["logit_l2"] == 0
    changed = study(client, dict(kind="intervention", trace_ids=pair, stage=stage))
    assert changed["report"]["result"]["logit_l2"] > 0
    assert changed["report"]["result"]["restoration_max_logit_error"] == 0


def test_head_ablation_matches_independent_preprojection_calculation():
    model = TinyModel().eval()
    attn = model.model.layers[0].self_attn
    h = torch.randn(1, 5, 64)
    with torch.inference_mode():
        b, t, _ = h.shape
        q, k, v = [p(h).reshape(b, t, 4, 16).transpose(1, 2) for p in [attn.q_proj, attn.k_proj, attn.v_proj]]
        scores = q @ k.transpose(-1, -2) / 4
        scores = scores.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), float("-inf"))
        z = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, 5, 64)
        z[:, [3, 4], 16:32] = 0
        expected = attn.o_proj(z)
        handle = install_intervention(
            model,
            StudyRequest(kind="intervention", trace_ids=["test"], mode="head_ablation", head=1),
            [3, 4],
            [],
        )
        try:
            actual = attn(h)[0]
        finally:
            handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_zero_strength_and_mlp_ablation(client, pair):
    for mode in ["head_ablation", "mlp_ablation"]:
        zero = study(client, dict(kind="intervention", trace_ids=pair[:1], mode=mode, strength=0))
        full = study(client, dict(kind="intervention", trace_ids=pair[:1], mode=mode, strength=1))
        assert zero["report"]["result"]["logit_l2"] == 0
        assert full["report"]["result"]["logit_l2"] > 0


def test_statistics_mean_of_experiments_and_exports(client, pair):
    item = study(client, dict(kind="statistics", trace_ids=pair))
    result = item["report"]["result"]
    assert not any(r["metric"] == "projected_update" for r in result["rows"])
    row = next(r for r in result["rows"] if r["metric"] == "norm" and r["stage"] == "layer.0.mlp")
    store = client.app.state.store
    means = []
    for trace_id in pair:
        m = store.manifest(trace_id)
        r = next(r for r in m["tensors"] if r["kind"] == "metrics")
        means.append(np.mean(store.load(trace_id, r)[:, 2, 0], dtype=np.float64))
    assert row["mean"] == np.mean(means)
    assert row["std"] == np.std(means, ddof=1)
    assert (
        client.get(f"/api/research/studies/{item['id']}/export?format=json").json()["model_fingerprint"]
        == tiny_registration()["fingerprint"]
    )
    csv = client.get(f"/api/research/studies/{item['id']}/export?format=csv")
    assert csv.status_code == 200 and "metric" in csv.text


def test_tsne_all_points_and_original_projection_is_unchanged(client, pair):
    original = client.get(f"/api/traces/{pair[0]}").json()["projection"]
    item = study(client, dict(kind="tsne", trace_ids=pair[:1], seed=7, perplexity=5))
    result = item["report"]["result"]
    assert np.asarray(result["coordinates"]).shape == (6, 9, 3)
    assert result["point_count"] == 54
    assert client.get(f"/api/traces/{pair[0]}").json()["projection"] == original


def test_invalid_scopes_fail_without_changing_trace(client, pair):
    response = client.post(
        "/api/research/studies", json=dict(kind="statistics", trace_ids=[pair[0], pair[0]])
    )
    assert response.status_code == 400
    r = client.post(
        "/api/research/studies", json=dict(kind="intervention", trace_ids=pair, stage="layer.0.attention")
    )
    item = wait(client, f"/api/research/studies/{r.json()['id']}")
    assert item["state"] == "failed" and "Attention" in item["error"]


def test_queued_research_cancellation(client, pair):
    import threading

    entered, release = threading.Event(), threading.Event()
    client.app.state.jobs.executor.submit(lambda: (entered.set(), release.wait(5)))
    assert entered.wait(5)
    try:
        item = client.post("/api/research/studies", json=dict(kind="statistics", trace_ids=pair)).json()
        assert client.post(f"/api/research/studies/{item['id']}/cancel").status_code == 200
    finally:
        release.set()
    assert wait(client, f"/api/research/studies/{item['id']}")["state"] == "cancelled"


def test_active_research_protects_source_from_delete_move_and_cleanup(client, pair):
    import threading

    store = client.app.state.store
    entered, release = threading.Event(), threading.Event()
    client.app.state.jobs.executor.submit(lambda: (entered.set(), release.wait(10)))
    assert entered.wait(5)
    try:
        item = client.post("/api/research/studies", json=dict(kind="statistics", trace_ids=pair)).json()
        with pytest.raises(ValueError):
            store.delete(pair[0])
        with pytest.raises(ValueError):
            store.retain(pair[0], True)
        original = store.manifest(pair[0])
        expired = {**original, "created_at": 0}
        store.save_manifest(expired)
        assert pair[0] not in {r["id"] for r in store.cleanup_preview()["targets"]}
        store.save_manifest(original)
    finally:
        release.set()
    assert wait(client, f"/api/research/studies/{item['id']}")["state"] == "completed"


def test_tsne_resource_limit_rejects_without_sampling(client, pair):
    from transformer_observatory.research import Research

    manifest = client.app.state.store.manifest(pair[0])
    oversized = {**manifest, "tokens": manifest["tokens"] * 100}
    import threading

    with pytest.raises(ValueError, match="4096"):
        Research(client.app.state.store, client.app.state.jobs).tsne(
            StudyRequest(kind="tsne", trace_ids=pair[:1]), oversized, threading.Event(), lambda _: None
        )


def test_interrupted_study_is_recovered_but_completed_report_survives(tmp_path):
    store = Store(tmp_path / "data", tmp_path / "traces")
    store.put("studies", "running", dict(id="running", state="running", request={"trace_ids": []}))
    store.put("studies", "done", dict(id="done", state="completed", report={"result": "kept"}))
    reopened = Store(tmp_path / "data", tmp_path / "traces")
    assert reopened.get("studies", "running")["state"] == "failed"
    assert reopened.get("studies", "done")["report"]["result"] == "kept"


@pytest.mark.parametrize("kind", ["statistics", "tsne"])
def test_research_rejects_changed_source_tensor(client, pair, kind):
    store = client.app.state.store
    manifest = store.manifest(pair[0])
    record = next(
        r for r in manifest["tensors"] if r["kind"] == ("metrics" if kind == "statistics" else "residual")
    )
    path = store.tensor_path(pair[0], record)
    original = path.read_bytes()
    changed = bytearray(original)
    changed[-1] ^= 1
    path.write_bytes(changed)
    try:
        item = client.post(
            "/api/research/studies",
            json=dict(kind=kind, trace_ids=pair if kind == "statistics" else pair[:1]),
        ).json()
        state = wait(client, f"/api/research/studies/{item['id']}")
        assert state["state"] == "failed" and "ハッシュ" in state["error"]
    finally:
        path.write_bytes(original)


def make_lens(tmp_path, format="safe", fingerprint=None):
    root = tmp_path / format
    root.mkdir()
    config = dict(
        lens_type="linear_tuned_lens",
        d_model=64,
        num_hidden_layers=4,
        bias=True,
        observatory_model_fingerprint=fingerprint or tiny_registration()["fingerprint"],
        training_info="Numerical fixture, not a pretrained semantic lens",
    )
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    values = {
        f"{i}.{name}": torch.zeros((64, 64) if name == "weight" else (64,))
        for i in range(4)
        for name in ["weight", "bias"]
    }
    if format == "safe":
        save_file(values, str(root / "params.safetensors"))
    else:
        torch.save(values, root / "params.pt")
    return root


@pytest.mark.parametrize("format", ["safe", "pt"])
def test_tuned_lens_official_residual_affine_and_stage_mapping(client, pair, tmp_path, format):
    root = make_lens(tmp_path, format)
    registered = client.post(
        "/api/research/lenses/register", json=dict(model_id="tiny-transformer", path=str(root))
    )
    assert registered.status_code == 200, registered.text
    lens = registered.json()
    body = dict(stage="layer.0.mlp", token=1, step=0)
    baseline = client.post(f"/api/traces/{pair[0]}/lens", json=body).json()
    tuned = client.post(f"/api/traces/{pair[0]}/lens", json={**body, "lens_id": lens["id"]}).json()
    assert tuned["candidates"] == baseline["candidates"]
    assert tuned["lens"]["index"] == 1  # HF hidden_states[1] follows Block 0.
    assert "Tuned Lens" in tuned["method"]
    invalid = client.post(
        f"/api/traces/{pair[0]}/lens", json={**body, "stage": "layer.0.attention", "lens_id": lens["id"]}
    )
    assert invalid.status_code == 400


def test_tuned_lens_rejects_wrong_model_and_changed_weights(client, pair, tmp_path):
    root = make_lens(tmp_path, fingerprint="wrong-model")
    assert (
        client.post(
            "/api/research/lenses/register", json=dict(model_id="tiny-transformer", path=str(root))
        ).status_code
        == 400
    )
    config = json.loads((root / "config.json").read_text())
    config["observatory_model_fingerprint"] = tiny_registration()["fingerprint"]
    (root / "config.json").write_text(json.dumps(config))
    lens = client.post(
        "/api/research/lenses/register", json=dict(model_id="tiny-transformer", path=str(root))
    ).json()
    config["training_info"] = "changed"
    (root / "config.json").write_text(json.dumps(config))
    response = client.post(
        f"/api/traces/{pair[0]}/lens", json=dict(stage="embedding", token=0, step=0, lens_id=lens["id"])
    )
    assert response.status_code == 400 and "変更" in response.json()["detail"]


def test_nonzero_tuned_lens_matches_residual_affine_reference(tmp_path):
    from safetensors.torch import load_file

    root = make_lens(tmp_path)
    state = load_file(str(root / "params.safetensors"))
    state["0.weight"] = torch.eye(64) * 0.25
    state["0.bias"] = torch.linspace(-0.3, 0.3, 64)
    save_file(state, str(root / "params.safetensors"))
    store = Store(tmp_path / "data", tmp_path / "traces")
    model = tiny_registration()
    registered = register_lens(store, model, str(root))
    x = torch.randn(64)
    actual, metadata = apply_lens(store, registered, model, "embedding", x)
    torch.testing.assert_close(actual, x * 1.25 + state["0.bias"])
    assert metadata["index"] == 0


@pytest.mark.parametrize("architecture", ["qwen3", "gpt2"])
def test_huggingface_patch_replaces_actual_final_residual(architecture):
    from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM

    if architecture == "gpt2":
        config = GPT2Config(vocab_size=32, n_embd=32, n_layer=2, n_head=4, n_positions=16)
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).eval()
    else:
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
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        baseline = model(input_ids=inputs, use_cache=False).logits
        handle = install_intervention(
            model,
            StudyRequest(kind="intervention", trace_ids=["test", "test"], stage="layer.1.mlp"),
            list(range(4)),
            [np.zeros(32) for _ in range(4)],
        )
        try:
            changed = model(input_ids=inputs, use_cache=False).logits
        finally:
            handle.remove()
        restored = model(input_ids=inputs, use_cache=False).logits
    assert torch.count_nonzero(changed) == 0
    assert torch.count_nonzero(baseline) > 0
    torch.testing.assert_close(restored, baseline, atol=0, rtol=0)
