import json
import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from transformer_observatory.api import create_app
from transformer_observatory.model_discovery import (
    ModelDiscovery,
    browse_folders,
    inspect_model,
    search_roots,
)
from transformer_observatory.models import register_model
from transformer_observatory.storage import Store


def candidate_folder(path, architecture="gpt2"):
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(dict(model_type=architecture, n_embd=16, n_layer=1, n_head=2))
    )
    (path / "model.safetensors").write_bytes(b"metadata-only test fixture")
    (path / "tokenizer.json").write_text("{}")
    return path


def completed(discovery, job):
    for _ in range(300):
        result = discovery.get(job["id"])
        if result["state"] != "running":
            return result
        time.sleep(0.01)
    pytest.fail("Discovery did not finish")


def test_discovery_inspects_cache_snapshots_deduplicates_and_limits_depth(tmp_path, monkeypatch):
    snapshot = candidate_folder(tmp_path / "models--owner--model" / "snapshots" / "revision")
    candidate_folder(tmp_path / "one" / "two" / "three" / "four" / "too-deep")
    unsupported = candidate_folder(tmp_path / "unsupported", "llama")
    # Candidate detection must not read/hash the weights.
    original = Path.read_bytes

    def read_metadata(path):
        assert path.suffix != ".safetensors"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_metadata)
    discovery = ModelDiscovery()
    try:
        job = discovery.start([dict(path=str(tmp_path)), dict(path=str(tmp_path))])
        result = completed(discovery, job)
        assert result["state"] == "completed"
        assert len(result["candidates"]) == 2
        found = {item["path"]: item for item in result["candidates"]}
        assert found[str(snapshot)]["name"] == "owner/model"
        assert found[str(snapshot)]["status"] == "available"
        assert found[str(unsupported)]["status"] == "unsupported"
    finally:
        discovery.close()


def test_inspection_reports_incomplete_quantized_and_malformed_models(tmp_path):
    folder = candidate_folder(tmp_path / "incomplete")
    (folder / "model.safetensors.index.json").write_text(
        json.dumps(dict(weight_map={"x": "missing.safetensors"}))
    )
    assert "分割重みが不足" in inspect_model(folder)["reason"]
    (folder / "model.safetensors.index.json").unlink()
    config = json.loads((folder / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "gptq"}
    (folder / "config.json").write_text(json.dumps(config))
    assert inspect_model(folder)["status"] == "unsupported"
    (folder / "config.json").write_text("[]")
    assert inspect_model(folder)["status"] == "invalid"
    (folder / "config.json").write_text("not json")
    assert inspect_model(folder)["status"] == "invalid"
    with pytest.raises(ValueError, match="絶対パス"):
        inspect_model("relative/model")


def test_discovery_can_cancel_and_enforces_time_budget(tmp_path, monkeypatch):
    import transformer_observatory.model_discovery as module

    folder = candidate_folder(tmp_path / "candidate")
    entered, release = threading.Event(), threading.Event()
    original = module.inspect_model

    def gated(path):
        entered.set()
        assert release.wait(5)
        return original(path)

    monkeypatch.setattr(module, "inspect_model", gated)
    discovery = ModelDiscovery()
    job = discovery.start([dict(path=str(folder))])
    try:
        assert entered.wait(5)
        assert discovery.start([])["id"] == job["id"]
        discovery.cancel(job["id"])
    finally:
        release.set()
    assert completed(discovery, job)["state"] == "cancelled"
    times = iter([0, 16])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    limited = discovery.start([dict(path=str(folder))])
    # Avoid the helper's timer: this worker exits before inspecting any directory.
    discovery.thread.join(timeout=5)
    result = discovery.get(limited["id"])
    assert result["truncated"] and result["scanned"] == 0
    discovery.close()


def test_search_settings_and_folder_browser_survive_catalog_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    store = Store(tmp_path / "data", tmp_path / "traces")
    folder = tmp_path / "custom models"
    folder.mkdir()
    (folder / "子フォルダー").mkdir()
    (folder / "file.txt").write_text("Not a folder")
    with TestClient(create_app(store, testing=True)) as client:
        client.headers["x-observatory-token"] = client.get("/api/session").json()["token"]
        assert client.put("/api/models/search-paths", json={"paths": ["relative"]}).status_code == 400
        response = client.put("/api/models/search-paths", json={"paths": [str(folder), str(folder)]})
        assert response.status_code == 200
        assert response.json()["custom_paths"] == [str(folder)]
        listing = client.get("/api/models/folders", params={"path": str(folder)}).json()
        assert [item["name"] for item in listing["entries"]] == ["子フォルダー"]
        assert listing["parent"] == str(tmp_path)
        assert client.get("/api/models/folders", params={"path": str(folder / "absent")}).status_code == 400
    reopened = Store(store.data_root, store.trace_root)
    assert any(root["path"] == str(folder) for root in search_roots(reopened))
    assert any(root["path"] == str(tmp_path / "cache") for root in search_roots(reopened))
    assert browse_folders(str(folder), reopened)["path"] == str(folder)
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    assert search_roots(reopened)[2]["path"] == str(tmp_path / "xdg-cache/huggingface/hub")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    assert search_roots(reopened)[2]["path"] == str(tmp_path / "hf-home/hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy-hub"))
    assert search_roots(reopened)[2]["path"] == str(tmp_path / "legacy-hub")


@pytest.fixture
def saved_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    root = tmp_path / "random-gpt2-fixture"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "sky": 2, ".": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(root)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4, n_embd=16, n_layer=1, n_head=2, n_positions=64, bos_token_id=0, eos_token_id=0
        )
    ).save_pretrained(root)
    return root


def test_registration_persistence_relocation_and_real_analysis(tmp_path, saved_model):
    store = Store(tmp_path / "data", tmp_path / "traces")
    with TestClient(create_app(store, testing=True)) as client:
        client.headers["x-observatory-token"] = client.get("/api/session").json()["token"]
        response = client.post(
            "/api/models/register", json={"path": str(saved_model), "name": "Random fixture"}
        )
        assert response.status_code == 200, response.text
        model = response.json()
        assert model["name"] == "Random fixture"
        assert client.get("/api/models").json()[1]["available"]
        moved = tmp_path / "moved model"
        shutil.move(str(saved_model), moved)
        assert client.get("/api/models").json()[1]["available"] is False
        response = client.post(f"/api/models/{model['id']}/relocate", json={"path": str(moved)})
        assert response.status_code == 200, response.text
        assert response.json()["id"] == model["id"]
        job = client.post(
            "/api/analysis/jobs", json={"model_id": model["id"], "prompt": "hello sky .", "device": "cpu"}
        ).json()
        for _ in range(300):
            result = client.get(f"/api/analysis/jobs/{job['id']}").json()
            if result["state"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert result["state"] == "completed", result
        (moved / "tokenizer_config.json").write_text('{"changed": true}')
        response = client.post(f"/api/models/{model['id']}/relocate", json={"path": str(moved)})
        assert response.status_code == 400
    reopened = Store(store.data_root, store.trace_root)
    assert reopened.get("models", model["id"])["path"] == str(moved)


def test_registration_rejects_unreadable_tokenizer(tmp_path):
    folder = candidate_folder(tmp_path / "broken-tokenizer")
    with pytest.raises(ValueError, match="Tokenizer"):
        register_model(folder)


def test_registration_rejects_incomplete_safetensors(saved_model):
    (saved_model / "model.safetensors").write_bytes(b"incomplete download")
    with pytest.raises(ValueError, match="safetensors"):
        register_model(saved_model)


def test_permission_errors_are_reported_without_hiding_other_candidates(tmp_path, monkeypatch):
    import transformer_observatory.model_discovery as module

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    model = candidate_folder(tmp_path / "valid")
    original = module.os.scandir

    def scan(path):
        if Path(path) == blocked:
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(module.os, "scandir", scan)
    store = Store(tmp_path / "data", tmp_path / "traces")
    with pytest.raises(ValueError, match="読み取れません"):
        browse_folders(str(blocked), store)
    discovery = ModelDiscovery()
    try:
        result = completed(discovery, discovery.start([dict(path=str(tmp_path))]))
        assert result["state"] == "completed"
        assert len(result["warnings"]) == 1
        assert [item["path"] for item in result["candidates"]] == [str(model)]
    finally:
        discovery.close()
