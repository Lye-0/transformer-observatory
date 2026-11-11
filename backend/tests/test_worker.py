"""Exercise the production process boundary, alongside the in-process numerical suite."""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from transformer_observatory.api import create_app
from transformer_observatory.schema import AnalysisRequest
from transformer_observatory.storage import Store
from transformer_observatory.worker import RuntimeManager, RuntimeUnavailable, serve


def delayed_worker(connection, data_root, trace_root, startup_log=None):
    time.sleep(2)
    serve(connection, data_root, trace_root, startup_log)


def broken_worker(connection, *_):
    connection.send(("failed", None, "injected native import failure"))
    connection.close()


def await_state(function, expected, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = function()
        if value["state"] in expected:
            return value
        time.sleep(0.05)
    pytest.fail(f"Timed out: {value}")


def test_ui_startup_never_imports_inference_libraries(tmp_path):
    code = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sklearn', 'scipy'}:
            raise AssertionError('Heavy import on UI startup: ' + fullname)
sys.meta_path.insert(0, Reject())
from transformer_observatory.api import create_app
from transformer_observatory.storage import Store
from fastapi.testclient import TestClient
app = create_app(Store(sys.argv[1], sys.argv[2]))
with TestClient(app, base_url='http://127.0.0.1') as c:
    c.get('/api/session')
    for path in ['/api/models', '/api/settings', '/api/traces', '/api/research/studies', '/api/runtime']:
        assert c.get(path).status_code == 200, path
    assert app.state.runtime.status()['state'] == 'idle'
    assert app.state.runtime.process is None
print('lightweight startup verified')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "db"), str(tmp_path / "traces")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_production_runtime_wait_cancel_reuse_and_numerics(tmp_path):
    store = Store(tmp_path / "db", tmp_path / "traces")
    app = create_app(store)
    app.state.runtime.worker_target = delayed_worker
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = client.get("/api/session").json()["token"]
        client.headers["x-observatory-token"] = token
        request = AnalysisRequest(prompt="abc", level=4, device="cpu")
        started = time.monotonic()
        job = client.post("/api/analysis/jobs", json=request.model_dump()).json()
        assert job["state"] == "loading_runtime"
        assert time.monotonic() - started < 1.5
        assert client.get("/api/settings").status_code == 200
        cancelled = client.post(f"/api/analysis/jobs/{job['id']}/cancel").json()
        assert cancelled["state"] == "cancelled"
        with ThreadPoolExecutor(6) as executor:
            statuses = list(executor.map(lambda _: app.state.runtime.start(), range(6)))
        assert len({s["pid"] for s in statuses}) == 1
        second = client.post("/api/analysis/jobs", json=request.model_dump()).json()
        done = await_state(lambda: store.get("jobs", second["id"]), {"completed", "failed"})
        assert done["state"] == "completed", done
        assert done["elapsed_seconds"] >= 1
        assert store.get("jobs", job["id"])["state"] == "cancelled"
        manifest = store.manifest(second["id"])
        assert manifest["integrity"]["all_passed"]
        assert len(manifest["checks"]) > 0
        pid = app.state.runtime.status()["pid"]
        pf = client.post("/api/analysis/preflight", json=request.model_dump())
        assert pf.status_code == 200, pf.text
        assert pf.json()["input_tokens"] == 4
        lens = client.post(f"/api/traces/{second['id']}/lens", json=dict(stage="embedding", token=1, step=0))
        assert lens.status_code == 200, lens.text
        assert lens.json()["candidates"]
        assert app.state.runtime.status()["pid"] == pid
        # Settings/retention changes must reach a worker that was already prepared.
        new_root = tmp_path / "new-traces"
        new_root.mkdir()
        store.trace_root = new_root
        third = client.post("/api/analysis/jobs", json=request.model_dump()).json()
        third = await_state(lambda: store.get("jobs", third["id"]), {"completed", "failed"})
        assert third["state"] == "completed", third
        assert (new_root / third["id"] / "manifest.json").is_file()
        comparison = client.post("/api/traces/compare", json=dict(left=second["id"], right=third["id"]))
        assert comparison.status_code == 200, comparison.text
        study = client.post(
            "/api/research/studies", json=dict(kind="statistics", trace_ids=[second["id"], third["id"]])
        )
        assert study.status_code == 202, study.text
        study_id = study.json()["id"]
        report = await_state(lambda: store.get("studies", study_id), {"completed", "failed"})
        assert report["state"] == "completed", report
        assert client.get(f"/api/research/studies/{study_id}/export?format=csv").status_code == 200
        app.state.runtime.process.terminate()
        app.state.runtime.process.join(timeout=5)
        failed = await_state(app.state.runtime.status, {"failed"})
        assert failed["error"]
        assert client.get("/api/traces").status_code == 200
    assert not app.state.runtime.process.is_alive()


def test_failure_and_explicit_retry(tmp_path):
    store = Store(tmp_path / "db", tmp_path / "traces")
    runtime = RuntimeManager(store, worker_target=broken_worker)
    try:
        runtime.start()
        with pytest.raises(RuntimeUnavailable, match="injected native"):
            runtime.wait()
        failed_pid = runtime.status()["pid"]
        assert runtime.start()["pid"] == failed_pid  # no automatic failure/retry loop
        runtime.start(retry=True)
        assert runtime.status()["pid"] != failed_pid
        with pytest.raises(RuntimeUnavailable, match="injected native"):
            runtime.wait()
    finally:
        runtime.close()


def test_worker_attach_does_not_recover_live_jobs(tmp_path):
    store = Store(tmp_path / "db", tmp_path / "traces")
    store.put("jobs", "pending", dict(id="pending", state="loading_runtime"))
    Store(store.data_root, store.trace_root, recover=False)
    assert store.get("jobs", "pending")["state"] == "loading_runtime"
