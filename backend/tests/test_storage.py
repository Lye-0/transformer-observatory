import json
import os
import subprocess
import sys

import numpy as np
import pytest

from transformer_observatory.storage import Store, TraceWriter


@pytest.mark.parametrize("code", [5, 32, 33])
def test_atomic_json_retries_transient_windows_errors(tmp_path, monkeypatch, code):
    from transformer_observatory import storage

    path = tmp_path / "manifest.json"
    storage.atomic_json(path, {"old": True})
    original = storage.os.replace
    calls = []

    def blocked(source, destination):
        calls.append(source)
        if len(calls) < 3:
            assert json.loads(path.read_text()) == {"old": True}
            error = PermissionError("temporarily held file")
            error.winerror = code
            raise error
        original(source, destination)

    monkeypatch.setattr(storage.os, "replace", blocked)
    monkeypatch.setattr(storage.time, "sleep", lambda _: None)
    storage.atomic_json(path, {"new": True})
    assert len(calls) == 3
    assert json.loads(path.read_text()) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("code, expected_calls", [(5, 8), (112, 1)])
def test_atomic_json_preserves_previous_file_on_permanent_failure(
    tmp_path, monkeypatch, code, expected_calls
):
    from transformer_observatory import storage

    path = tmp_path / "manifest.json"
    storage.atomic_json(path, {"old": True})
    calls = []

    def blocked(*args):
        calls.append(args)
        error = OSError("persistent error")
        error.winerror = code
        raise error

    monkeypatch.setattr(storage.os, "replace", blocked)
    monkeypatch.setattr(storage.time, "sleep", lambda _: None)
    with pytest.raises(OSError, match="persistent error"):
        storage.atomic_json(path, {"new": True})
    assert len(calls) == expected_calls
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_atomic_json_survives_real_windows_reader_handle(tmp_path):
    import ctypes
    import threading
    from ctypes import wintypes

    from transformer_observatory.storage import atomic_json

    path = tmp_path / "manifest.json"
    atomic_json(path, {"old": True})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != wintypes.HANDLE(-1).value
    timer = threading.Timer(0.06, kernel.CloseHandle, args=(handle,))
    timer.start()
    try:
        atomic_json(path, {"new": True})
    finally:
        timer.join()
    assert json.loads(path.read_text()) == {"new": True}


def make_trace(tmp_path, level=2):
    store = Store(tmp_path / "data", tmp_path / "traces")
    writer = TraceWriter(store, "a" * 32, dict(title="fixture", level=level, model={}, tokens=[]))
    return store, writer


def test_each_chunk_is_durable_and_catalog_matches_manifest(tmp_path):
    from transformer_observatory.storage import digest

    store, writer = make_trace(tmp_path)
    writer.manifest["title"] = "部分データの復旧"
    for index in range(3):
        writer.write(np.full((2, 4), index), kind="residual", stage=f"layer.{index}")
        reopened = Store(store.data_root, store.trace_root, recover=False)
        manifest = reopened.manifest(writer.trace_id)
        assert manifest == reopened.get("traces", writer.trace_id)
        assert manifest["status"] == "incomplete"
        assert len(manifest["tensors"]) == index + 1
        for record in manifest["tensors"]:
            assert digest(reopened.tensor_path(writer.trace_id, record)) == record["sha256"]


def test_invalid_manifest_does_not_replace_file_or_catalog(tmp_path):
    store, writer = make_trace(tmp_path)
    before = store.get("traces", writer.trace_id)
    writer.manifest["invalid"] = float("nan")
    with pytest.raises(ValueError):
        store.save_manifest(writer.manifest)
    assert store.manifest(writer.trace_id) == before
    assert store.get("traces", writer.trace_id) == before


def test_create_in_permanent_root_is_owned_and_cannot_replace_existing_trace(tmp_path):
    store = Store(tmp_path / "data", tmp_path / "temporary")
    manifest = dict(title="import", level=2)
    trace_id = "e" * 32
    permanent = store.data_root / "retained/traces"
    path = store.create(trace_id, manifest, storage_root=permanent)
    assert path.parent == permanent.resolve()
    assert store.manifest(trace_id)["storage_root"] == str(permanent.resolve())
    assert (path / ".trace-owner").read_text() == trace_id
    with pytest.raises(ValueError, match="使用済み"):
        store.create(trace_id, {}, storage_root=tmp_path / "other")
    assert store.manifest(trace_id)["title"] == "import"


def test_connection_session_commits_each_chunk_and_closes_on_failure(tmp_path):
    store, writer = make_trace(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        with store.connection_session():
            with store.connection_session():  # Nested lifetimes must not close the outer connection.
                for index in range(3):
                    writer.write(np.full((2, 4), index), kind="residual", stage=f"layer.{index}")
                    observer = Store(store.data_root, store.trace_root, recover=False)
                    assert len(observer.get("traces", writer.trace_id)["tensors"]) == index + 1
                    assert observer.manifest(writer.trace_id) == observer.get("traces", writer.trace_id)
            raise RuntimeError("injected")
    reopened = Store(store.data_root, store.trace_root, recover=False)
    assert len(reopened.manifest(writer.trace_id)["tensors"]) == 3
    assert not hasattr(store._connections, "current")
    # A completed lifetime leaves no open WAL handle (also checked on Windows).
    assert not (store.data_root / "catalog.sqlite3-wal").exists()


def test_committed_chunks_recover_after_process_exits_with_open_connection(tmp_path):
    script = """
import os
import sys
from pathlib import Path
import numpy as np
from transformer_observatory.storage import Store, TraceWriter

root = Path(sys.argv[1])
store = Store(root / "data", root / "traces")
with store.connection_session():
    writer = TraceWriter(store, "a" * 32, dict(title="crash", level=2, model={}, tokens=[]))
    store.put("jobs", writer.trace_id, dict(id=writer.trace_id, state="capturing"))
    for index in range(3):
        writer.write(np.full((2, 4), index), kind="residual", stage=f"layer.{index}")
    os._exit(0)  # Bypass connection.close(), finally blocks, and normal WAL checkpointing.
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    recovered = Store(tmp_path / "data", tmp_path / "traces")
    manifest = recovered.manifest("a" * 32)
    assert manifest["status"] == "incomplete"
    assert recovered.get("jobs", "a" * 32)["state"] == "failed"
    assert len(manifest["tensors"]) == 3
    for index, record in enumerate(manifest["tensors"]):
        np.testing.assert_array_equal(recovered.load("a" * 32, record), np.full((2, 4), index))


def test_level_one_discards_raw_only_after_success(tmp_path):
    store, writer = make_trace(tmp_path, 1)
    r = writer.write(np.ones((2, 4)), kind="residual", stage="embedding")
    path = store.tensor_path(writer.trace_id, r)
    assert path.exists()
    writer.write(np.zeros((2, 3, 3), dtype=np.float32), kind="coordinates", stage="all")
    writer.finalize(dict(status="verified"))
    assert not path.exists()
    assert [r["kind"] for r in store.manifest(writer.trace_id)["tensors"]] == ["coordinates"]


def test_storage_budget_interrupt_and_recovery(tmp_path):
    store, writer = make_trace(tmp_path)
    writer.limit = 32
    with pytest.raises(OSError, match="容量上限"):
        writer.write(np.zeros((20, 5)), kind="residual", stage="embedding")
    store.put("jobs", writer.trace_id, dict(id=writer.trace_id, state="capturing"))
    reopened = Store(store.data_root, store.trace_root)
    assert reopened.get("jobs", writer.trace_id)["state"] == "failed"
    assert reopened.manifest(writer.trace_id)["status"] == "incomplete"


def test_cleanup_does_not_delete_retained_or_running(tmp_path):
    store, writer = make_trace(tmp_path)
    writer.manifest["created_at"] = 0
    writer.manifest["retained"] = True
    store.save_manifest(writer.manifest)
    assert store.cleanup_preview()["targets"] == []
    with pytest.raises(ValueError):
        store.delete(writer.trace_id)
    writer.manifest["retained"] = False
    store.save_manifest(writer.manifest)
    store.put("jobs", writer.trace_id, dict(id=writer.trace_id, state="capturing"))
    assert store.cleanup_preview()["targets"] == []
    store.put("jobs", writer.trace_id, dict(id=writer.trace_id, state="cancelled"))
    preview = store.cleanup_preview()
    assert len(preview["targets"]) == 1
    # Retaining after preview must still prevent deletion at confirmation.
    writer.manifest["retained"] = True
    store.save_manifest(writer.manifest)
    assert store.cleanup_confirm(preview["token"]) == []
    assert store.path(writer.trace_id).exists()


def test_unowned_and_traversal_paths_rejected(tmp_path):
    store, writer = make_trace(tmp_path)
    for trace_id in ["..", "../outside", "C:\\", "x" * 32]:
        with pytest.raises(ValueError):
            store.path(trace_id)
    (store.path(writer.trace_id) / ".trace-owner").unlink()
    with pytest.raises(ValueError):
        store.delete(writer.trace_id)


def test_retaining_moves_verified_copy_out_of_temp(tmp_path):
    store, writer = make_trace(tmp_path)
    writer.write(np.arange(12).reshape(3, 4), kind="residual", stage="embedding")
    writer.finalize(dict(status="verified"))
    source = store.path(writer.trace_id)
    m = store.retain(writer.trace_id, True)
    assert m["retained"]
    assert not source.exists()
    assert store.path(writer.trace_id).is_relative_to(store.data_root)
    np.testing.assert_array_equal(store.load(writer.trace_id, m["tensors"][0]), np.arange(12).reshape(3, 4))


def test_switching_default_root_preserves_previous_trace(tmp_path):
    store, writer = make_trace(tmp_path)
    writer.write(np.ones((2, 2)), kind="residual", stage="embedding")
    old = store.path(writer.trace_id)
    store.trace_root = store.validate_root(str(tmp_path / "new-root"))
    assert store.path(writer.trace_id) == old
    assert store.load(writer.trace_id, writer.manifest["tensors"][0]).shape == (2, 2)


def test_backend_catalog_lock_rejects_second_instance_and_releases(tmp_path):
    from transformer_observatory.runtime import InstanceLock

    first = InstanceLock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="起動済み"):
            InstanceLock(tmp_path)
    finally:
        first.close()
    reopened = InstanceLock(tmp_path)
    reopened.close()
