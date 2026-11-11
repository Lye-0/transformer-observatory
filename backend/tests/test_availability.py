"""Presence checks must stay current and safe without repeating root IO for every chunk."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from transformer_observatory.storage import Store


def fixture(tmp_path, files):
    store = Store(tmp_path / "data", tmp_path / "traces")
    manifest = dict(title="presence")
    trace_id = "a" * 32
    root = store.create(trace_id, manifest)
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    manifest["tensors"] = [dict(file=name) for name in files]
    store.save_manifest(manifest)
    return store, root, trace_id


def test_bulk_presence_checks_directories_once_and_never_cache_missing_files(tmp_path, monkeypatch):
    store, root, trace_id = fixture(tmp_path, [f"tensors/{i}.npy" for i in range(200)])
    paths, scans = [], []
    original_path, original_scan = store.path, os.scandir

    def path(*args, **kwargs):
        paths.append(args)
        return original_path(*args, **kwargs)

    def scan(folder):
        scans.append(Path(folder))
        return original_scan(folder)

    monkeypatch.setattr(store, "path", path)
    monkeypatch.setattr(os, "scandir", scan)
    assert store.availability(trace_id)["available"]
    assert len(paths) == 2
    assert set(scans) == {root, root / "tensors"}
    missing = root / "tensors/199.npy"
    missing.unlink()
    assert not store.availability(trace_id)["available"]
    missing.touch()
    assert store.availability(trace_id)["available"]
    (root / ".trace-owner").write_text("different")
    assert not store.availability(trace_id)["available"]


@pytest.mark.parametrize(
    "name",
    [
        "../outside.npy",
        "/outside.npy",
        "C:/outside.npy",
        "tensors\\file.npy",
        "a.npy:stream.npy",
        "wrong.txt",
    ],
)
def test_presence_rejects_escaping_and_non_npy_paths(tmp_path, name):
    store, _, trace_id = fixture(tmp_path, [])
    m = store.manifest(trace_id)
    m["tensors"] = [dict(file=name)]
    store.save_manifest(m)
    assert not store.availability(trace_id)["available"]


def test_nested_chunks_and_platform_case_semantics(tmp_path):
    store, root, trace_id = fixture(tmp_path, ["tensors/nested/chunk.npy"])
    assert store.availability(trace_id)["available"]
    (root / "tensors/nested/chunk.npy").rename(root / "tensors/nested/CHUNK.NPY")
    assert store.availability(trace_id)["available"] is (os.name == "nt")


@pytest.mark.parametrize("value", [None, [], {"schema_version": 99}, {"schema_version": 1, "id": "wrong"}])
def test_malformed_manifest_is_unavailable(tmp_path, value):
    store, root, trace_id = fixture(tmp_path, [])
    (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    assert not store.availability(trace_id)["available"]


def test_presence_rejects_linked_parent_folder(tmp_path):
    store, root, trace_id = fixture(tmp_path, [])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "chunk.npy").touch()
    link = root / "linked"
    if os.name == "nt":
        # A junction does not require Developer Mode or symlink privileges.
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        assert result.returncode == 0, result.stderr
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        m = store.manifest(trace_id)
        m["tensors"] = [dict(file="linked/chunk.npy")]
        store.save_manifest(m)
        assert not store.availability(trace_id)["available"]
    finally:
        # Remove only the link itself, never its target or contents.
        if os.name == "nt":
            link.rmdir()
        else:
            link.unlink()
    assert (outside / "chunk.npy").exists()
