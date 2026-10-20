"""Versioned, bounded trace storage. No pickle or unowned recursive deletion."""

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

TRACE_ID = re.compile(r"^[a-f0-9]{32}$")
SCHEMA_VERSION = 1


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default_data_root():
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "TransformerObservatory"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "transformer-observatory"


def default_trace_root():
    if os.name == "nt":
        return Path(tempfile.gettempdir()) / "TransformerObservatory/traces"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "transformer-observatory/traces"


class Store:
    def __init__(self, data_root=None, trace_root=None):
        self.data_root = Path(data_root or os.environ.get("OBSERVATORY_DATA_DIR") or default_data_root())
        self.trace_root = Path(trace_root or os.environ.get("OBSERVATORY_TRACE_DIR") or default_trace_root())
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.trace_root.mkdir(parents=True, exist_ok=True)
        if self.trace_root.is_symlink() or self.trace_root.is_junction():
            raise ValueError("トレースのルートにリンクは指定できません")
        self.trace_root = self.trace_root.resolve()
        self.lock = threading.RLock()
        self.previews = {}
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            """)
        if not self.get("settings", "main"):
            self.put(
                "settings",
                "main",
                dict(retention_days=7, cache_limit_gib=20, trace_limit_gib=8, ram_budget_mib=512),
            )
        configured = self.settings.get("trace_root")
        if configured and not trace_root and not os.environ.get("OBSERVATORY_TRACE_DIR"):
            self.trace_root = self.validate_root(configured)
        # A restarted process cannot claim a vanished worker is still running.
        for job in self.list("jobs"):
            if job["state"] not in {"completed", "cancelled", "failed"}:
                job.update(state="failed", error="バックエンドが中断されました。部分トレースを確認できます。")
                self.put("jobs", job["id"], job)
                self.mark_incomplete(job["id"], job["error"])
        self.auto_cleanup()

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.data_root / "catalog.sqlite3", timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def put(self, table, key, data):
        assert table in {"traces", "jobs", "models", "settings"}
        with self.lock, self.db() as db:
            db.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
                (key, json.dumps(data, ensure_ascii=False, allow_nan=False)),
            )

    def get(self, table, key):
        assert table in {"traces", "jobs", "models", "settings"}
        with self.db() as db:
            row = db.execute(f"SELECT data FROM {table} WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, table):
        assert table in {"traces", "jobs", "models", "settings"}
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute(f"SELECT data FROM {table} ORDER BY rowid DESC")]

    @property
    def settings(self):
        return self.get("settings", "main")

    def path(self, trace_id, require_owner=True):
        if not TRACE_ID.fullmatch(trace_id):
            raise ValueError("無効なトレースID")
        metadata = self.get("traces", trace_id)
        root = Path(metadata.get("storage_root", str(self.trace_root))) if metadata else self.trace_root
        root = root.resolve()
        path = root / trace_id
        if path.is_symlink() or path.is_junction() or path.resolve().parent != root:
            raise ValueError("管理領域外のパス")
        if require_owner:
            owner = path / ".trace-owner"
            if owner.is_symlink() or not owner.is_file() or owner.read_text() != trace_id:
                raise ValueError("トレース所有マーカーが一致しません")
        return path

    def create(self, trace_id, manifest):
        with self.lock:
            path = self.path(trace_id, require_owner=False)
            path.mkdir(exist_ok=False)
            (path / ".trace-owner").write_text(trace_id)
            manifest.update(
                id=trace_id,
                schema_version=SCHEMA_VERSION,
                status="incomplete",
                retained=False,
                created_at=time.time(),
                tensors=[],
                stages=[],
                storage_root=str(self.trace_root),
            )
            atomic_json(path / "manifest.json", manifest)
            self.put("traces", trace_id, manifest)
            return path

    @staticmethod
    def validate_root(value):
        path = Path(value).expanduser()
        if not path.is_absolute() or path.resolve() in {Path(path.anchor), Path.home().resolve()}:
            raise ValueError("保存専用フォルダーの絶対パスを指定してください")
        for item in [path, *path.parents]:
            if item.is_symlink() or item.is_junction():
                raise ValueError("保存先のリンク・ジャンクションは使用できません")
        path.mkdir(parents=True, exist_ok=True)
        # Test actual write permission without touching existing files.
        with tempfile.TemporaryFile(dir=path) as f:
            f.write(b"observatory")
        return path.resolve()

    def retain(self, trace_id, enabled):
        with self.lock:
            m = self.manifest(trace_id)
            job = self.get("jobs", trace_id)
            if job and job["state"] not in {"completed", "cancelled", "failed"}:
                raise ValueError("実行中のトレースは保持先へ移動できません")
            if enabled and not m["retained"]:
                source = self.path(trace_id)
                destination_root = self.validate_root(str(self.data_root.resolve() / "retained/traces"))
                destination = destination_root / trace_id
                if destination.exists():
                    raise ValueError("保持先に同じIDが存在します")
                for p in source.rglob("*"):
                    if p.is_symlink() or p.is_junction():
                        raise ValueError("リンクを含むトレースは移動できません")
                size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
                if shutil.disk_usage(destination_root).free < size + 64 * 2**20:
                    raise OSError("保持先の容量が不足しています")
                shutil.copytree(source, destination)
                for r in m["tensors"]:
                    if digest(destination / r["file"]) != r["sha256"]:
                        raise ValueError("保持先コピーのハッシュが一致しません。元データは維持されています")
                m.update(retained=True, storage_root=str(destination_root))
                atomic_json(destination / "manifest.json", m)
                self.put("traces", trace_id, m)
                # Source ownership and path were checked before copying; source is removed only after verification.
                shutil.rmtree(source)
            else:
                m["retained"] = enabled
                self.save_manifest(m)
            return m

    def manifest(self, trace_id):
        path = self.path(trace_id)
        p = path / "manifest.json"
        if p.is_symlink():
            raise ValueError("不正なmanifest")
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("schema_version") != SCHEMA_VERSION or m.get("id") != trace_id:
            raise ValueError("未対応のトレース形式")
        return m

    def save_manifest(self, m):
        atomic_json(self.path(m["id"]) / "manifest.json", m)
        self.put("traces", m["id"], m)

    def mark_incomplete(self, trace_id, reason):
        if self.get("traces", trace_id):
            try:
                m = self.manifest(trace_id)
                m.update(status="incomplete", error=reason)
                self.save_manifest(m)
            except (ValueError, OSError):
                pass

    def tensor_path(self, trace_id, record):
        root = self.path(trace_id)
        rel = Path(record["file"])
        if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".npy":
            raise ValueError("不正なテンソルパス")
        path = root / rel
        if path.resolve().is_relative_to(root) is False:
            raise ValueError("トレース領域外のテンソル")
        for item in [path, *path.parents]:
            if item == root:
                break
            if item.is_symlink() or item.is_junction():
                raise ValueError("リンクされたテンソル")
        return path

    def load(self, trace_id, record):
        array = np.load(self.tensor_path(trace_id, record), allow_pickle=False, mmap_mode="r")
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError("テンソルの形状・dtypeがmanifestと一致しません")
        return array

    def delete(self, trace_id):
        with self.lock:
            m = self.manifest(trace_id)
            j = self.get("jobs", trace_id)
            if m["retained"] or (j and j["state"] not in {"completed", "cancelled", "failed"}):
                raise ValueError("保持中・実行中のトレースは削除できません")
            root = self.path(trace_id)
            for p in root.rglob("*"):
                if p.is_symlink() or p.is_junction():
                    raise ValueError("リンクを含むトレースの削除を拒否しました")
            shutil.rmtree(root)
            with self.db() as db:
                db.execute("DELETE FROM traces WHERE id=?", (trace_id,))

    def cleanup_preview(self):
        now = time.time()
        candidates = []
        total = 0
        for m in reversed(self.list("traces")):
            if m.get("retained"):
                continue
            j = self.get("jobs", m["id"])
            if j and j["state"] not in {"completed", "cancelled", "failed"}:
                continue
            try:
                root = self.path(m["id"])
                size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
                candidates.append((m, size))
                total += size
            except (ValueError, OSError):
                continue
        targets = []
        for m, size in candidates:
            expired = now - m["created_at"] > self.settings["retention_days"] * 86400
            over = total > self.settings["cache_limit_gib"] * 2**30
            if expired or over:
                targets.append(dict(id=m["id"], title=m.get("title", ""), bytes=size))
                total -= size
        token = uuid.uuid4().hex
        self.previews = {k: v for k, v in self.previews.items() if now - v[0] < 300}
        self.previews[token] = (now, [x["id"] for x in targets])
        return dict(token=token, targets=targets, bytes=sum(x["bytes"] for x in targets))

    def cleanup_confirm(self, token):
        with self.lock:
            preview = self.previews.pop(token, None)
            if not preview or time.time() - preview[0] > 300:
                raise ValueError("削除プレビューの有効期限が切れました")
            removed = []
            for trace_id in preview[1]:
                try:
                    self.delete(trace_id)
                    removed.append(trace_id)
                except ValueError:
                    continue
            return removed

    def auto_cleanup(self):
        """Apply the configured TTL/cap to owned, unretained, inactive traces only."""
        preview = self.cleanup_preview()
        return self.cleanup_confirm(preview["token"])


class TraceWriter:
    def __init__(self, store, trace_id, manifest):
        self.store, self.trace_id, self.manifest = store, trace_id, manifest
        self.root = store.create(trace_id, manifest)
        self.bytes = 0
        self.limit = int(store.settings["trace_limit_gib"] * 2**30)
        self.counter = 0

    def write(self, array, *, kind, stage, layer=-1, step=0, positions=None, space="residual", heads=None):
        a = np.asarray(array)
        if a.dtype.hasobject or not np.isfinite(a).all():
            raise ValueError("非有限値・objectテンソルは保存できません")
        if self.bytes + a.nbytes > self.limit:
            raise OSError("トレース容量上限に達しました")
        if shutil.disk_usage(self.root).free < a.nbytes + 64 * 2**20:
            raise OSError("ディスク空き容量が不足しています")
        folder = self.root / ("residual" if kind == "residual" else "tensors")
        folder.mkdir(exist_ok=True)
        name = f"{self.counter:07d}.npy"
        self.counter += 1
        path = folder / name
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.save(f, a, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self.bytes += path.stat().st_size
        record = dict(
            file=path.relative_to(self.root).as_posix(),
            kind=kind,
            stage=stage,
            layer=layer,
            step=step,
            positions=positions,
            space=space,
            heads=heads,
            shape=list(a.shape),
            dtype=str(a.dtype),
            sha256=digest(path),
        )
        self.manifest["tensors"].append(record)
        if kind == "residual" and stage not in self.manifest["stages"]:
            self.manifest["stages"].append(stage)
        # Journal every chunk so cancellation/crashes retain valid partial metadata.
        self.store.save_manifest(self.manifest)
        return record

    def finalize(self, integrity):
        if self.manifest["level"] == 1:
            keep = []
            for r in self.manifest["tensors"]:
                if r["kind"] in {"residual", "tensor"}:
                    self.store.tensor_path(self.trace_id, r).unlink()
                else:
                    keep.append(r)
            self.manifest["tensors"] = keep
        self.manifest.update(status="completed", integrity=integrity, completed_at=time.time())
        self.manifest["size_bytes"] = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        atomic_json(self.root / "integrity.json", integrity)
        self.store.save_manifest(self.manifest)
