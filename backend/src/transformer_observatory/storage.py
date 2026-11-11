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
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

TRACE_ID = re.compile(r"^[a-f0-9]{32}$")
SCHEMA_VERSION = 1


def relative_tensor_path(record):
    value = record["file"]
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ValueError("不正なテンソルパス")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".npy":
        raise ValueError("不正なテンソルパス")
    return rel


def check_present_chunks(root, records):
    """Check referenced entries once per directory, without following filesystem links.

    The trie is local to this request: missing/changed files are never hidden by a TTL cache.
    Unreferenced files are ignored, matching the per-record presence check.
    """
    tree = {}
    for record in records:
        parts = relative_tensor_path(record).parts
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if node is None:
                raise ValueError("テンソルのファイルとフォルダーが競合しています")
        if parts[-1] in node and node[parts[-1]] is not None:
            raise ValueError("テンソルのファイルとフォルダーが競合しています")
        node[parts[-1]] = None
    pending = [(root, tree)] if tree else []
    while pending:
        folder, expected = pending.pop()
        with os.scandir(folder) as scan:
            entries = {entry.name: entry for entry in scan}
        aliases = (
            {os.path.normcase(name): entry for name, entry in entries.items()} if os.name == "nt" else {}
        )
        for name, children in expected.items():
            entry = entries.get(name)
            if entry is None and os.name == "nt":
                # Respect case-insensitive aliases, but also case-sensitive Windows directories.
                candidate = aliases.get(os.path.normcase(name))
                if candidate is not None and (folder / name).exists():
                    entry = candidate
            if entry is None:
                raise FileNotFoundError("保存したテンソルファイルが見つかりません")
            if entry.is_symlink() or (os.name == "nt" and entry.is_junction()):
                raise ValueError("リンクされたテンソル")
            if children is None:
                if not entry.is_file(follow_symlinks=False):
                    raise FileNotFoundError("保存したテンソルファイルが見つかりません")
            else:
                if not entry.is_dir(follow_symlinks=False):
                    raise FileNotFoundError("保存したテンソルフォルダーが見つかりません")
                pending.append((Path(entry.path), children))


def replace_json_file(source, destination):
    # Windows readers/scanners may briefly hold a handle without FILE_SHARE_DELETE.
    # Keep the previous file intact; persistent permission errors must still fail.
    delays = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def atomic_json(path: Path, data):
    # Encode once with the C encoder instead of streaming Python encoder tokens.
    _atomic_json_text(path, json.dumps(data, ensure_ascii=False, allow_nan=False))


def _atomic_json_text(path: Path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        replace_json_file(tmp, path)
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
    def __init__(self, data_root=None, trace_root=None, *, recover=True):
        self.data_root = Path(data_root or os.environ.get("OBSERVATORY_DATA_DIR") or default_data_root())
        self.trace_root = Path(trace_root or os.environ.get("OBSERVATORY_TRACE_DIR") or default_trace_root())
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.trace_root.mkdir(parents=True, exist_ok=True)
        if self.trace_root.is_symlink() or self.trace_root.is_junction():
            raise ValueError("トレースのルートにリンクは指定できません")
        self.trace_root = self.trace_root.resolve()
        self.lock = threading.RLock()
        self._connections = threading.local()
        self.previews = {}
        self.roots = {}
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS studies (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS lenses (id TEXT PRIMARY KEY, data TEXT NOT NULL);
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
        if recover:
            # A restarted process cannot claim a vanished worker is still running.
            for job in self.list("jobs"):
                if job["state"] not in {"completed", "cancelled", "failed"}:
                    job.update(
                        state="failed", error="バックエンドが中断されました。部分トレースを確認できます。"
                    )
                    self.put("jobs", job["id"], job)
                    self.mark_incomplete(job["id"], job["error"])
            for study in self.list("studies"):
                if study["state"] not in {"completed", "cancelled", "failed"}:
                    study.update(
                        state="failed",
                        error="バックエンドが中断されました。確定済み研究レポートは維持されています。",
                    )
                    self.put("studies", study["id"], study)
            self.auto_cleanup()

    @contextmanager
    def db(self):
        conn = getattr(self._connections, "current", None)
        owned = conn is None
        if owned:
            conn = sqlite3.connect(self.data_root / "catalog.sqlite3", timeout=30)
        try:
            if owned:
                conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                yield conn
        finally:
            if owned:
                conn.close()

    @contextmanager
    def connection_session(self):
        """Reuse this thread's connection, committing each db() call independently.

        Keeping WAL open avoids a last-connection checkpoint after every chunk.
        This is a connection lifetime, never a transaction around the whole job.
        """
        if getattr(self._connections, "current", None) is not None:
            yield
            return
        conn = sqlite3.connect(self.data_root / "catalog.sqlite3", timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            self._connections.current = conn
            yield
        finally:
            self._connections.__dict__.pop("current", None)
            conn.close()

    def put(self, table, key, data):
        with self.lock:
            self._put_json(table, key, json.dumps(data, ensure_ascii=False, allow_nan=False))

    def _put_json(self, table, key, text):
        assert table in {"traces", "jobs", "models", "settings", "studies", "lenses"}
        with self.lock, self.db() as db:
            db.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?, ?)",
                (key, text),
            )

    def get(self, table, key):
        assert table in {"traces", "jobs", "models", "settings", "studies", "lenses"}
        with self.db() as db:
            row = db.execute(f"SELECT data FROM {table} WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, table):
        assert table in {"traces", "jobs", "models", "settings", "studies", "lenses"}
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute(f"SELECT data FROM {table} ORDER BY rowid DESC")]

    @property
    def settings(self):
        return self.get("settings", "main")

    def path(self, trace_id, require_owner=True):
        if not TRACE_ID.fullmatch(trace_id):
            raise ValueError("無効なトレースID")
        root = self.roots.get(trace_id)
        if root is None:
            metadata = self.get("traces", trace_id)
            root = Path(metadata.get("storage_root", str(self.trace_root))) if metadata else self.trace_root
            if metadata:
                self.roots[trace_id] = root
        for item in [root, *root.parents]:
            if item.is_symlink() or item.is_junction():
                raise ValueError("リンクされた保存領域")
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
            self.roots[trace_id] = self.trace_root
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
            if self.trace_busy(trace_id):
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
                self.roots[trace_id] = destination_root
                # Source ownership and path were checked before copying; source is removed only after verification.
                shutil.rmtree(source)
            else:
                m["retained"] = enabled
                self.save_manifest(m)
            return m

    def manifest(self, trace_id):
        with self.lock:
            return self._manifest_at(self.path(trace_id), trace_id)

    @staticmethod
    def _manifest_at(path, trace_id):
        p = path / "manifest.json"
        if p.is_symlink():
            raise ValueError("不正なmanifest")
        m = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(m, dict) or m.get("schema_version") != SCHEMA_VERSION or m.get("id") != trace_id:
            raise ValueError("未対応のトレース形式")
        return m

    def save_manifest(self, m):
        with self.lock:
            text = json.dumps(m, ensure_ascii=False, allow_nan=False)
            _atomic_json_text(self.path(m["id"]) / "manifest.json", text)
            self._put_json("traces", m["id"], text)

    def availability(self, trace_id):
        """Reachable manifest and present chunks, separate from full integrity verification."""
        try:
            with self.lock:
                root = self.path(trace_id)
                manifest = self._manifest_at(root, trace_id)
                check_present_chunks(root, manifest["tensors"])
                self.path(trace_id)  # Recheck ownership/root after enumerating the directories.
            return dict(available=True, availability_error=None)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return dict(available=False, availability_error=str(exc))

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
        rel = relative_tensor_path(record)
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

    def trace_busy(self, trace_id):
        terminal = {"completed", "cancelled", "failed"}
        job = self.get("jobs", trace_id)
        return bool(job and job["state"] not in terminal) or any(
            study["state"] not in terminal and trace_id in study["request"]["trace_ids"]
            for study in self.list("studies")
        )

    def delete(self, trace_id):
        with self.lock:
            m = self.manifest(trace_id)
            if m["retained"] or self.trace_busy(trace_id):
                raise ValueError("保持中・実行中のトレースは削除できません")
            root = self.path(trace_id)
            for p in root.rglob("*"):
                if p.is_symlink() or p.is_junction():
                    raise ValueError("リンクを含むトレースの削除を拒否しました")
            shutil.rmtree(root)
            self.roots.pop(trace_id, None)
            with self.db() as db:
                db.execute("DELETE FROM traces WHERE id=?", (trace_id,))

    def cleanup_preview(self):
        now = time.time()
        candidates = []
        total = 0
        for m in reversed(self.list("traces")):
            if m.get("retained"):
                continue
            if self.trace_busy(m["id"]):
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
    def __init__(self, store, trace_id, manifest, diagnostics=None):
        self.store, self.trace_id, self.manifest = store, trace_id, manifest
        self.diagnostics = diagnostics
        self.root = store.create(trace_id, manifest)
        self.bytes = 0
        self.limit = int(store.settings["trace_limit_gib"] * 2**30)
        self.counter = 0

    def _measure(self, operation):
        return self.diagnostics.measure(operation) if self.diagnostics else nullcontext()

    def write(
        self,
        array,
        *,
        kind,
        stage,
        layer=-1,
        step=0,
        positions=None,
        space="residual",
        heads=None,
        provenance="derived",
    ):
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
        with self._measure("chunk_write"):
            with tmp.open("wb") as f:
                np.save(f, a, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        self.bytes += path.stat().st_size
        with self._measure("chunk_hash"):
            checksum = digest(path)
        record = dict(
            file=path.relative_to(self.root).as_posix(),
            kind=kind,
            stage=stage,
            layer=layer,
            step=step,
            positions=positions,
            space=space,
            heads=heads,
            provenance=provenance,
            shape=list(a.shape),
            dtype=str(a.dtype),
            sha256=checksum,
        )
        self.manifest["tensors"].append(record)
        if kind == "residual" and stage not in self.manifest["stages"]:
            self.manifest["stages"].append(stage)
        # Journal every chunk so cancellation/crashes retain valid partial metadata.
        with self._measure("manifest_save"):
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
