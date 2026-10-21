"""Bounded local discovery and folder browsing. Discovery never loads or hashes weights."""

import copy
import json
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path


def local_path(value):
    path = Path(str(value).strip().strip('"')).expanduser()
    if not path.is_absolute():
        raise ValueError("この環境の絶対パスを指定してください")
    return path.resolve()


def path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def inspect_model(value):
    root = local_path(value)
    name = root.name
    if root.parent.name == "snapshots" and root.parent.parent.name.startswith("models--"):
        name = root.parent.parent.name.removeprefix("models--").replace("--", "/")
    result = dict(path=str(root), name=name, architecture=None, status="invalid", reason="", weight_bytes=0)
    try:
        if not root.is_dir():
            raise ValueError("フォルダーが見つかりません。保存場所を確認してください")
        config_path = root / "config.json"
        if not config_path.is_file():
            raise ValueError("config.jsonを含むモデルフォルダーを指定してください")
        if config_path.stat().st_size > 2**20:
            raise ValueError("config.jsonが大きすぎます（上限1MiB）")
        config = json.loads(config_path.read_bytes())
        if not isinstance(config, dict):
            raise ValueError("config.jsonはJSONオブジェクトである必要があります")
        architecture = config.get("model_type")
        result["architecture"] = architecture
        if architecture not in {"qwen3", "gpt2"}:
            result["status"] = "unsupported"
            raise ValueError(f"未対応のモデル構造: {architecture or '不明'}（現在はQwen3／GPT-2）")
        if config.get("quantization_config"):
            result["status"] = "unsupported"
            raise ValueError("量子化モデルの内部取得は現在非対応です")
        if config.get("sliding_window") and config.get("use_sliding_window"):
            result["status"] = "unsupported"
            raise ValueError("Sliding-window Attentionの内部取得は未検証です")
        dimensions = dict(
            hidden_size=config.get("hidden_size", config.get("n_embd")),
            layers=config.get("num_hidden_layers", config.get("n_layer")),
            heads=config.get("num_attention_heads", config.get("n_head")),
        )
        if any(type(v) is not int or v <= 0 for v in dimensions.values()):
            raise ValueError("モデルの次元・層数・Head数の設定が不足しています")
        result.update(dimensions)
        weights = sorted(root.glob("*.safetensors"))
        if not weights:
            raise ValueError("safetensorsの重みがありません。ダウンロード完了と形式を確認してください")
        index = root / "model.safetensors.index.json"
        if index.is_file():
            if index.stat().st_size > 8 * 2**20:
                raise ValueError("重みのindexが大きすぎます")
            data = json.loads(index.read_bytes())
            mapping = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("重みのindexにweight_mapがありません")
            if any(not isinstance(filename, str) for filename in mapping.values()):
                raise ValueError("重みのindexに不正なファイル名があります")
            for filename in set(mapping.values()):
                if Path(filename).name != filename:
                    raise ValueError("重みのindexに不正なファイル名があります")
                if not filename.endswith(".safetensors") or not (root / filename).is_file():
                    raise ValueError(f"分割重みが不足しています: {filename}")
        elif len(weights) > 1 and not (root / "model.safetensors").is_file():
            raise ValueError("分割重みのmodel.safetensors.index.jsonがありません")
        if any(file.stat().st_size == 0 for file in weights):
            raise ValueError("空の重みファイルがあります。ダウンロード完了を確認してください")
        if not (root / "tokenizer.json").is_file() and not (
            (root / "vocab.json").is_file() and (root / "merges.txt").is_file()
        ):
            raise ValueError(
                "Tokenizerのファイルが不足しています（tokenizer.json または vocab.json＋merges.txt）"
            )
        result.update(
            status="available",
            reason="必要なファイルが見つかりました。登録時に内容を照合します",
            weight_bytes=sum(file.stat().st_size for file in weights),
        )
    except (OSError, ValueError, TypeError) as exc:
        result["reason"] = str(exc)
    return result


def search_roots(store):
    repo = Path(__file__).resolve().parents[3]
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    default_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    home = os.environ.get("HF_HOME") or str(default_home)
    hub = Path(os.path.expandvars(os.path.expanduser(cache or str(Path(home) / "hub"))))
    entries = [
        ("プロジェクトのモデル", repo / ".tools/models"),
        ("modelsフォルダー", repo / "models"),
        ("Hugging Faceキャッシュ", hub),
        *[
            ("追加した探索先", Path(p))
            for p in (store.get("settings", "model_search") or {}).get("paths", [])
        ],
    ]
    seen, roots = set(), []
    for label, path in entries:
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            roots.append(dict(label=label, path=str(path.resolve()), exists=path.is_dir()))
    return roots


def browse_folders(value, store):
    root = local_path(value) if value else Path.home()
    if not root.is_dir():
        raise ValueError("フォルダーが見つかりません")
    roots = [dict(label="ホーム", path=str(Path.home())), *search_roots(store)]
    if os.name == "nt":
        roots.extend(dict(label=drive, path=drive) for drive in os.listdrives())
    else:
        roots.append(dict(label="ファイルシステム", path="/"))
    entries, truncated = [], False
    try:
        with os.scandir(root) as items:
            for item in items:
                if item.is_dir(follow_symlinks=False) and not item.is_symlink() and not item.is_junction():
                    entries.append(dict(name=item.name, path=str(Path(item.path))))
                    if len(entries) >= 300:
                        truncated = True
                        break
    except OSError as exc:
        raise ValueError("このフォルダーを読み取れません。権限または保存場所を確認してください") from exc
    return dict(
        path=str(root),
        parent=str(root.parent),
        roots=roots,
        entries=sorted(entries, key=lambda item: item["name"].casefold()),
        truncated=truncated,
    )


class ModelDiscovery:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.active = None
        self.thread = None
        self.cancelled = threading.Event()

    def start(self, roots):
        with self.lock:
            if self.active and self.jobs[self.active]["state"] == "running":
                return copy.deepcopy(self.jobs[self.active])
            job_id = uuid.uuid4().hex
            self.cancelled = threading.Event()
            job = dict(
                id=job_id,
                state="running",
                roots=roots,
                candidates=[],
                scanned=0,
                current="",
                warnings=[],
                truncated=False,
            )
            self.jobs[job_id] = job
            self.active = job_id
            while len(self.jobs) > 8:
                del self.jobs[next(iter(self.jobs))]
            self.thread = threading.Thread(target=self._scan, args=(job, self.cancelled), daemon=True)
            self.thread.start()
            return copy.deepcopy(job)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("モデル検出の記録が見つかりません。再検出してください")
            return copy.deepcopy(self.jobs[job_id])

    def cancel(self, job_id):
        with self.lock:
            if job_id == self.active:
                self.cancelled.set()
        return self.get(job_id)

    def close(self):
        self.cancelled.set()
        if self.thread:
            self.thread.join(timeout=2)

    def _scan(self, job, cancelled):
        queue = deque((Path(root["path"]), 0) for root in job["roots"])
        seen, started = set(), time.monotonic()
        skip = {".git", ".venv", "node_modules", "blobs", "__pycache__", "traces"}
        try:
            while queue and not cancelled.is_set():
                if job["scanned"] >= 2000 or len(job["candidates"]) >= 200 or time.monotonic() - started > 15:
                    with self.lock:
                        job["truncated"] = True
                    break
                path, depth = queue.popleft()
                key = path_key(path)
                if key in seen:
                    continue
                seen.add(key)
                with self.lock:
                    job.update(current=str(path), scanned=job["scanned"] + 1)
                try:
                    if not path.is_dir() or path.is_symlink() or path.is_junction():
                        continue
                    if (path / "config.json").is_file():
                        candidate = inspect_model(path)
                        with self.lock:
                            job["candidates"].append(candidate)
                        continue
                    if depth < 4:
                        with os.scandir(path) as items:
                            for item in items:
                                if cancelled.is_set():
                                    break
                                if len(queue) >= 2000:
                                    with self.lock:
                                        job["truncated"] = True
                                    break
                                if (
                                    item.name not in skip
                                    and item.is_dir(follow_symlinks=False)
                                    and not item.is_junction()
                                ):
                                    queue.append((Path(item.path), depth + 1))
                except OSError:
                    with self.lock:
                        if len(job["warnings"]) < 30:
                            job["warnings"].append(f"読み取れないフォルダーをスキップしました: {path}")
            with self.lock:
                job.update(state="cancelled" if cancelled.is_set() else "completed", current="")
        except Exception:
            with self.lock:
                job.update(
                    state="failed", current="", error="モデル検出に失敗しました。探索先を確認してください"
                )
