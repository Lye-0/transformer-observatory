import asyncio
import json
import os
import secrets
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .analysis import compare
from .folders import open_folder
from .jobs import Jobs
from .models import register_model
from .runtime import InstanceLock
from .schema import (
    TERMINAL,
    AnalysisRequest,
    ComparisonRequest,
    ModelRegistration,
    SettingsUpdate,
    TraceMetadataUpdate,
)
from .storage import SCHEMA_VERSION, Store, atomic_json, default_data_root, digest
from .trace_validation import validate_finite, validate_manifest


def create_app(store=None, testing=False):
    directory = (
        store.data_root if store else Path(os.environ.get("OBSERVATORY_DATA_DIR") or default_data_root())
    )
    guard = None if testing else InstanceLock(directory)
    try:
        store = store or Store()
        jobs = Jobs(store)
    except Exception:
        if guard:
            guard.close()
        raise
    session = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        async def maintenance():
            while True:
                await asyncio.sleep(1800)
                await asyncio.to_thread(store.auto_cleanup)

        task = asyncio.create_task(maintenance())
        try:
            yield
        finally:
            task.cancel()
            jobs.close()
            if guard:
                guard.close()

    app = FastAPI(title="Transformer Observatory", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store, app.state.jobs = store, jobs

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        host = request.headers.get("host", "")
        hostname = urlsplit("http://" + host).hostname
        allowed = {"127.0.0.1", "localhost", "::1"} | ({"testserver"} if testing else set())
        if hostname not in allowed:
            return JSONResponse({"detail": "Loopback Hostのみ許可します"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{host}":
            return JSONResponse({"detail": "Originが一致しません"}, status_code=403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/session":
            if request.cookies.get("observatory_session") != session:
                return JSONResponse({"detail": "ローカルセッションを再接続してください"}, status_code=401)
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                if not secrets.compare_digest(request.headers.get("x-observatory-token", ""), session):
                    return JSONResponse({"detail": "操作トークンが一致しません"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request, exc):
        return JSONResponse(
            {"detail": "ファイルが見つかりません。一時データが削除された可能性があります。"}, status_code=404
        )

    @app.get("/api/session")
    def begin_session():
        response = JSONResponse(dict(token=session, version="0.1.0", backend_os=os.name))
        response.set_cookie("observatory_session", session, httponly=True, samesite="strict")
        return response

    @app.get("/api/models")
    def models():
        return jobs.models()

    @app.post("/api/models/register")
    def register(body: ModelRegistration):
        m = register_model(body.path, body.name)
        store.put("models", m["id"], m)
        return m

    @app.get("/api/models/{model_id}/capabilities")
    def capabilities(model_id: str):
        return jobs.model(model_id)["capabilities"]

    @app.get("/api/models/discover-local")
    def discover_local():
        repo = Path(__file__).resolve().parents[3]
        candidate = repo / ".tools/models/Qwen3-0.6B"
        return dict(paths=[str(candidate)] if candidate.is_dir() else [])

    @app.post("/api/analysis/preflight")
    def preflight(body: AnalysisRequest):
        return jobs.preflight(body)

    @app.post("/api/analysis/jobs", status_code=202)
    def submit(body: AnalysisRequest):
        return jobs.submit(body)

    @app.get("/api/analysis/jobs")
    def job_list():
        return store.list("jobs")[:100]

    @app.get("/api/analysis/jobs/{job_id}")
    def job_status(job_id: str):
        job = store.get("jobs", job_id)
        if not job:
            raise HTTPException(404, "ジョブが見つかりません")
        return job

    @app.post("/api/analysis/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return jobs.cancel(job_id)

    @app.get("/api/analysis/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        job_status(job_id)

        async def stream():
            last = None
            while not await request.is_disconnected():
                job = store.get("jobs", job_id)
                encoded = json.dumps(job, ensure_ascii=False)
                if encoded != last:
                    yield f"data: {encoded}\n\n"
                    last = encoded
                if job["state"] in TERMINAL:
                    break
                await asyncio.sleep(0.4)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/api/traces")
    def traces():
        return [
            dict(
                id=m["id"],
                title=m["title"],
                level=m["level"],
                status=m["status"],
                model=m["model"]["name"],
                created_at=m["created_at"],
                retained=m["retained"],
                size_bytes=m.get("size_bytes", 0),
                token_count=len(m["tokens"]),
                integrity=m.get("integrity"),
                error=m.get("error"),
                description=m.get("description", ""),
                tags=m.get("tags", []),
                active=(store.get("jobs", m["id"]) or {}).get("state", "completed") not in TERMINAL,
            )
            for m in store.list("traces")
        ]

    @app.post("/api/traces/compare")
    def comparison(body: ComparisonRequest):
        return compare(store, body.left, body.right)

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str):
        return store.manifest(trace_id)

    @app.patch("/api/traces/{trace_id}")
    def update_trace(trace_id: str, body: TraceMetadataUpdate):
        with store.lock:
            if (store.get("jobs", trace_id) or {}).get("state", "completed") not in TERMINAL:
                raise ValueError("実行中のトレースは編集できません")
            m = store.manifest(trace_id)
            m.update(body.model_dump())
            store.save_manifest(m)
        return dict(saved=True)

    @app.post("/api/storage/open")
    def open_storage(body: dict):
        path = store.path(str(body["trace_id"])) if body.get("trace_id") else store.trace_root
        open_folder(path)
        return dict(path=str(path))

    def find_record(trace_id, kind):
        m = store.manifest(trace_id)
        record = next((r for r in m["tensors"] if r["kind"] == kind), None)
        if not record:
            raise HTTPException(404, "このデータは未取得です")
        return m, record

    @app.get("/api/traces/{trace_id}/coordinates")
    def coordinates(trace_id: str, view: str = "scientific"):
        kind = "coordinates" if view == "scientific" else "displacement"
        _, r = find_record(trace_id, kind)
        a = store.load(trace_id, r)
        return Response(
            a.astype("<f4").tobytes(),
            media_type="application/octet-stream",
            headers={"X-Array-Shape": json.dumps(r["shape"]), "X-Array-Dtype": "<f4"},
        )

    @app.get("/api/traces/{trace_id}/metrics")
    def metrics(trace_id: str, token: int = Query(0, ge=0)):
        m, r = find_record(trace_id, "metrics")
        a = store.load(trace_id, r)
        if token >= len(a):
            raise ValueError("トークン位置が範囲外です")
        values = a[token].tolist()
        valid_record = next((r for r in m["tensors"] if r["kind"] == "metric_validity"), None)
        if valid_record:
            validity = store.load(trace_id, valid_record)[token]
            values = [
                [v if validity[s, k] else None for k, v in enumerate(row)] for s, row in enumerate(values)
            ]
        return dict(names=m["metric_names"], stages=m["stages"], values=values)

    @app.get("/api/traces/{trace_id}/metric-values")
    def metric_values(trace_id: str):
        _, r = find_record(trace_id, "metrics")
        return Response(
            store.load(trace_id, r).astype("<f4").tobytes(), media_type="application/octet-stream"
        )

    @app.post("/api/traces/{trace_id}/lens")
    def lens(trace_id: str, body: dict):
        try:
            token, step, stage = int(body["token"]), int(body["step"]), str(body["stage"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("token・step・stageを指定してください") from exc
        return jobs.executor.submit(jobs.lens, trace_id, stage, token, step).result()

    @app.get("/api/traces/{trace_id}/tensor")
    def tensor(
        trace_id: str,
        kind: str = "residual",
        stage: str = "embedding",
        layer: int = -1,
        step: int = Query(0, ge=0),
        token: int = Query(0, ge=0),
        name: str = "",
    ):
        m = store.manifest(trace_id)
        matches = [
            r
            for r in m["tensors"]
            if r["kind"] == kind
            and r["step"] == step
            and (
                (kind == "residual" and r["stage"] == stage)
                or (kind == "tensor" and r["layer"] == layer and (not name or r["stage"] == name))
            )
        ]
        if not matches:
            return dict(status="未取得", tensors=[])
        data = []
        for r in matches:
            a = store.load(trace_id, r)
            pos = r["positions"] or []
            if token not in pos:
                continue
            index = pos.index(token)
            # Attention matrix layout is H x query x key. Other tensors are token-major.
            value = a[:, index] if r["stage"] == "attention_weights" else a[index]
            flat = np.asarray(value).reshape(-1)
            data.append(
                dict(
                    name=r["stage"],
                    space=r["space"],
                    shape=list(value.shape),
                    heads=r.get("heads"),
                    provenance=r.get("provenance", "unspecified"),
                    min=float(flat.min()),
                    max=float(flat.max()),
                    mean=float(flat.mean()),
                    std=float(flat.std()),
                    norm=float(np.linalg.norm(flat)),
                    values=flat[:16384].tolist(),
                    truncated=len(flat) > 16384,
                    dimensions=int(len(flat)),
                )
            )
        return dict(status="取得済み" if data else "未取得", tensors=data)

    @app.post("/api/traces/{trace_id}/retain")
    def retain(trace_id: str, enabled: bool = True):
        m = store.retain(trace_id, enabled)
        return dict(retained=m["retained"], storage_root=m["storage_root"])

    @app.delete("/api/traces/{trace_id}")
    def delete(trace_id: str, request: Request):
        if request.headers.get("x-confirm-trace") != trace_id:
            raise HTTPException(400, "削除対象の確認が必要です")
        store.delete(trace_id)
        return dict(deleted=True)

    @app.post("/api/storage/cleanup/preview")
    def cleanup_preview():
        return store.cleanup_preview()

    @app.post("/api/storage/cleanup/confirm")
    def cleanup_confirm(body: dict):
        return dict(deleted=store.cleanup_confirm(body.get("token", "")))

    @app.get("/api/settings")
    def settings():
        return dict(
            **store.settings,
            data_root=str(store.data_root),
            trace_root=str(store.trace_root),
            backend_os=platform_name(),
            free_disk_bytes=__import__("shutil").disk_usage(store.trace_root).free,
        )

    @app.patch("/api/settings")
    def update_settings(body: SettingsUpdate):
        values = body.model_dump()
        with store.lock:
            if body.trace_root:
                values["trace_root"] = str(store.validate_root(body.trace_root))
            else:
                values["trace_root"] = str(store.trace_root)
            store.put("settings", "main", values)
            store.trace_root = Path(values["trace_root"])
        return settings()

    @app.get("/api/traces/{trace_id}/verify")
    def verify(trace_id: str):
        m = store.manifest(trace_id)
        failures = []
        for r in m["tensors"]:
            p = store.tensor_path(trace_id, r)
            if digest(p) != r["sha256"]:
                failures.append(r["file"])
        return dict(verified=not failures, failures=failures, checked=len(m["tensors"]))

    @app.get("/api/traces/{trace_id}/export")
    def export(trace_id: str):
        m = store.manifest(trace_id)
        if m["status"] != "completed":
            raise ValueError("エクスポートは完了トレースのみ対応します")
        root = store.path(trace_id)
        fd, filename = tempfile.mkstemp(suffix=".zip", prefix="observatory-export-")
        os.close(fd)
        with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_STORED) as archive:
            for r in m["tensors"]:
                archive.write(store.tensor_path(trace_id, r), r["file"])
            for name in ["manifest.json", "integrity.json"]:
                if (root / name).is_file():
                    archive.write(root / name, name)
        return FileResponse(
            filename,
            filename=f"trace-{trace_id}.zip",
            background=BackgroundTask(Path(filename).unlink, missing_ok=True),
        )

    @app.post("/api/storage/import")
    async def import_trace(request: Request):
        limit = int(store.settings["trace_limit_gib"] * 2**30)
        # Spool upload to disk instead of loading the entire archive into RAM.
        with tempfile.TemporaryFile() as uploaded:
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise ValueError("インポート上限を超えました")
                uploaded.write(chunk)
            uploaded.seek(0)
            try:
                with zipfile.ZipFile(uploaded) as archive:
                    entries = archive.infolist()
                    if sum(i.file_size for i in entries) > limit or len(entries) > 200000:
                        raise ValueError("展開後の容量・ファイル数が上限を超えます")
                    if len(set(i.filename for i in entries)) != len(entries):
                        raise ValueError("重複したアーカイブパス")
                    if archive.getinfo("manifest.json").file_size > 32 * 2**20:
                        raise ValueError("manifestが大きすぎます")
                    m = json.loads(archive.read("manifest.json"))
                    if m.get("schema_version") != SCHEMA_VERSION or m.get("status") != "completed":
                        raise ValueError("未対応形式または未完了トレース")
                    if not all(
                        key in m for key in ["id", "stages", "tensors", "tokens", "model", "level", "title"]
                    ):
                        raise ValueError("必須のmanifest項目がありません")
                    validate_manifest(m)
                    source_id, trace_id = m["id"], uuid.uuid4().hex
                    allowed = {"manifest.json", "integrity.json"} | {r["file"] for r in m["tensors"]}
                    for i in entries:
                        p = Path(i.filename)
                        if (
                            i.filename not in allowed
                            or p.is_absolute()
                            or ".." in p.parts
                            or "\\" in i.filename
                            or ":" in i.filename
                            or i.is_dir()
                        ):
                            raise ValueError("不正なアーカイブパス")
                    stages, tensors = m["stages"], m["tensors"]
                    # create() initializes a journal; restore validated imported definitions afterwards.
                    root = store.create(trace_id, m)
                    m.update(stages=stages, tensors=tensors, source_id=source_id)
                    try:
                        for r in tensors:
                            p = store.tensor_path(trace_id, r)
                            p.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(r["file"]) as src, p.open("wb") as dst:
                                __import__("shutil").copyfileobj(src, dst, 1024 * 1024)
                            if digest(p) != r["sha256"]:
                                raise ValueError("インポートのハッシュが一致しません")
                            validate_finite(store.load(trace_id, r))
                        m.update(status="completed", retained=False)
                        store.save_manifest(m)
                        atomic_json(root / "integrity.json", m.get("integrity", {}))
                        store.retain(trace_id, True)
                    except Exception:
                        store.mark_incomplete(trace_id, "インポート検証に失敗")
                        raise
                    return dict(id=trace_id)
            except zipfile.BadZipFile as exc:
                raise ValueError("有効なZIPファイルを指定してください") from exc
            except (KeyError, IndexError, TypeError, UnicodeDecodeError) as exc:
                raise ValueError("トレースの必須項目またはアーカイブ構造が不正です") from exc

    frontend = Path(__file__).resolve().parents[3] / "frontend/dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


def platform_name():
    return "Windows" if os.name == "nt" else "Linux / WSL"
