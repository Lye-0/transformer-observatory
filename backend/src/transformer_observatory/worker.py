"""Persistent inference process. Importing this module must remain lightweight."""

import multiprocessing
import signal
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .catalog import ModelCatalog
from .schema import TERMINAL
from .startup import startup_progress


class RuntimeUnavailable(RuntimeError):
    pass


def serve(connection, data_root, trace_root, startup_log=None):
    """Only the child imports native inference libraries; keep the command loop responsive."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    send_lock = threading.Lock()

    def send(message):
        with send_lock:
            try:
                connection.send(message)
            except (EOFError, OSError):
                pass  # The parent may have completed shutdown while a task was returning.

    try:
        with startup_progress(startup_log) as progress:
            progress("Loading PyTorch in analysis worker")
            import torch  # noqa: F401 - explicitly time the native import

            progress("Loading analysis libraries in worker")
            from .analysis import compare
            from .jobs import Jobs
            from .lenses import register_lens
            from .models import register_model
            from .research import Research
            from .storage import Store

            store = Store(data_root, trace_root, recover=False)
            jobs = Jobs(store)
            research = Research(store, jobs)
            progress("Analysis engine ready")
        send(("ready", None, None))
    except Exception as exc:
        send(("failed", None, f"{type(exc).__name__}: {exc}"))
        connection.close()
        return

    def respond(request_id, future):
        try:
            result = future.result()
            send(("result", request_id, result))
        except Exception as exc:
            send(("error", request_id, (type(exc).__name__, str(exc))))

    try:
        while True:
            request_id, command, args = connection.recv()
            if command == "shutdown":
                break
            try:
                if command not in {"analysis.cancel", "research.cancel"}:
                    with store.lock:
                        store.roots.clear()
                        store.trace_root = Path(args[-1])
                    args = args[:-1]
                if command == "analysis.accept":
                    trace_id, request = args
                    jobs.cancellations[trace_id] = threading.Event()
                    jobs.executor.submit(jobs.run, trace_id, request)
                    send(("result", request_id, None))
                elif command == "analysis.cancel":
                    send(("result", request_id, jobs.cancel(*args)))
                elif command == "research.cancel":
                    send(("result", request_id, research.cancel(*args)))
                elif command == "research.submit":
                    # submit schedules on the shared execution queue, so it must not run on it.
                    send(("result", request_id, research.submit(*args)))
                else:
                    functions = {
                        "preflight": jobs.preflight,
                        "lens": jobs.lens,
                        "register_model": register_model,
                        "compare": lambda left, right: compare(store, left, right),
                        "register_lens": lambda model_id, path, name: register_lens(
                            store, jobs.model(model_id), path, name
                        ),
                    }
                    future = jobs.executor.submit(functions[command], *args)
                    future.add_done_callback(lambda f, rid=request_id: respond(rid, f))
            except Exception as exc:
                send(("error", request_id, (type(exc).__name__, str(exc))))
    except (EOFError, OSError):
        pass
    finally:
        jobs.close()
        connection.close()


class RuntimeManager:
    def __init__(self, store, *, worker_target=serve):
        self.store = store
        self.worker_target = worker_target
        self.startup_log = None
        self.lock = threading.RLock()
        self.changed = threading.Event()
        self.state, self.error = "idle", None
        self.process = self.connection = None
        self.pending = {}
        self.closed = False
        self.started_at = self.ready_at = None

    def status(self):
        with self.lock:
            return dict(
                state=self.state,
                error=self.error,
                started_at=self.started_at,
                ready_at=self.ready_at,
                pid=self.process.pid if self.process else None,
            )

    def start(self, retry=False):
        with self.lock:
            if self.closed:
                raise RuntimeUnavailable("解析エンジンは終了しました")
            if self.state in {"starting", "ready"} or (self.state == "failed" and not retry):
                return self.status()
            self.state, self.error = "starting", None
            self.started_at, self.ready_at = time.time(), None
            self.changed.clear()
            if self.process and self.process.pid:
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1)
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(
                target=self.worker_target,
                args=(child, str(self.store.data_root), str(self.store.trace_root), self.startup_log),
                name="observatory-runtime",
                daemon=True,
            )
            try:
                self.process.start()
                child.close()
                threading.Thread(
                    target=self._receive,
                    args=(self.connection, self.process),
                    daemon=True,
                    name="runtime-responses",
                ).start()
            except Exception as exc:
                child.close()
                self.connection.close()
                self._fail(str(exc))
            return self.status()

    def _fail(self, error):
        with self.lock:
            self.state, self.error = "failed", error
            self.changed.set()
            for future in self.pending.values():
                future.set_exception(RuntimeUnavailable(error))
            self.pending.clear()
            # A crashed process cannot finish its persisted jobs.
            for table in ("jobs", "studies"):
                for item in self.store.list(table):
                    if item["state"] not in TERMINAL:
                        item.update(state="failed", error=error, updated_at=time.time())
                        if table == "jobs" and item.get("created_at"):
                            item["elapsed_seconds"] = item["updated_at"] - item["created_at"]
                        self.store.put(table, item["id"], item)
                        if table == "jobs":
                            self.store.mark_incomplete(item["id"], error)

    def _receive(self, connection, process):
        try:
            while True:
                if not connection.poll(0.2):
                    if not process.is_alive():
                        raise EOFError("解析ワーカーが終了しました")
                    continue
                kind, request_id, value = connection.recv()
                with self.lock:
                    if self.closed or process is not self.process:
                        return
                    if kind == "ready":
                        self.state, self.ready_at = "ready", time.time()
                        self.changed.set()
                    elif kind == "failed":
                        self._fail(value)
                        return
                    else:
                        future = self.pending.pop(request_id, None)
                        if future is None:
                            continue
                        if kind == "error":
                            name, message = value
                            error = (
                                FileNotFoundError(message)
                                if name == "FileNotFoundError"
                                else ValueError(message)
                            )
                            future.set_exception(error)
                        else:
                            future.set_result(value)
        except (EOFError, OSError) as exc:
            with self.lock:
                if not self.closed and process is self.process:
                    self._fail(str(exc) or "解析ワーカーとの接続が切れました")
        finally:
            connection.close()

    def wait(self):
        self.start()
        self.changed.wait()
        with self.lock:
            if self.closed or self.state != "ready":
                raise RuntimeUnavailable(self.error or "解析エンジンは終了しました")

    def call(self, command, *args):
        self.wait()
        with self.lock:
            if self.closed or self.state != "ready":
                raise RuntimeUnavailable(self.error or "解析エンジンは終了しました")
            request_id = uuid.uuid4().hex
            future = Future()
            self.pending[request_id] = future
            try:
                if command not in {"analysis.cancel", "research.cancel"}:
                    args = (*args, str(self.store.trace_root))
                self.connection.send((request_id, command, args))
            except OSError as exc:
                self._fail(str(exc))
        return future.result()

    def close(self):
        with self.lock:
            self.closed = True
            self.changed.set()
            for future in self.pending.values():
                future.set_exception(RuntimeUnavailable("解析エンジンは終了しました"))
            self.pending.clear()
            process = self.process
            if process and process.is_alive():
                try:
                    self.connection.send((None, "shutdown", ()))
                except OSError:
                    pass
        if process and process.pid:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)


class RemoteJobs(ModelCatalog):
    def __init__(self, store, runtime):
        self.store, self.runtime = store, runtime
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="runtime-wait")

    def preflight(self, request):
        return self.runtime.call("preflight", request)

    def lens(self, *args):
        return self.runtime.call("lens", *args)

    def submit(self, request):
        self.model(request.model_id)  # lightweight registration/path validation
        trace_id = uuid.uuid4().hex
        now = time.time()
        item = dict(
            id=trace_id,
            trace_id=trace_id,
            state="loading_runtime",
            stage=0,
            percent=0,
            detail="解析エンジンを準備中",
            request=request.model_dump(),
            created_at=now,
            updated_at=now,
        )
        with self.lock:
            self.store.put("jobs", trace_id, item)
            self.executor.submit(self._dispatch, trace_id, request)
        return item

    def _dispatch(self, trace_id, request):
        try:
            self.runtime.wait()
            with self.lock:
                if self.store.get("jobs", trace_id)["state"] in TERMINAL:
                    return
                item = self.store.get("jobs", trace_id)
                item.update(state="queued", detail="解析の実行を待機中", updated_at=time.time())
                self.store.put("jobs", trace_id, item)
                self.runtime.call("analysis.accept", trace_id, request)
        except Exception as exc:
            with self.lock:
                item = self.store.get("jobs", trace_id)
                if item["state"] not in TERMINAL:
                    item.update(state="failed", error=str(exc), updated_at=time.time())
                    item["elapsed_seconds"] = item["updated_at"] - item["created_at"]
                    self.store.put("jobs", trace_id, item)

    def cancel(self, trace_id):
        with self.lock:
            item = self.store.get("jobs", trace_id)
            if not item:
                raise ValueError("ジョブがありません")
            if item["state"] in TERMINAL:
                return item
            if item["state"] == "loading_runtime":
                item.update(state="cancelled", detail="準備待ちを中止しました", updated_at=time.time())
                item["elapsed_seconds"] = item["updated_at"] - item["created_at"]
                self.store.put("jobs", trace_id, item)
                return item
            return self.runtime.call("analysis.cancel", trace_id)

    def close(self):
        self.runtime.close()
        self.executor.shutdown(wait=True, cancel_futures=True)


class RemoteResearch:
    def __init__(self, store, runtime):
        self.store, self.runtime = store, runtime

    def get(self, study_id):
        item = self.store.get("studies", study_id)
        if not item:
            raise ValueError("研究ジョブが見つかりません")
        return item

    def submit(self, request):
        return self.runtime.call("research.submit", request)

    def cancel(self, study_id):
        item = self.get(study_id)
        if item["state"] in TERMINAL:
            return item
        return self.runtime.call("research.cancel", study_id)
