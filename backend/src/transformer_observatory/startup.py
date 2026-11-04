"""Opt-in startup diagnostics; never change interrupt handling or impose a timeout."""

import faulthandler
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def startup_progress(log_path=None):
    started = time.monotonic()
    stage = "initializing"
    stopped = threading.Event()
    log = None
    worker = None
    lock = threading.Lock()

    def report(message):
        nonlocal stage
        stage = message
        line = f"Startup [{time.monotonic() - started:.1f}s]: {message}"
        with lock:
            print(line, flush=True)
            if log:
                print(line, file=log, flush=True)

    def heartbeat():
        while not stopped.wait(10):
            report(f"Still loading: {stage.removeprefix('Still loading: ')}")

    try:
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("w", encoding="utf-8")
            print(f"pid={os.getpid()} python={sys.version} executable={sys.executable}", file=log, flush=True)
            report(f"Diagnostic log: {path}")
            # Dumps stacks without cancelling the startup or raising a signal.
            faulthandler.dump_traceback_later(30, repeat=True, file=log)
            worker = threading.Thread(target=heartbeat, daemon=True)
            worker.start()
        yield report
    except BaseException:
        if log:
            with lock:
                print(f"Startup interrupted or failed during: {stage}", file=log)
                traceback.print_exc(file=log)
                log.flush()
        raise
    finally:
        stopped.set()
        if worker:
            worker.join()
        if log:
            faulthandler.cancel_dump_traceback_later()
            log.close()
