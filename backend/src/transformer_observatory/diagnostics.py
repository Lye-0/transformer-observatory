"""Low-overhead analysis timings and stacks only when progress stalls."""

import logging
import sys
import threading
import time
import traceback
from contextlib import contextmanager

from .storage import atomic_json

logger = logging.getLogger(__name__)


class AnalysisDiagnostics:
    def __init__(self, trace_id, path):
        self.trace_id, self.path = trace_id, path
        self.started = self.activity = time.monotonic()
        self.last_warning = self.started - 30
        self.operation = "analysis"
        self.thread_id = threading.get_ident()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.timings, self.stalls = {}, []
        self.worker = threading.Thread(target=self._watch, name="analysis-diagnostics", daemon=True)

    def start(self):
        self.worker.start()

    @contextmanager
    def measure(self, operation):
        started = time.monotonic()
        with self.lock:
            previous = self.operation
            self.operation, self.activity = operation, started
        try:
            yield
        finally:
            finished = time.monotonic()
            duration = finished - started
            with self.lock:
                stats = self.timings.setdefault(operation, dict(count=0, total_seconds=0, max_seconds=0))
                stats["count"] += 1
                stats["total_seconds"] += duration
                stats["max_seconds"] = max(stats["max_seconds"], duration)
                self.operation, self.activity = previous, finished

    def _watch(self):
        while not self.stopped.wait(5):
            self._check_stall(time.monotonic())

    def _check_stall(self, now):
        with self.lock:
            inactive = now - self.activity
            if inactive < 10 or now - self.last_warning < 30:
                return
            self.last_warning = now
            frame = sys._current_frames().get(self.thread_id)
            stack = "".join(traceback.format_stack(frame, limit=16)) if frame else ""
            record = dict(operation=self.operation, inactive_seconds=inactive, stack=stack)
            self.stalls.append(record)
            self.stalls = self.stalls[-20:]
        logger.warning(
            "Analysis %s: no checkpoint for %.1fs (%s)\n%s",
            self.trace_id,
            inactive,
            record["operation"],
            stack,
        )

    def close(self):
        self.stopped.set()
        if self.worker.ident is not None:
            self.worker.join(timeout=1)
        try:
            atomic_json(
                self.path,
                dict(
                    trace_id=self.trace_id,
                    elapsed_seconds=time.monotonic() - self.started,
                    timings=self.timings,
                    stalls=self.stalls,
                    note="Nested timings overlap; forward includes capture and storage.",
                ),
            )
        except (OSError, ValueError, TypeError):
            logger.warning("Analysis %s: could not save timing diagnostics", self.trace_id, exc_info=True)
