import subprocess
import sys

import pytest


@pytest.mark.parametrize("interrupt", [True, False])
def test_server_exit(interrupt):
    script = """
import signal
import sys

import uvicorn
from fastapi import FastAPI
from transformer_observatory import __main__ as entrypoint

interrupt = sys.argv[1] == "interrupt"

async def main_loop(self):
    if interrupt:
        signal.raise_signal(signal.SIGINT)
    else:
        raise RuntimeError("server failure remains visible")

uvicorn.Server.main_loop = main_loop
entrypoint.create_app = FastAPI
sys.argv = ["transformer_observatory", "--port", "0", "--no-browser"]
entrypoint.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "interrupt" if interrupt else "failure"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if interrupt:
        assert result.returncode == 0, result.stderr
        assert "Application shutdown complete." in result.stderr
        assert "Finished server process" in result.stderr
        assert "Traceback" not in result.stderr
        assert "KeyboardInterrupt" not in result.stderr
        assert "CancelledError" not in result.stderr
    else:
        assert result.returncode != 0
        assert "RuntimeError: server failure remains visible" in result.stderr
