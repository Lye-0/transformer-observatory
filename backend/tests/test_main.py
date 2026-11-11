import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
@pytest.mark.parametrize("arguments, returncode", [(["--help"], 0), (["--port", "65536"], 2)])
def test_cli_output_does_not_load_analysis_libraries(encoding, arguments, returncode):
    script = """
import runpy
import sys

def reject_heavy_import(name, args):
    if name == "import" and args[0] in {"uvicorn", "transformer_observatory.api", "scipy", "torch"}:
        raise AssertionError("help must not load analysis libraries")

sys.addaudithook(reject_heavy_import)
sys.argv = ["transformer_observatory", *sys.argv[1:]]
runpy.run_module("transformer_observatory", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        env={**os.environ, "PYTHONIOENCODING": f"{encoding}:strict"},
        capture_output=True,
        encoding=encoding,
        timeout=15,
    )
    assert result.returncode == returncode, result.stderr
    if returncode == 0:
        assert "--startup-log" in result.stdout
        message = "起動時の段階・待機中のスタック・例外を保存するファイル"
        output = result.stdout
    else:
        message = "portは0〜65535で指定してください"
        output = result.stderr
    assert message.encode(encoding, errors="backslashreplace").decode(encoding) in output
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("failure", [KeyboardInterrupt, ValueError])
def test_startup_diagnostics_preserve_failure(tmp_path, monkeypatch, capsys, failure):
    from transformer_observatory import __main__ as entrypoint

    path = tmp_path / "diagnostic.log"

    def fail():
        raise failure("injected startup failure")

    monkeypatch.setattr(entrypoint, "create_app", fail)
    monkeypatch.setattr(sys, "argv", ["observatory", "--startup-log", str(path), "--no-browser"])
    with pytest.raises(failure):
        entrypoint.main()
    text = path.read_text(encoding="utf-8")
    assert "Preparing lightweight application" in text
    assert f"{failure.__name__}: injected startup failure" in text
    assert "Traceback" in text
    assert "pid=" in text
    assert "Loading server libraries" in capsys.readouterr().out


def test_startup_diagnostics_close_after_success(tmp_path):
    from transformer_observatory.startup import startup_progress

    path = tmp_path / "diagnostic.log"
    with startup_progress(path) as progress:
        progress("Application prepared")
    assert "Application prepared" in path.read_text(encoding="utf-8")
    # No background worker or faulthandler timer may keep the file open on Windows.
    path.unlink()


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
