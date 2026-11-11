import argparse
import socket
import threading
import webbrowser
from pathlib import Path

from .startup import startup_progress


def create_app():
    from .api import create_app as factory

    return factory()


def main():
    parser = argparse.ArgumentParser(description="Transformer Observatory local server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--startup-log", help="起動時の段階・待機中のスタック・例外を保存するファイル")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("portは0〜65535で指定してください")
    try:
        with startup_progress(args.startup_log) as progress:
            progress("Loading server libraries")
            import uvicorn

            progress("Preparing lightweight application")
            app = create_app()
            if args.startup_log and getattr(app.state, "runtime", None):
                log_path = Path(args.startup_log)
                app.state.runtime.startup_log = str(log_path.with_stem(log_path.stem + "-runtime"))
            progress("Application prepared; starting HTTP server (analysis engine loads after UI opens)")
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    # Bind once and pass the same socket to uvicorn, avoiding a free-port race.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", args.port))
    except OSError:
        sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    print(f"Transformer Observatory: http://127.0.0.1:{port}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(f"http://127.0.0.1:{port}",)).start()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info"))
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        # Uvicorn re-raises the interrupt after completing graceful shutdown.
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
