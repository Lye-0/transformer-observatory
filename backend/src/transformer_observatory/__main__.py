import argparse
import socket
import threading
import webbrowser

import uvicorn

from .api import create_app


def main():
    parser = argparse.ArgumentParser(description="Transformer Observatory local server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("portは0〜65535で指定してください")
    try:
        app = create_app()
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
