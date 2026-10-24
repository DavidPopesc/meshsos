"""Tiny local server for the 3D viewer: serves viewer/ and one run directory."""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path
from urllib.parse import unquote, urlparse

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


class Handler(http.server.SimpleHTTPRequestHandler):
    run_dir: Path = Path(".")
    capture_dir: Path | None = None

    def translate_path(self, path: str) -> str:
        p = unquote(urlparse(path).path)
        if p.startswith("/run/"):
            return str(self.run_dir / p[len("/run/"):])
        if p == "/" or p.startswith("/viewer"):
            rest = p[len("/viewer"):].lstrip("/") if p.startswith("/viewer") else ""
            return str(VIEWER_DIR / (rest or "index.html"))
        return str(VIEWER_DIR / "__missing__")

    def do_POST(self):
        # Frame capture for README screenshots / GIF: POST /capture/<name>.png
        p = unquote(urlparse(self.path).path)
        if self.capture_dir is None or not p.startswith("/capture/"):
            self.send_error(404)
            return
        name = Path(p).name
        if not name.endswith(".png") or "/" in name or name.startswith("."):
            self.send_error(400)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        (self.capture_dir / name).write_bytes(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"saved": name}).encode())

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def serve(run: str = "runs/latest", port: int = 8000, open_browser: bool = True, query: str = "",
          capture_dir: str | None = None) -> None:
    run_dir = Path(run).resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    if not (run_dir / "timeline.json").exists():
        raise SystemExit(f"no timeline.json in {run_dir} - run `python -m meshsos run` first")
    handler = type("MeshHandler", (Handler,), {"run_dir": run_dir,
                                               "capture_dir": Path(capture_dir).resolve() if capture_dir else None})
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/viewer/index.html?run=/run/timeline.json{('&' + query) if query else ''}"
        print(f"MeshSOS viewer for {run_dir.name}: {url}\n(ctrl+c to stop; add &demo=1 for the judges' fly-through)")
        if open_browser:
            threading.Timer(0.4, partial(webbrowser.open, url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
