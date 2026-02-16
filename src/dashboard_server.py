import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
import threading
import time
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


class _DashboardState:
    def __init__(self, artifact_root: Path, stream_fps: float) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.stream_fps = max(1.0, float(stream_fps))
        self.lock = threading.Lock()
        self.latest_jpeg_bytes = None  # type: Optional[bytes]
        self.latest_stats = {}  # type: Dict[str, object]
        self.stop_event = threading.Event()

    def set_frame(self, jpeg_bytes: bytes) -> None:
        with self.lock:
            self.latest_jpeg_bytes = jpeg_bytes

    def set_stats(self, stats: Dict[str, object]) -> None:
        with self.lock:
            self.latest_stats = dict(stats)

    def get_snapshot(self) -> Tuple[Optional[bytes], Dict[str, object]]:
        with self.lock:
            frame = self.latest_jpeg_bytes
            stats = dict(self.latest_stats)
        return frame, stats


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _is_subpath(target: Path, base: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _build_handler(state: _DashboardState):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "OccupancyDashboard/1.0"

        def log_message(self, format, *args):  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                self._serve_index()
                return
            if path == "/stats.json":
                self._serve_stats()
                return
            if path == "/stream.mjpeg":
                self._serve_stream()
                return
            if path == "/download":
                self._serve_download(parsed.query)
                return
            self.send_error(404, "Not found")

        def _serve_index(self) -> None:
            html = self._index_html()
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_stats(self) -> None:
            _, stats = state.get_snapshot()
            payload = json.dumps(stats, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _serve_stream(self) -> None:
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            sleep_sec = 1.0 / max(1.0, state.stream_fps)
            try:
                while not state.stop_event.is_set():
                    frame, _ = state.get_snapshot()
                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(("Content-Length: %d\r\n\r\n" % len(frame)).encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(sleep_sec)
            except Exception:
                return

        def _serve_download(self, query: str) -> None:
            params = parse_qs(query)
            requested = unquote(params.get("path", [""])[0]).strip()
            if not requested:
                self.send_error(400, "Missing path query")
                return

            if requested.startswith("/") or requested.startswith("\\"):
                self.send_error(403, "Absolute paths are not allowed")
                return

            if ".." in requested.replace("\\", "/").split("/"):
                self.send_error(403, "Invalid path")
                return

            target = (state.artifact_root / requested).resolve()
            if not _is_subpath(target, state.artifact_root):
                self.send_error(403, "Path escapes session")
                return

            if not target.exists() or not target.is_file():
                self.send_error(404, "File not found")
                return

            ctype, _ = mimetypes.guess_type(str(target))
            if not ctype:
                ctype = "application/octet-stream"

            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % target.name)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _index_html(self) -> str:
            links = [
                ("Annotated MP4", "outputs/annotated.mp4"),
                ("Annotated AVI", "outputs/annotated.avi"),
                ("Events CSV", "logs/events.csv"),
                ("Summary JSON", "logs/summary.json"),
                ("Occupancy Plot", "reports/occupancy_plot.png"),
                ("Zone Heatmap", "reports/heatmap.png"),
                ("Run Info", "run_info.json"),
            ]
            link_html = "\n".join(
                [
                    '<li><a href="/download?path=%s">%s</a></li>' % (path, label)
                    for (label, path) in links
                ]
            )
            return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Occupancy Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background: #101418; color: #eef3f8; }
    .row { display: flex; flex-wrap: wrap; gap: 16px; }
    .card { background: #1a232d; border: 1px solid #2f3d4a; border-radius: 8px; padding: 12px; }
    #stream { max-width: 960px; width: 100%%; border-radius: 8px; border: 1px solid #2f3d4a; }
    pre { margin: 0; white-space: pre-wrap; font-size: 13px; }
    a { color: #69b7ff; }
  </style>
</head>
<body>
  <h2>Smart Occupancy Dashboard</h2>
  <div class="row">
    <div class="card" style="flex: 2; min-width: 360px;">
      <img id="stream" src="/stream.mjpeg" alt="Live stream" />
    </div>
    <div class="card" style="flex: 1; min-width: 280px;">
      <h3>Live Stats</h3>
      <pre id="stats">Loading...</pre>
      <h3>Downloads</h3>
      <ul>
        %s
      </ul>
    </div>
  </div>
  <script>
    async function refreshStats() {
      try {
        const resp = await fetch('/stats.json', {cache: 'no-store'});
        const data = await resp.json();
        document.getElementById('stats').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('stats').textContent = 'Stats unavailable';
      }
    }
    setInterval(refreshStats, 1000);
    refreshStats();
  </script>
</body>
</html>
            """ % link_html

    return DashboardHandler


class DashboardServer:
    def __init__(self, host: str, port: int, stream_fps: float, artifact_root: Path) -> None:
        self.host = str(host)
        self.port = int(port)
        self._state = _DashboardState(artifact_root=Path(artifact_root), stream_fps=float(stream_fps))
        self._httpd = None  # type: Optional[_ThreadedHTTPServer]
        self._thread = None  # type: Optional[threading.Thread]

    def start(self) -> bool:
        handler = _build_handler(self._state)
        try:
            self._httpd = _ThreadedHTTPServer((self.host, self.port), handler)
        except Exception as exc:
            print("Dashboard failed to start at %s:%d (%s)" % (self.host, self.port, exc))
            self._httpd = None
            return False

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print("Dashboard listening on http://%s:%d" % (self.host, self.port))
        return True

    def stop(self) -> None:
        self._state.stop_event.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def update_frame(self, jpeg_bytes: bytes) -> None:
        self._state.set_frame(jpeg_bytes)

    def update_stats(self, stats: Dict[str, object]) -> None:
        self._state.set_stats(stats)
