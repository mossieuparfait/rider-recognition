#!/usr/bin/env python3
"""face_recog_server.py — DeepStream pipeline + serveur HTTP MJPEG.

Pilote la pipeline GStreamer via gst-python. Le hot path (NVDEC, nvinfer,
nvtracker, cublas matcher, nvdsosd) reste 100% C/CUDA inside GStreamer.
Python ne touche que les buffers JPEG déjà encodés sortant de l'appsink
pour les pousser aux clients HTTP — pas dans le hot path.

Pipeline :
    udpsrc port=5000 → tsdemux → h265parse → nvv4l2decoder →
    nvvideoconvert (P010→NV12) → nvstreammux →
    nvinfer (YOLOv8L-Face) → nvtracker (NvSORT) →
    nvinfer (ArcFace + cublas matcher) → nvdsosd →
    nvvideoconvert → jpegenc → appsink

HTTP routes :
    GET /                  → page HTML simple avec embed du MJPEG
    GET /stream.mjpeg      → MJPEG multipart, 1 frame par JPEG
    GET /healthz           → "ok"
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402


# ──────────────────────── État global ────────────────────────────────
class FrameBuffer:
    """Holds the latest JPEG frame for all subscribers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.subscribers = 0

    def push(self, data: bytes) -> None:
        with self.cond:
            self.jpeg = data
            self.frame_id += 1
            self.cond.notify_all()

    def wait_new(self, last_id: int, timeout: float = 2.0):
        with self.cond:
            if not self.cond.wait_for(
                lambda: self.frame_id > last_id, timeout=timeout
            ):
                return None
            return (self.jpeg, self.frame_id)


fbuf = FrameBuffer()


# ──────────────────────── GStreamer ──────────────────────────────────
def on_new_sample(appsink) -> int:
    """Callback appelé par GStreamer à chaque nouvelle frame JPEG."""
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.ERROR
    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.ERROR
    try:
        fbuf.push(bytes(mapinfo.data))
    finally:
        buf.unmap(mapinfo)
    return Gst.FlowReturn.OK


PIPELINE_DESC = """
udpsrc port={port} buffer-size=8388608 !
tsdemux !
h265parse !
nvv4l2decoder !
video/x-raw(memory:NVMM),format=P010_10LE !
nvvideoconvert !
video/x-raw(memory:NVMM),format=NV12 !
mux.sink_0 nvstreammux name=mux batch-size=1 width=1920 height=1080
           batched-push-timeout=33000 live-source=1 !
nvinfer config-file-path={work}/configs/yolov8l_face.txt unique-id=1 !
nvtracker tracker-width=640 tracker-height=384
          ll-lib-file={ds}/lib/libnvds_nvmultiobjecttracker.so
          ll-config-file={ds}/samples/configs/deepstream-app/config_tracker_NvSORT.yml
          gpu-id=0 !
nvinfer config-file-path={work}/configs/arcface_secondary.txt unique-id=2 !
nvvideoconvert !
nvdsosd !
nvvideoconvert !
video/x-raw,format=I420 !
jpegenc quality={jpeg_quality} !
image/jpeg !
appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false
"""


def build_pipeline(udp_port: int, work_dir: str, ds_root: str,
                   jpeg_q: int) -> Gst.Pipeline:
    desc = PIPELINE_DESC.format(
        port=udp_port, work=work_dir, ds=ds_root, jpeg_quality=jpeg_q,
    ).strip()
    print(f"[srv] launching pipeline (udp :{udp_port})", flush=True)
    pipeline = Gst.parse_launch(desc)
    sink = pipeline.get_by_name("sink")
    sink.connect("new-sample", on_new_sample)
    return pipeline


def bus_loop(pipeline: Gst.Pipeline, mainloop: GLib.MainLoop) -> None:
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def restart_pipeline():
        # NULL → PLAYING flush l'état du h265parse et reconnecte
        # proprement udpsrc — utile quand la source live coupe.
        print("[srv] restarting pipeline", flush=True)
        pipeline.set_state(Gst.State.NULL)
        pipeline.set_state(Gst.State.PLAYING)
        return False  # one-shot

    def on_msg(_bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[srv] gst ERROR: {err.message} | {dbg}", flush=True)
            GLib.timeout_add_seconds(2, restart_pipeline)
        elif msg.type == Gst.MessageType.EOS:
            print("[srv] gst EOS — restart in 2s", flush=True)
            GLib.timeout_add_seconds(2, restart_pipeline)
        elif msg.type == Gst.MessageType.WARNING:
            warn, dbg = msg.parse_warning()
            print(f"[srv] gst WARN: {warn.message}", flush=True)
        return True

    bus.connect("message", on_msg)


# ──────────────────────── HTTP server ─────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>arbox face-recog</title>
<style>
  body { margin: 0; background: #000; color: #ccc;
         font-family: -apple-system, sans-serif; }
  header { padding: 8px 16px; background: #161b22;
           border-bottom: 1px solid #30363d;
           display: flex; align-items: center; gap: 12px;
           font-size: 12px; }
  header h1 { margin: 0; font-size: 13px; color: #a371f7; }
  header .meta { color: #8b949e; }
  header .fps { color: #3fb950; margin-left: auto; }
  main { display: flex; justify-content: center;
         background: #0d1117; }
  img { max-width: 100%; max-height: calc(100vh - 32px);
        display: block; }
</style>
</head>
<body>
<header>
  <h1>arbox face-recog</h1>
  <span class="meta">DeepStream 7.1 — YOLOv8L-Face + ArcFace + cublas</span>
  <span class="fps" id="fps">— fps</span>
</header>
<main>
  <img id="live" alt="live face-recog">
</main>
<script>
  // Polling JPEG direct via <img src> — pattern le plus simple,
  // marche sur tous les browsers, c'est le browser qui gère
  // download + decode JPEG natif (zero traitement JS).
  const img = document.getElementById('live');
  const fpsEl = document.getElementById('fps');
  let count = 0, lastSec = performance.now();

  img.onload = () => {
    count++;
    const now = performance.now();
    if (now - lastSec >= 1000) {
      fpsEl.textContent = count + ' fps';
      count = 0; lastSec = now;
    }
    next();
  };
  img.onerror = () => { setTimeout(next, 200); };

  function next() { img.src = '/snapshot.jpg?t=' + Date.now(); }
  next();
</script>
</body>
</html>
""".encode("utf-8")


BOUNDARY = b"rider-mjpeg"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # silence

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(INDEX_HTML)
        elif path == "/healthz":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/snapshot.jpg":
            self._serve_snapshot()
        elif path == "/stream.mjpeg":
            self._serve_mjpeg()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()

    def _serve_snapshot(self):
        # Renvoie le dernier JPEG en réponse one-shot — utile pour debug
        # browser et pour les clients qui ne consomment pas MJPEG.
        res = fbuf.wait_new(0, timeout=3.0)
        if res is None:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        jpeg, _ = res
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
        )
        self.end_headers()
        self.connection.settimeout(3.0)
        with fbuf.cond:
            fbuf.subscribers += 1
        last_id = 0
        try:
            while True:
                res = fbuf.wait_new(last_id, timeout=3.0)
                if res is None:
                    continue
                jpeg, last_id = res
                chunk = b"--" + BOUNDARY + b"\r\n"
                chunk += b"Content-Type: image/jpeg\r\n"
                chunk += f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                chunk += jpeg
                chunk += b"\r\n"
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        finally:
            with fbuf.cond:
                fbuf.subscribers = max(0, fbuf.subscribers - 1)


class ThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ──────────────────────── Main ────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp-port", type=int,
                    default=int(os.environ.get("UDP_PORT", "5000")))
    ap.add_argument("--http-port", type=int,
                    default=int(os.environ.get("HTTP_PORT", "8810")))
    ap.add_argument("--work", default=os.environ.get("WORK", "/work"))
    ap.add_argument("--ds-root",
                    default=os.environ.get(
                        "DS_ROOT", "/opt/nvidia/deepstream/deepstream"))
    ap.add_argument("--jpeg-quality", type=int,
                    default=int(os.environ.get("JPEG_QUALITY", "80")))
    args = ap.parse_args()

    Gst.init(None)

    pipeline = build_pipeline(args.udp_port, args.work,
                               args.ds_root, args.jpeg_quality)
    mainloop = GLib.MainLoop()
    bus_loop(pipeline, mainloop)

    server = ThreadingServer(("0.0.0.0", args.http_port), Handler)
    print(f"[srv] HTTP up on :{args.http_port}", flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    pipeline.set_state(Gst.State.PLAYING)
    try:
        mainloop.run()
    except KeyboardInterrupt:
        print("[srv] SIGINT", flush=True)
    finally:
        pipeline.set_state(Gst.State.NULL)
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
