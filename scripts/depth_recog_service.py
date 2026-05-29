#!/usr/bin/env python3
"""depth_recog_service.py — DepthAnything-V2-Small en continu sur un MJPEG,
publie une depth map fraîche pour usage downstream (bullet-time, etc).

Architecture :
- Drainer thread lit le MJPEG en continu (drop-oldest, comme body_recog).
- Main loop d'inférence à DEPTH_PERIOD (0.5s = 2 fps par défaut).
- Écrit la depth en raw uint8 (.npy) + métadonnées JSON (path, ts, shape).

Un client externe peut consommer cette depth à la demande pour effets
type freeze+parallax warp.

Env :
    STREAM_URL       http://localhost:8810/stream.mjpeg
    OUTPUT_NPY       /tmp/rider-depth.npy
    OUTPUT_JSON      /tmp/rider-depth.json
    DEPTH_PERIOD     0.5    (2 fps suffit, depth utilisée que sur freeze)
    GPU_ID           0
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from rider_recognition.depth_recog import DepthRecognizer


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


STREAM_URL   = env("STREAM_URL", "http://localhost:8810/stream.mjpeg")
OUTPUT_NPY   = Path(env("OUTPUT_NPY", "/tmp/rider-depth.npy"))
OUTPUT_JSON  = Path(env("OUTPUT_JSON", "/tmp/rider-depth.json"))
DEPTH_PERIOD = float(env("DEPTH_PERIOD", "0.5"))
GPU_ID       = int(env("GPU_ID", "-1"))


def log(msg: str) -> None:
    print(f"[depth_recog] {msg}", flush=True)


class MJPEGDrainer:
    """Même pattern que body_recog_service : thread dédié drainage drop-
    oldest + reconnect auto si stream cassé."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.cap = self._open()
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True,
                         name="mjpeg-drainer-depth").start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        n_fail = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                n_fail += 1
                time.sleep(0.05)
                if n_fail >= 20:
                    log("drainer: stream cassé, reconnect")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = self._open()
                    n_fail = 0
                continue
            n_fail = 0
            with self._lock:
                self._frame = frame
                self._frame_ts = time.monotonic()

    def get_latest(self, max_age_s: float = 2.0):
        with self._lock:
            f = self._frame
            ts = self._frame_ts
        if f is None:
            return None
        if time.monotonic() - ts > max_age_s:
            return None
        return f


def write_atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def main() -> int:
    log(f"démarrage : stream={STREAM_URL} npy={OUTPUT_NPY} period={DEPTH_PERIOD}s gpu={GPU_ID}")
    log("init DepthRecognizer (DepthAnything-V2-Small, ~10s)")
    rec = DepthRecognizer(gpu_id=GPU_ID)

    log("ouverture drainer thread")
    drainer = MJPEGDrainer(STREAM_URL)

    next_run = 0.0
    n_done = 0
    last_stats = time.monotonic()
    while True:
        now = time.monotonic()
        sleep = next_run - now
        if sleep > 0:
            time.sleep(sleep)
        next_run = time.monotonic() + DEPTH_PERIOD

        frame = drainer.get_latest()
        if frame is None:
            time.sleep(0.1)
            continue

        t0 = time.monotonic()
        depth = rec.detect(frame)
        dt_ms = (time.monotonic() - t0) * 1000

        # Écrit raw bytes (uint8 contigu) + métadonnées.
        write_atomic_bytes(OUTPUT_NPY, depth.tobytes())
        write_atomic_json(OUTPUT_JSON, {
            "ts": time.time(),
            "shape": list(depth.shape),     # [h, w]
            "dtype": "uint8",
            "frame_w": frame.shape[1],
            "frame_h": frame.shape[0],
            "infer_ms": round(dt_ms, 1),
        })

        n_done += 1
        if now - last_stats >= 5.0:
            log(f"{n_done / (now - last_stats):.1f} det/s, "
                f"dernier : depth {depth.shape} en {dt_ms:.0f}ms")
            n_done = 0
            last_stats = now


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
