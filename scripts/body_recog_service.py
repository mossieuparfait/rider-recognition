#!/usr/bin/env python3
"""body_recog_service.py — consomme un flux MJPEG, extrait pose YOLOv8-pose,
publie un JSON (chemin OUTPUT_JSON) consommable par tout client externe.

Format JSON :
    {
      "ts": float epoch,
      "frame_w": int, "frame_h": int,
      "infer_ms": float,
      "persons": [
        {
          "track_id": int|null,
          "person_bbox": [x1,y1,x2,y2],
          "confidence": float,
          "face_kp": [x, y, conf]|null,    // position visage estimée
          "keypoints": [[x,y,conf], ...]    // 17 keypoints COCO
        }
      ]
    }

Variables d'env :
    STREAM_URL    http://localhost:8810/stream.mjpeg
    OUTPUT_JSON   /tmp/rider-bodies.json
    BODY_PERIOD   0.1                     période détection (s, 10 fps)
    GPU_ID        -1                      -1=CPU, 0=cuda:0
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from rider_recognition.body_recog import BodyRecognizer


class MJPEGDrainer:
    """Thread dédié à drainer le flux MJPEG en continu.

    cv2.VideoCapture sur HTTP MJPEG bufferise en interne (libavformat)
    indépendamment de CAP_PROP_BUFFERSIZE qui ne marche que sur V4L2.
    Si le main loop d'inférence est plus lent que la prod (60 fps), le
    backlog s'accumule → on lit du vieux contenu, retard de plusieurs
    secondes.

    Ce thread lit cap.read() AUSSI VITE que possible (= cadence MJPEG)
    et stocke la dernière frame dans un slot 1-emplacement (drop-oldest).
    Le main loop prend just la dernière dispo à son rythme.

    Recovery : si le stream MJPEG crash ("Stream ends prematurely",
    keep-alive cassé), reconstruit cap après N échecs consécutifs sinon
    le drainer se fige sur une vieille frame et le main loop tourne en
    boucle sur le même contenu (debug nightmare).
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.cap = self._open()
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0   # monotonic, dernière frame réussie
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="mjpeg-drainer")
        self._thread.start()

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
                if n_fail >= 20:  # ~1s d'échec → reset stream
                    log("drainer: stream cassé (20 reads failed), reconnect")
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

    def get_latest(self, max_age_s: float | None = None):
        """Retourne (frame, age_s). Si max_age_s est dépassé, retourne
        (None, age) pour signaler au caller de skipper l'inférence."""
        with self._lock:
            f = self._frame
            ts = self._frame_ts
        if f is None:
            return None, float("inf")
        age = time.monotonic() - ts
        if max_age_s is not None and age > max_age_s:
            return None, age
        return f, age

    def stop(self) -> None:
        self._stop.set()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


STREAM_URL  = env("STREAM_URL", "http://localhost:8810/stream.mjpeg")
OUTPUT_JSON = Path(env("OUTPUT_JSON", "/tmp/rider-bodies.json"))
BODY_PERIOD = float(env("BODY_PERIOD", "0.1"))
GPU_ID      = int(env("GPU_ID", "-1"))


def log(msg: str) -> None:
    print(f"[body_recog] {msg}", flush=True)


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(path)


def main() -> int:
    log(f"démarrage : stream={STREAM_URL} output={OUTPUT_JSON} period={BODY_PERIOD}s")
    log(f"init BodyRecognizer (YOLOv8-pose, gpu_id={GPU_ID})")
    rec = BodyRecognizer(gpu_id=GPU_ID)

    log(f"ouverture drainer thread sur {STREAM_URL}")
    drainer = MJPEGDrainer(STREAM_URL)

    next_run = 0.0
    n_done = 0
    last_stats = time.monotonic()
    n_stale = 0
    while True:
        now = time.monotonic()
        sleep = next_run - now
        if sleep > 0:
            time.sleep(sleep)
        next_run = time.monotonic() + BODY_PERIOD

        # Le drainer thread lit MJPEG aussi vite que possible et stocke
        # la dernière frame ; on prend juste ça (zéro backlog). Si la
        # frame est trop ancienne (> 2s), on skippe l'inférence pour ne
        # pas publier du vieux contenu en boucle.
        frame, age = drainer.get_latest(max_age_s=2.0)
        if frame is None:
            n_stale += 1
            if n_stale % 20 == 0:
                log(f"frame stale (age={age:.1f}s), skip inférence")
            time.sleep(0.05)
            continue
        n_stale = 0

        t0 = time.monotonic()
        persons = rec.detect(frame)
        dt_ms = (time.monotonic() - t0) * 1000

        persons_out = [{
            "track_id": p.track_id,
            "person_bbox": list(p.person_bbox),
            "confidence": p.confidence,
            "face_kp": list(p.face_kp) if p.face_kp else None,
            "keypoints": [list(k) for k in p.keypoints],
        } for p in persons]

        payload = {
            "ts": time.time(),
            "frame_w": frame.shape[1],
            "frame_h": frame.shape[0],
            "infer_ms": round(dt_ms, 1),
            "persons": persons_out,
        }
        write_atomic(OUTPUT_JSON, payload)

        n_done += 1
        if now - last_stats >= 5.0:
            log(f"{n_done / (now - last_stats):.1f} det/s, "
                f"dernière : {len(persons_out)} persons en {dt_ms:.0f}ms")
            n_done = 0
            last_stats = now


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
