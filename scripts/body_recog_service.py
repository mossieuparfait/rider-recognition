#!/usr/bin/env python3
"""body_recog_service.py — consomme un flux MJPEG, extrait pose YOLOv8-pose,
publie /tmp/avtowan-bodies.json consommé par face_recog_service AVtoWan.

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
    OUTPUT_JSON   /tmp/avtowan-bodies.json
    BODY_PERIOD   0.1                     période détection (s, 10 fps)
    GPU_ID        -1                      -1=CPU, 0=cuda:0
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from rider_recognition.body_recog import BodyRecognizer


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


STREAM_URL  = env("STREAM_URL", "http://localhost:8810/stream.mjpeg")
OUTPUT_JSON = Path(env("OUTPUT_JSON", "/tmp/avtowan-bodies.json"))
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

    log("ouverture du flux MJPEG")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        log(f"FATAL: impossible d'ouvrir {STREAM_URL}")
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    next_run = 0.0
    n_done = 0
    last_stats = time.monotonic()
    while True:
        now = time.monotonic()
        sleep = next_run - now
        if sleep > 0:
            time.sleep(sleep)
        next_run = time.monotonic() + BODY_PERIOD

        ok, frame = cap.read()
        if not ok or frame is None:
            log("frame illisible, retry dans 1s")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

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
