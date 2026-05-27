#!/usr/bin/env python3
"""bib_recog_service.py — consomme un flux MJPEG, détecte les dossards,
publie un JSON consommé par face_recog_service.py AVtoWan.

Pourquoi un service séparé : torch (YOLO) et onnxruntime (InsightFace
côté face-recog) ont des versions cuDNN qui conflictent dans le même
process. On les sépare en 2 venvs / 2 processus, IPC via fichier JSON.

Variables d'env :
    STREAM_URL    http://localhost:8810/stream.mjpeg   MJPEG à consommer
    OUTPUT_JSON   /tmp/avtowan-bibs.json               état publié
    BIB_PERIOD    0.5                                  période détection (s)
    RACE_JSON     /home/ben/rider-recognition/data/race_tdf2024_partants.json
                                                       mapping bib→coureur
    GPU_ID        -1                                   -1 = CPU (défaut)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from rider_recognition.bib_mapping import load_bib_mapping
from rider_recognition.bib_recog import BibRecognizer


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


STREAM_URL  = env("STREAM_URL", "http://localhost:8810/stream.mjpeg")
OUTPUT_JSON = Path(env("OUTPUT_JSON", "/tmp/avtowan-bibs.json"))
BIB_PERIOD  = float(env("BIB_PERIOD", "0.5"))
RACE_JSON   = Path(env(
    "RACE_JSON",
    "/home/ben/rider-recognition/data/race_tdf2024_partants.json",
))
GPU_ID      = int(env("GPU_ID", "-1"))


def log(msg: str) -> None:
    print(f"[bib_recog] {msg}", flush=True)


def write_atomic(path: Path, payload: dict) -> None:
    """Écriture atomique via rename pour que le lecteur ne voie jamais
    un fichier à moitié écrit."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(path)


def main() -> int:
    log(f"démarrage : stream={STREAM_URL} output={OUTPUT_JSON} period={BIB_PERIOD}s")

    # Mapping bib → coureur (1 lookup au boot, course statique sur la
    # journée de prod). À recharger à chaud si on bascule de course.
    log(f"chargement mapping : {RACE_JSON}")
    mapping = load_bib_mapping(RACE_JSON)
    log(f"  → {len(mapping)} coureurs indexés par bib")

    log("init BibRecognizer (YOLO + PaddleOCR, ~10s)")
    rec = BibRecognizer(gpu_id=GPU_ID)

    log("ouverture du flux MJPEG")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        log(f"FATAL: impossible d'ouvrir {STREAM_URL}")
        return 1
    # Pas de buffering — on veut la frame la plus récente, pas la queue.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    next_run = 0.0
    n_done = 0
    last_stats = time.monotonic()
    while True:
        # Skip frames jusqu'au prochain BIB_PERIOD pour éviter de lire le
        # backlog du flux.
        now = time.monotonic()
        sleep = next_run - now
        if sleep > 0:
            time.sleep(sleep)
        next_run = time.monotonic() + BIB_PERIOD

        # cv2.VideoCapture sur MJPEG bufferise un peu — on grab + retrieve
        # une seule fois (BUFFERSIZE=1 limite l'accumulation).
        ok, frame = cap.read()
        if not ok or frame is None:
            log("frame illisible, retry dans 1s")
            time.sleep(1)
            # Recovery : ré-ouvrir le flux si la connexion est cassée.
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        t0 = time.monotonic()
        dets, persons = rec.detect(frame)
        dt_ms = (time.monotonic() - t0) * 1000

        # Compose le JSON consommé par face_recog_service.
        bibs_out = []
        for d in dets:
            rider = mapping.get(d.bib)
            bibs_out.append({
                "bib": d.bib,
                "name": rider.display_name if rider else None,
                "team": rider.team_name if rider else None,
                "uciid": rider.uciid if rider else None,
                "person_bbox": list(d.person_bbox),
                "bib_bbox": list(d.bib_bbox),
                "confidence": d.confidence,
                "track_id": d.track_id,
            })

        # Personnes trackées (BoT-SORT) avec ou sans bib lu. Sert au
        # tracking body côté face_recog.
        persons_out = [{
            "track_id": p.track_id,
            "person_bbox": list(p.person_bbox),
            "confidence": p.confidence,
        } for p in persons]

        payload = {
            "ts": time.time(),
            "frame_w": frame.shape[1],
            "frame_h": frame.shape[0],
            "infer_ms": round(dt_ms, 1),
            "bibs": bibs_out,
            "persons": persons_out,
        }
        write_atomic(OUTPUT_JSON, payload)

        n_done += 1
        if now - last_stats >= 5.0:
            log(f"{n_done / (now - last_stats):.1f} det/s, "
                f"dernière : {len(persons_out)} persons, {len(bibs_out)} bibs "
                f"en {dt_ms:.0f}ms ({[b['bib'] for b in bibs_out]})")
            n_done = 0
            last_stats = now


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
