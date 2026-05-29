#!/usr/bin/env python3
"""Calcule l'index d'embeddings ArcFace SANS face alignment, pour matcher
le path inference DeepStream (YOLOv8L-Face → bbox → resize 112×112 →
ArcFace) sur la box arbox.

Différence vs build_index.py : on saute le warpAffine sur les 5 keypoints
(SCRFD landmarks) et on passe directement le bbox crop brut, resize
112×112 sans préserver l'aspect ratio — exactement comme nvinfer
secondary classifier de DeepStream le fait.

Pour la détection on garde SCRFD (via insightface buffalo_l) car c'est
fiable sur des photos studio ; ce qui compte c'est le BBOX, pas
l'alignement landmarks.

Sortie : embeddings.bin (raw float32 [N×512] L2-normalisé) +
names.txt (un nom par ligne) — format attendu par face_match.cpp.

Usage:
    python3 scripts/build_index_raw_bbox.py \\
        --photos-dir /home/ben/AIlocal/signatureNG/signature/public/data/rider_photos \\
        --output-dir /tmp/index_raw_bbox
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import sys
import time
from pathlib import Path

# Bootstrap libs CUDA depuis le venv (cf build_index.py original)
_nv_glob = os.path.join(
    os.path.dirname(sys.executable),
    "..", "lib", "python*", "site-packages", "nvidia",
    "*", "lib", "lib*.so*",
)
for _lib in sorted(glob.glob(_nv_glob)):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

sys.path.insert(0, str(Path(__file__).parent.parent))
from rider_recognition.dataset import DEFAULT_PHOTOS_DIR, load_dataset


def build_index_raw(
    photos_dir: Path,
    output_dir: Path,
    det_size: int = 640,
    ctx_id: int = 0,
) -> None:
    print(f"Chargement dataset : {photos_dir}")
    riders = load_dataset(photos_dir)
    n_photos_total = sum(len(r.local_photos) for r in riders.values())
    print(f"  → {len(riders)} coureurs, {n_photos_total} photos")

    print(f"\nInit InsightFace buffalo_l (ctx_id={ctx_id}, det_size={det_size})")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    print(f"  providers actifs : {app.det_model.session.get_providers()}")

    rec_model = app.models["recognition"]
    rec_input = rec_model.input_size  # typiquement (112, 112)
    print(f"  recognition model input : {rec_input}")

    embeddings: list[np.ndarray] = []
    names: list[str] = []

    skipped_no_face = 0
    skipped_read_fail = 0
    skipped_bad_crop = 0

    print(f"\nProcessing {n_photos_total} photos (raw bbox, no alignment)...")
    t0 = time.time()
    n_done = 0

    for uciid, rider in riders.items():
        for photo in rider.local_photos:
            n_done += 1

            img = cv2.imread(str(photo.path))
            if img is None:
                skipped_read_fail += 1
                continue

            faces = app.get(img)
            if not faces:
                skipped_no_face += 1
                continue

            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                skipped_bad_crop += 1
                continue

            crop = img[y1:y2, x1:x2]
            crop_112 = cv2.resize(crop, rec_input)
            feat = rec_model.get_feat(crop_112)
            if feat.ndim == 2:
                feat = feat[0]
            feat = feat.astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm < 1e-6:
                skipped_bad_crop += 1
                continue
            feat = feat / norm

            embeddings.append(feat)
            names.append(rider.name)

            if n_done % 200 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed else 0
                eta = (n_photos_total - n_done) / rate if rate else 0
                print(
                    f"  {n_done}/{n_photos_total} "
                    f"({rate:.1f} photos/s, ETA {eta:.0f}s)"
                )

    elapsed = time.time() - t0
    n_kept = len(embeddings)

    if n_kept == 0:
        print("\nAUCUN embedding produit — abort.")
        sys.exit(1)

    emb_arr = np.stack(embeddings).astype(np.float32)
    print(f"\nFait en {elapsed:.1f}s ({n_done / elapsed:.1f} photos/s) :")
    print(f"  embeddings retenus : {n_kept} / {n_photos_total}")
    print(f"  skipped (lecture)  : {skipped_read_fail}")
    print(f"  skipped (no face)  : {skipped_no_face}")
    print(f"  skipped (bad crop) : {skipped_bad_crop}")
    print(f"  shape emb_arr      : {emb_arr.shape}, dtype {emb_arr.dtype}")

    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / "embeddings.bin"
    names_path = output_dir / "names.txt"

    emb_arr.tofile(emb_path)
    with open(names_path, "w") as f:
        for n in names:
            f.write(n + "\n")

    size_mb = emb_path.stat().st_size / 1024 / 1024
    print(f"\nIndex écrit :")
    print(f"  {emb_path} ({size_mb:.2f} MB)")
    print(f"  {names_path} ({n_kept} lignes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/tmp/index_raw_bbox"))
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--ctx-id", type=int, default=0)
    args = ap.parse_args()

    build_index_raw(args.photos_dir, args.output_dir,
                    args.det_size, args.ctx_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
