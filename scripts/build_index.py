#!/usr/bin/env python3
"""Calcule l'index d'embeddings ArcFace pour tous les coureurs du dataset.

Pour chaque photo : détection du visage (SCRFD), choix du plus gros visage,
extraction de l'embedding ArcFace 512D L2-normalisé.

Sortie : un .npz avec embeddings (N, 512) + uciids (N,) + photo_tags (N,) +
names (N,). N est le nombre de photos avec visage détecté (≤ nb photos
totales du dataset).

Usage:
    python3 scripts/build_index.py --photos-dir /home/ben/rider_photos \\
        --output data/embeddings.npz
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import sys
import time
from pathlib import Path

# ── Bootstrap libs CUDA depuis le venv ──
# Sans dlopen RTLD_GLOBAL, onnxruntime CUDAExecutionProvider échoue à charger
# libcublasLt.so.12 (livré dans site-packages/nvidia/cublas/lib/) et fallback
# silencieusement sur CPU (~10× plus lent).
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


def build_index(
    photos_dir: Path,
    output_path: Path,
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

    embeddings: list[np.ndarray] = []
    uciids: list[str] = []
    photo_tags: list[str] = []
    rider_names: list[str] = []

    skipped_no_face = 0
    skipped_read_fail = 0

    print(f"\nProcessing {n_photos_total} photos...")
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

            # Si plusieurs visages : prendre le plus gros (= coureur en
            # avant-plan plutôt que badaud/fond).
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            embeddings.append(face.normed_embedding.astype(np.float32))
            uciids.append(uciid)
            photo_tags.append(f"{photo.race}:{photo.type}")
            rider_names.append(rider.name)

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

    emb_arr = np.stack(embeddings)
    print(f"\nFait en {elapsed:.1f}s ({n_done / elapsed:.1f} photos/s) :")
    print(f"  embeddings retenus : {n_kept} / {n_photos_total}")
    print(f"  skipped (lecture)  : {skipped_read_fail}")
    print(f"  skipped (no face)  : {skipped_no_face}")
    print(f"  shape emb_arr      : {emb_arr.shape}, dtype {emb_arr.dtype}")

    # Coureurs effectivement représentés dans l'index
    unique_riders = len(set(uciids))
    print(f"  coureurs distincts indexés : {unique_riders} / {len(riders)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=emb_arr,
        uciids=np.array(uciids),
        photo_tags=np.array(photo_tags),
        names=np.array(rider_names),
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nIndex écrit : {output_path} ({size_mb:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--photos-dir",
        type=Path,
        default=DEFAULT_PHOTOS_DIR,
        help=f"Racine du dataset signatureNG (défaut: {DEFAULT_PHOTOS_DIR})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/embeddings.npz"),
        help="Chemin de sortie de l'index (défaut: data/embeddings.npz)",
    )
    ap.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Taille d'entrée du détecteur (défaut 640)",
    )
    ap.add_argument(
        "--ctx-id",
        type=int,
        default=0,
        help="GPU id (-1 pour CPU, défaut 0)",
    )
    args = ap.parse_args()

    build_index(args.photos_dir, args.output, args.det_size, args.ctx_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
