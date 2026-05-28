#!/usr/bin/env python3
"""convert_index.py — face-index.npz → embeddings.bin + names.txt.

Pour notre custom nvinfer parser C++ qui matche via cublas sgemv en
mémoire GPU, on a besoin d'un format binaire trivial (float32 contigu)
qu'on lit en C++ sans dépendance npz.

Layout output :
    embeddings.bin : raw float32 [N × DIM] row-major (N×512 pour
                     buffalo_l ArcFace w600k_r50). Pré-normalisé L2
                     ici pour économiser cette passe runtime.
    names.txt      : 1 nom par ligne, ordre identique à embeddings.

Usage :
    convert_index.py /var/lib/face-recog/face-index.npz \\
                     /opt/face-recog-ds/index/
"""
from __future__ import annotations

import os
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <face-index.npz> <output-dir>",
              file=sys.stderr)
        return 1
    src = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    z = np.load(src, allow_pickle=True)
    embeddings = z["embeddings"].astype(np.float32)
    names = z["names"]

    # Re-normalize L2 par sécurité (l'index actuel l'est déjà côté
    # build_index.py, mais c'est cheap et garantit le bon mapping cosine
    # = dot product).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0  # éviter div par zéro
    embeddings = embeddings / norms

    emb_path = os.path.join(out_dir, "embeddings.bin")
    names_path = os.path.join(out_dir, "names.txt")

    embeddings.tofile(emb_path)
    with open(names_path, "w", encoding="utf-8") as f:
        for n in names:
            f.write(str(n) + "\n")

    print(f"[convert_index] {embeddings.shape[0]} embeddings × "
          f"{embeddings.shape[1]} dims → {emb_path}")
    print(f"[convert_index] {len(names)} names → {names_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
