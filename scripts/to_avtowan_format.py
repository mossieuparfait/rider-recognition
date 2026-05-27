#!/usr/bin/env python3
"""Convertit notre index multi-embedding vers le format AVtoWan face-recog.

Format AVtoWan (cf videoWan/cmd/avtowan-face-recog/face_recog_service.py) :
    embeddings : (N, 512) float32, L2-normalisé
    names      : (N,)    str        une entrée par personne

Notre format : 1 embedding PAR photo (donc plusieurs lignes par coureur).
Conversion : on moyenne les embeddings d'un même UCIID puis on re-normalise L2.

Nom final = "<Nom> (<UCIID>)" — garde l'UCIID pour traçabilité tout en
affichant le nom humain dans la webui.

Usage:
    python3 scripts/to_avtowan_format.py \\
        --in data/embeddings.npz \\
        --out /var/lib/avtowan/face-index.npz \\
        [--backup]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def convert(in_path: Path, out_path: Path, backup: bool = True) -> None:
    print(f"Lecture index multi : {in_path}")
    data = np.load(in_path, allow_pickle=True)
    embs = data["embeddings"].astype(np.float32)  # (N_photos, 512)
    uciids = data["uciids"]
    names = data["names"]
    print(f"  → {embs.shape[0]} embeddings, {len(set(uciids))} coureurs distincts")

    # Regroupe par UCIID → mean → re-normalise L2
    by_uciid: dict[str, list[np.ndarray]] = {}
    name_by_uciid: dict[str, str] = {}
    for emb, uciid, name in zip(embs, uciids, names):
        by_uciid.setdefault(str(uciid), []).append(emb)
        name_by_uciid[str(uciid)] = str(name)

    out_embs = np.empty((len(by_uciid), 512), dtype=np.float32)
    out_names = []
    for i, uciid in enumerate(sorted(by_uciid)):
        stack = np.stack(by_uciid[uciid])  # (k, 512), tous déjà L2-normalisés
        mean = stack.mean(axis=0)
        # Renormalise (la moyenne de N vecteurs unitaires n'est plus unitaire)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        out_embs[i] = mean
        out_names.append(name_by_uciid[uciid])

    out_names_arr = np.array(out_names, dtype=object)
    print(f"  → format AVtoWan : {out_embs.shape}, {len(out_names)} noms")

    # Backup si la cible existe déjà
    if backup and out_path.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = out_path.with_suffix(f".npz.bak-{ts}")
        shutil.copy2(out_path, bak)
        print(f"  backup ancien index → {bak}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=out_embs, names=out_names_arr)
    size_kb = out_path.stat().st_size / 1024
    print(f"Écrit : {out_path} ({size_kb:.1f} KB)")
    print(f"\nExemples de noms (5 premiers) :")
    for n in out_names[:5]:
        print(f"  {n}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    convert(args.in_path, args.out_path, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
