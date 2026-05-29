#!/usr/bin/env python3
"""snapshot_review.py — review humain des snapshots auto-générés.

Lit SNAPSHOTS_DIR/<uciid>/*.jpg (publiés par face_recog_service.py en
mode SNAPSHOT_ENABLE=1), affiche par rider :
  - les photos déjà présentes dans face-db/<uciid>/
  - les snapshots staging candidats

L'opérateur décide pour chaque rider :
  (a) accept all   → move tous les snapshots vers face-db/<uciid>/
  (r) reject all   → supprime tous les snapshots
  (i) individual   → review snapshot-par-snapshot
  (s) skip         → laisse en staging pour plus tard
  (q) quit         → arrête (le reste reste en staging)

Mode individual prompt y/n/s pour chaque snapshot :
  (y) accept       → move vers face-db
  (n) reject       → supprime
  (s) skip         → laisse en staging
  (o) open         → ouvre l'image avec xdg-open (eog, feh, etc.)

Le mapping uciid → name vient du manifest rider-recognition (passé en
arg --manifest). Sans manifest, on affiche juste l'uciid.

Promotions loggées dans SNAPSHOTS_DIR/_promoted.log (1 ligne JSON / move).

Usage typique :
    snapshot_review.py \\
        --snapshots-dir /var/lib/face-recog/snapshots \\
        --face-db-dir   /var/lib/avtowan/face-db \\
        --manifest      /var/lib/avtowan/rider_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def load_manifest(path: str | None) -> dict[str, dict]:
    """{uciid: {name, ...}} depuis le manifest rider-recognition."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[review] manifest échec lecture: {e}", file=sys.stderr)
        return {}


def list_riders_with_snapshots(snapshots_dir: str) -> list[str]:
    """Retourne la liste des uciid ayant au moins 1 snapshot en staging."""
    if not os.path.isdir(snapshots_dir):
        return []
    out = []
    for entry in sorted(os.listdir(snapshots_dir)):
        if entry.startswith("_"):
            continue  # _promoted.log, _meta.json, etc.
        rider_dir = os.path.join(snapshots_dir, entry)
        if not os.path.isdir(rider_dir):
            continue
        jpgs = [f for f in os.listdir(rider_dir)
                if f.lower().endswith(".jpg")]
        if jpgs:
            out.append(entry)
    return out


def list_snapshots(rider_dir: str) -> list[str]:
    """Liste triée des JPG dans le dossier rider, du plus ancien au + récent."""
    if not os.path.isdir(rider_dir):
        return []
    files = [f for f in os.listdir(rider_dir)
             if f.lower().endswith(".jpg")]
    return sorted(files)


def list_face_db_photos(face_db_dir: str, uciid: str) -> list[str]:
    """Liste les photos existantes dans face-db/<uciid>/ (full paths)."""
    folder = os.path.join(face_db_dir, uciid)
    if not os.path.isdir(folder):
        return []
    out = []
    for f in sorted(os.listdir(folder)):
        low = f.lower()
        if low.endswith((".jpg", ".jpeg", ".png", ".webp")):
            out.append(os.path.join(folder, f))
    return out


def open_image(path: str) -> None:
    """Ouvre l'image avec le viewer par défaut (non-blocking)."""
    try:
        subprocess.Popen(
            ["xdg-open", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        # Fallback : print le chemin, l'op ouvre manuellement
        print(f"  (ouvre manuellement : {path})")


def log_promoted(snapshots_dir: str, entry: dict) -> None:
    """Append une ligne JSON dans _promoted.log."""
    log_path = os.path.join(snapshots_dir, "_promoted.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def promote(src_path: str, face_db_dir: str, uciid: str,
            snapshots_dir: str) -> str | None:
    """Move snapshot vers face-db/<uciid>/. Renvoie le destination path
    ou None si échec.

    Le filename de destination est préfixé "snap_" + timestamp d'origine
    pour distinguer des photos officielles dataset (typiquement aso_*).
    """
    rider_folder = os.path.join(face_db_dir, uciid)
    try:
        os.makedirs(rider_folder, exist_ok=True)
    except OSError as e:
        print(f"  [erreur] mkdir {rider_folder}: {e}")
        return None
    base = os.path.basename(src_path)
    dest_name = f"snap_{base}"
    dest = os.path.join(rider_folder, dest_name)
    # Si collision (improbable, timestamp inclus), ajoute un suffixe.
    i = 0
    while os.path.exists(dest):
        i += 1
        stem, ext = os.path.splitext(dest_name)
        dest = os.path.join(rider_folder, f"{stem}_{i}{ext}")
    try:
        shutil.move(src_path, dest)
    except OSError as e:
        print(f"  [erreur] move {src_path} → {dest}: {e}")
        return None
    log_promoted(snapshots_dir, {
        "ts": time.time(),
        "uciid": uciid,
        "src": src_path,
        "dest": dest,
    })
    return dest


def reject(src_path: str) -> bool:
    """Supprime un snapshot rejeté."""
    try:
        os.remove(src_path)
        return True
    except OSError as e:
        print(f"  [erreur] remove {src_path}: {e}")
        return False


def review_rider(uciid: str, name: str, snapshots_dir: str,
                 face_db_dir: str) -> dict:
    """Boucle de review pour 1 rider. Retourne stats du passage."""
    stats = {"accepted": 0, "rejected": 0, "skipped": 0}
    rider_dir = os.path.join(snapshots_dir, uciid)
    snaps = list_snapshots(rider_dir)
    existing = list_face_db_photos(face_db_dir, uciid)

    print("\n" + "=" * 72)
    print(f"Rider {uciid}{' — ' + name if name else ''}")
    print(f"  Photos déjà en face-db : {len(existing)}")
    for p in existing[:5]:
        print(f"    {p}")
    if len(existing) > 5:
        print(f"    ... +{len(existing) - 5} autres")
    print(f"  Snapshots staging : {len(snaps)}")
    for i, s in enumerate(snaps, 1):
        print(f"    [{i}] {s}")

    while True:
        action = input("Actions: (a)ccept all | (r)eject all | (i)ndividual"
                       " | (s)kip rider | (q)uit > ").strip().lower()
        if action == "a":
            for s in snaps:
                src = os.path.join(rider_dir, s)
                if promote(src, face_db_dir, uciid, snapshots_dir):
                    stats["accepted"] += 1
            return stats
        elif action == "r":
            for s in snaps:
                src = os.path.join(rider_dir, s)
                if reject(src):
                    stats["rejected"] += 1
            return stats
        elif action == "s":
            stats["skipped"] = len(snaps)
            return stats
        elif action == "q":
            stats["skipped"] = len(snaps)
            stats["quit"] = True
            return stats
        elif action == "i":
            for s in snaps:
                src = os.path.join(rider_dir, s)
                print(f"\n  Snapshot : {s}")
                while True:
                    sub = input("    (y)accept | (n)reject | (s)kip | "
                                "(o)open viewer > ").strip().lower()
                    if sub == "y":
                        if promote(src, face_db_dir, uciid, snapshots_dir):
                            stats["accepted"] += 1
                        break
                    elif sub == "n":
                        if reject(src):
                            stats["rejected"] += 1
                        break
                    elif sub == "s":
                        stats["skipped"] += 1
                        break
                    elif sub == "o":
                        open_image(src)
                        # Ne pas break — on attend une vraie décision
                    else:
                        print("    réponse non comprise (y/n/s/o)")
            return stats
        else:
            print("Réponse non comprise (a/r/i/s/q)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshots-dir",
                    default="/var/lib/face-recog/snapshots",
                    help="Dossier de staging des snapshots (default: %(default)s)")
    ap.add_argument("--face-db-dir",
                    default="/var/lib/avtowan/face-db",
                    help="Dossier face-db cible pour les promotions "
                         "(default: %(default)s)")
    ap.add_argument("--manifest",
                    default="/var/lib/avtowan/rider_manifest.json",
                    help="Manifest rider-recognition pour résoudre uciid→name "
                         "(default: %(default)s)")
    args = ap.parse_args()

    if not os.path.isdir(args.snapshots_dir):
        print(f"[review] snapshots dir absent : {args.snapshots_dir}")
        return 1

    manifest = load_manifest(args.manifest)
    riders = list_riders_with_snapshots(args.snapshots_dir)
    if not riders:
        print("[review] aucun snapshot en staging — rien à faire.")
        return 0

    total = {"accepted": 0, "rejected": 0, "skipped": 0}
    print(f"\n[review] {len(riders)} rider(s) avec snapshots en attente")
    print(f"[review] snapshots dir : {args.snapshots_dir}")
    print(f"[review] face-db dir   : {args.face_db_dir}")

    for uciid in riders:
        name = ""
        if manifest and uciid in manifest:
            entry = manifest[uciid]
            if isinstance(entry, dict):
                name = entry.get("name", "")
        stats = review_rider(uciid, name, args.snapshots_dir,
                             args.face_db_dir)
        for k in ("accepted", "rejected", "skipped"):
            total[k] += stats.get(k, 0)
        if stats.get("quit"):
            break

    print("\n" + "=" * 72)
    print(f"[review] terminé. accepted={total['accepted']} "
          f"rejected={total['rejected']} skipped={total['skipped']}")
    print(f"  log promotions → {args.snapshots_dir}/_promoted.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
