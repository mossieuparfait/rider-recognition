#!/usr/bin/env python3
"""Rapport sur le dataset signatureNG chargé localement.

Usage:
    python3 scripts/scan_dataset.py
    python3 scripts/scan_dataset.py /chemin/vers/rider_photos
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Permet d'exécuter le script sans installer le package
sys.path.insert(0, str(Path(__file__).parent.parent))

from rider_recognition.dataset import DEFAULT_PHOTOS_DIR, load_dataset


def main() -> int:
    photos_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PHOTOS_DIR

    print(f"Dataset : {photos_dir}")
    riders = load_dataset(photos_dir)
    print(f"Coureurs avec dossier local : {len(riders)}")
    print()

    # Combien ont au moins une photo physiquement présente
    with_local = [r for r in riders.values() if r.has_local_photos]
    print(f"Coureurs avec ≥1 photo locale : {len(with_local)}")

    # Distribution du nombre de photos locales par coureur
    counts = Counter(len(r.local_photos) for r in riders.values())
    print(f"\nDistribution nb photos locales / coureur :")
    for n in sorted(counts):
        print(f"  {n:>2} photo(s) : {counts[n]:>4} coureurs")

    # Distribution par course (race tag)
    race_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for r in riders.values():
        for p in r.local_photos:
            race_counts[p.race] += 1
            type_counts[p.type] += 1

    print(f"\nPhotos locales par course :")
    for race, n in race_counts.most_common():
        print(f"  {race:>10} : {n:>5}")

    print(f"\nPhotos locales par type :")
    for ptype, n in type_counts.most_common():
        print(f"  {ptype:>10} : {n:>5}")

    # Quelques exemples
    print(f"\nExemples (5 premiers coureurs avec photos) :")
    for r in list(with_local)[:5]:
        print(f"  {r.uciid}  {r.name}")
        for p in r.local_photos[:3]:
            print(f"      [{p.type:>8} {p.race:>8}] {p.path.name if p.path else '—'}")
        if len(r.local_photos) > 3:
            print(f"      ... +{len(r.local_photos) - 3} autres")

    return 0


if __name__ == "__main__":
    sys.exit(main())
