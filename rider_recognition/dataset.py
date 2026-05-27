"""Chargement du dataset signatureNG (manifest + photos sur disque)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Path par défaut (dev local). À surcharger via env ou paramètre quand
# signatureNG migrera sur autre machine.
DEFAULT_PHOTOS_DIR = Path(
    "/home/ben/AIlocal/signatureNG/signature/public/data/rider_photos"
)
MANIFEST_FILENAME = "_manifest.json"


@dataclass
class Photo:
    """Une photo d'un coureur, telle que listée dans le manifest."""

    race: str          # ex: "VUE2024", "TDF2025", "PRX2026"
    type: str          # ex: "podium" (660×1000), "portrait" (400×400)
    url: str           # URL CDN img.aso.fr d'origine
    path: Path | None  # chemin local si fichier présent, sinon None


@dataclass
class Rider:
    """Un coureur avec ses photos disponibles localement."""

    uciid: str
    name: str
    photos: list[Photo] = field(default_factory=list)

    @property
    def has_local_photos(self) -> bool:
        return any(p.path is not None for p in self.photos)

    @property
    def local_photos(self) -> list[Photo]:
        return [p for p in self.photos if p.path is not None]


def load_dataset(photos_dir: Path = DEFAULT_PHOTOS_DIR) -> dict[str, Rider]:
    """Charge le manifest + croise avec les fichiers présents.

    Retourne `{uciid: Rider}` avec uniquement les coureurs ayant au moins
    un dossier local (même si certaines photos du manifest manquent en
    local).
    """
    manifest_path = photos_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest introuvable : {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    riders: dict[str, Rider] = {}
    for uciid, entry in manifest.items():
        rider_dir = photos_dir / uciid
        if not rider_dir.is_dir():
            # UCIID listé au manifest mais pas téléchargé localement → skip
            continue

        # On indexe les fichiers du dossier par (type, race) pour matcher
        # vs ce que dit le manifest.
        local_by_tag: dict[tuple[str, str], Path] = {}
        for f in rider_dir.iterdir():
            if not f.is_file() or not f.suffix.lower() == ".png":
                continue
            # nom : NN_<type>_<RACE><YEAR>.png  → on extrait type + race
            stem = f.stem  # ex: "01_podium_VUE2024"
            parts = stem.split("_", 2)
            if len(parts) != 3:
                continue
            _, ptype, race = parts
            local_by_tag[(ptype, race)] = f

        photos: list[Photo] = []
        for p in entry.get("photos", []):
            tag = (p.get("type", ""), p.get("race", ""))
            photos.append(
                Photo(
                    race=p.get("race", ""),
                    type=p.get("type", ""),
                    url=p.get("url", ""),
                    path=local_by_tag.get(tag),
                )
            )

        riders[uciid] = Rider(
            uciid=uciid, name=entry.get("name", ""), photos=photos
        )

    return riders
