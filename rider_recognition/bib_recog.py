"""Détection de personnes (YOLOv8) + OCR des dossards (PaddleOCR).

Pipeline :
    1. YOLOv8n détecte les bbox 'person' dans la frame.
    2. Pour chaque personne, crop la zone supérieure (dos/torse, ~60% haut).
    3. PaddleOCR cherche du texte dans le crop.
    4. Filtre les résultats : seuls 1-3 chiffres dans [1, 250] retenus
       (plage typique des dossards UCI).

Inférence à appeler dans un thread séparé du display loop (idem face) car
~50-150 ms par frame selon densité de personnes.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BibDetection:
    """Un dossard détecté + lu dans la frame.

    bib_bbox  : (x1,y1,x2,y2) du texte du dossard dans la frame (coord globales)
    person_bbox : (x1,y1,x2,y2) de la personne englobante (pour matching face)
    bib       : numéro lu (int)
    confidence: score OCR [0..1]
    """

    bib_bbox: tuple[int, int, int, int]
    person_bbox: tuple[int, int, int, int]
    bib: int
    confidence: float


# Zone à OCR-iser : toute la personne + extension en bas pour capturer le
# dossard sur le cadre du vélo (= 3ème occurrence du même numéro en plus
# des 2 sur le dos). Plus on a de lectures, plus le vote est fiable.
_CROP_BOTTOM_EXTENSION = 0.35  # +35% de la hauteur person sous py2
# Filtres anti-bruit : élimine le public arrière-plan (petit + détection
# basse confiance) avant l'OCR. Sur 1080p :
#   - h<150 px = personne trop lointaine, dossard ≤ 30 px = illisible
#   - YOLO conf<0.5 = détection ambigüe (bord, occlusion, faux positif)
# Effet typique : passe de 10 persons OCR-isées à 2-3 = 3-5× plus rapide.
_PERSON_MIN_H = 150
_PERSON_MIN_CONF = 0.5
# Plage des dossards UCI standard course route (1-220 typiquement, on
# laisse marge à 250 pour les courses élargies type Vuelta).
_BIB_MIN, _BIB_MAX = 1, 250
_BIB_LEN_MAX = 4  # brut max accepté avant tentative de split


def _refine_bib(raw: str) -> int | None:
    """Extrait un dossard valide depuis une lecture OCR brute.

    Si raw est dans [_BIB_MIN, _BIB_MAX] tel quel, garde. Sinon, essaie
    de splitter en sous-chaînes 1-3 digits et garde la plus longue qui
    matche la plage UCI. Utile quand PaddleOCR colle 2 dossards adjacents
    en un seul mot (ex: "5125" → "125" pour un coureur qui a son numéro
    répété 2 fois sur le dos, lu comme un bloc).
    """
    if not raw.isdigit():
        return None
    if len(raw) <= 3:
        n = int(raw)
        return n if _BIB_MIN <= n <= _BIB_MAX else None
    # Long > 3 digits → essaie de splitter en sous-chaînes 3 → 2 → 1
    # chiffres, retient le plus long valide.
    for sub_len in (3, 2, 1):
        for i in range(0, len(raw) - sub_len + 1):
            sub = raw[i:i + sub_len]
            if sub.startswith("0"):
                continue
            n = int(sub)
            if _BIB_MIN <= n <= _BIB_MAX:
                return n
    return None


class BibRecognizer:
    """Stack YOLOv8 person + PaddleOCR. Chargé 1× au boot."""

    def __init__(self, yolo_weights: str = "yolov8n.pt",
                 gpu_id: int = -1) -> None:
        """gpu_id : -1 = CPU pour tout, >=0 = YOLO sur ce GPU + OCR CPU.

        Note : on n'utilise pas paddle-gpu (lourd + conflit cuDNN avec
        torch). OCR reste CPU même si YOLO en CUDA.
        """
        # Imports différés : ces libs sont lourdes (~3-5s à importer).
        from ultralytics import YOLO
        from paddleocr import PaddleOCR

        self.yolo = YOLO(yolo_weights)
        self._yolo_device = "cuda" if gpu_id >= 0 else "cpu"
        if gpu_id >= 0:
            self.yolo.to(f"cuda:{gpu_id}")
        # PaddleOCR : use_angle_cls=False → dossards supposés à l'endroit
        # (gain ~20ms). use_gpu=False imposé pour éviter conflit cuDNN.
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
            show_log=False,
        )

    def detect(self, frame_bgr: np.ndarray,
               verbose: bool = False) -> list[BibDetection]:
        """Détecte personnes + OCR dossards sur la frame.

        verbose=True : log tout ce que YOLO et OCR détectent avant filtres,
        utile pour calibrer.
        """
        out: list[BibDetection] = []
        # YOLOv8 classes=[0] = 'person' (COCO).
        results = self.yolo(frame_bgr, classes=[0], verbose=False,
                            device=self._yolo_device)
        if not results:
            if verbose:
                print("[bib_recog] YOLO: aucun résultat")
            return out

        fh, fw = frame_bgr.shape[:2]
        n_persons = 0
        for r in results:
            if r.boxes is None:
                continue
            n_persons += len(r.boxes)
            for box in r.boxes:
                px1, py1, px2, py2 = box.xyxy[0].cpu().numpy().astype(int)
                ph = py2 - py1
                yolo_conf = float(box.conf[0])
                if verbose:
                    print(f"[bib_recog] person bbox=({px1},{py1},{px2},{py2}) "
                          f"h={ph} conf={yolo_conf:.2f}")
                if ph < _PERSON_MIN_H:
                    if verbose:
                        print(f"  → skip (h<{_PERSON_MIN_H} px)")
                    continue
                if yolo_conf < _PERSON_MIN_CONF:
                    if verbose:
                        print(f"  → skip (yolo conf<{_PERSON_MIN_CONF})")
                    continue
                # Crop = personne entière + extension en bas pour capturer
                # le dossard du cadre vélo (3ème occurrence du même numéro).
                crop_y2 = min(fh, py2 + int(ph * _CROP_BOTTOM_EXTENSION))
                crop = frame_bgr[py1:crop_y2, px1:px2]
                if crop.size == 0:
                    continue

                ocr_res = self.ocr.ocr(crop, cls=False)
                if not ocr_res or not ocr_res[0]:
                    if verbose:
                        print(f"  → OCR: rien")
                    continue

                # Récupère TOUTES les lectures numériques, raffine chaque
                # brute via _refine_bib (qui sait splitter "5125" → "125").
                candidates: list[tuple[int, float, list]] = []
                for line in ocr_res[0]:
                    poly, (text, conf) = line
                    raw = text.strip()
                    if verbose:
                        print(f"  → OCR: '{raw}' conf={conf:.2f}", end="")
                    if not raw.isdigit() or len(raw) > _BIB_LEN_MAX:
                        if verbose:
                            print(" → ignoré (non numérique ou trop long)")
                        continue
                    bib = _refine_bib(raw)
                    if bib is None:
                        if verbose:
                            print(" → ignoré (aucun split valide)")
                        continue
                    if verbose and str(bib) != raw:
                        print(f" → raffiné #{bib}")
                    elif verbose:
                        print()
                    candidates.append((bib, float(conf), poly))

                if not candidates:
                    continue

                # Vote : si on a 2+ lectures, le numéro le plus fréquent
                # gagne ; sinon (1 lecture seule) on garde tel quel.
                counter = Counter(c[0] for c in candidates)
                best_bib, count = counter.most_common(1)[0]

                # Bbox + conf du candidat retenu (premier qui matche).
                chosen = next(c for c in candidates if c[0] == best_bib)
                poly = chosen[2]
                bib = best_bib
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bx1 = int(px1 + min(xs))
                by1 = int(py1 + min(ys))
                bx2 = int(px1 + max(xs))
                by2 = int(py1 + max(ys))
                out.append(BibDetection(
                    bib_bbox=(bx1, by1, bx2, by2),
                    person_bbox=(int(px1), int(py1), int(px2), int(py2)),
                    bib=bib,
                    confidence=chosen[1],
                ))
                if verbose:
                    print(f"  ✓ retenu : #{bib} (vote {count}/{len(candidates)})")
        if verbose:
            print(f"[bib_recog] total YOLO persons={n_persons}, "
                  f"dossards retenus={len(out)}")
        return out


def _cli() -> int:
    """Test standalone : OCR sur une image PNG/JPG donnée en argument.

    Usage:
        python -m rider_recognition.bib_recog /chemin/vers/frame.jpg
    """
    if len(sys.argv) < 2:
        print("Usage: python -m rider_recognition.bib_recog <image>")
        return 1

    img_path = Path(sys.argv[1])
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Erreur : impossible de lire {img_path}")
        return 1

    print(f"Image {img.shape[1]}×{img.shape[0]}, init BibRecognizer (peut prendre 10s)...")
    rec = BibRecognizer()
    print("Inférence (verbose)...")
    import time
    t0 = time.time()
    dets = rec.detect(img, verbose=True)
    dt = (time.time() - t0) * 1000
    print(f"\n{len(dets)} dossard(s) retenu(s) en {dt:.0f} ms")
    for d in dets:
        print(f"  #{d.bib:>3}  conf={d.confidence:.2f}  bbox={d.bib_bbox}  person={d.person_bbox}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
