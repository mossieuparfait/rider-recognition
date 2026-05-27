"""Détection de pose (YOLOv8-pose) + tracking BoT-SORT.

Pour chaque personne dans la frame, sort :
- bbox englobante + track_id stable cross-frames
- 17 keypoints COCO (avec confiance par keypoint)
- face_kp = position estimée du visage (nez si confiant, sinon moyenne
  yeux), utile pour estimer où devrait être le visage même quand
  RetinaFace ne le détecte pas (casque baissé, profil partiel).

Beaucoup plus léger que bib_recog (pas d'OCR) → ~30 fps en CUDA sur
RTX 4060 avec yolov8n-pose.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# Filtres anti-public (mêmes seuils que bib_recog).
_PERSON_MIN_H = 150
_PERSON_MIN_CONF = 0.5
# Confiance minimum sur un keypoint pour le considérer fiable.
_KP_MIN_CONF = 0.3

# Indices COCO keypoints.
KP_NOSE         = 0
KP_LEFT_EYE     = 1
KP_RIGHT_EYE    = 2
KP_LEFT_EAR     = 3
KP_RIGHT_EAR    = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6


@dataclass
class PersonPose:
    """Pose d'une personne dans la frame.

    person_bbox : (x1, y1, x2, y2) bbox englobante YOLO
    track_id    : id stable cross-frames du tracker BoT-SORT (None si
                  pas encore confirmé)
    confidence  : conf YOLO de la détection person
    keypoints   : liste de 17 tuples (x, y, conf) — COCO ordre standard
    face_kp     : (x, y, conf) position estimée du visage, None si aucun
                  keypoint face fiable
    """

    person_bbox: tuple[int, int, int, int]
    track_id: int | None
    confidence: float
    keypoints: list[tuple[float, float, float]]
    face_kp: tuple[float, float, float] | None


def _estimate_face_kp(
    kps: list[tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Estime la position du visage depuis les keypoints.

    Priorité : nez seul (si confiant) → moyenne des deux yeux → None.
    On évite d'estimer depuis les épaules (trop imprécis verticalement).
    """
    nose = kps[KP_NOSE]
    if nose[2] >= _KP_MIN_CONF:
        return nose
    leye = kps[KP_LEFT_EYE]
    reye = kps[KP_RIGHT_EYE]
    if leye[2] >= _KP_MIN_CONF and reye[2] >= _KP_MIN_CONF:
        return (
            (leye[0] + reye[0]) / 2,
            (leye[1] + reye[1]) / 2,
            (leye[2] + reye[2]) / 2,
        )
    return None


class BodyRecognizer:
    """YOLOv8-pose + BoT-SORT tracker. Chargé 1× au boot."""

    def __init__(self, weights: str = "yolov8n-pose.pt",
                 gpu_id: int = -1) -> None:
        from ultralytics import YOLO

        self.yolo = YOLO(weights)
        self._device = "cuda" if gpu_id >= 0 else "cpu"
        if gpu_id >= 0:
            self.yolo.to(f"cuda:{gpu_id}")

    def detect(self, frame_bgr: np.ndarray,
               verbose: bool = False) -> list[PersonPose]:
        # classes=[0] = 'person' COCO ; persist=True = BoT-SORT cross-frames.
        results = self.yolo.track(frame_bgr, classes=[0], persist=True,
                                  verbose=False, device=self._device)
        out: list[PersonPose] = []
        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
            for box, kp_data in zip(r.boxes, r.keypoints):
                px1, py1, px2, py2 = box.xyxy[0].cpu().numpy().astype(int)
                ph = py2 - py1
                yolo_conf = float(box.conf[0])
                if verbose:
                    print(f"[body] bbox=({px1},{py1},{px2},{py2}) "
                          f"h={ph} conf={yolo_conf:.2f}")
                if ph < _PERSON_MIN_H or yolo_conf < _PERSON_MIN_CONF:
                    if verbose:
                        print("  → skip (filtres)")
                    continue

                track_id = int(box.id[0]) if box.id is not None else None
                # kp_data.data : tensor (1, 17, 3) = (x, y, conf) par kp
                kp_arr = kp_data.data[0].cpu().numpy()
                keypoints = [tuple(float(v) for v in row) for row in kp_arr]
                face_kp = _estimate_face_kp(keypoints)

                out.append(PersonPose(
                    person_bbox=(int(px1), int(py1), int(px2), int(py2)),
                    track_id=track_id,
                    confidence=yolo_conf,
                    keypoints=keypoints,
                    face_kp=face_kp,
                ))
                if verbose:
                    print(f"  ✓ track_id={track_id} face_kp={face_kp}")
        return out


def _cli() -> int:
    """Test standalone : pose sur une image PNG/JPG.

    Usage:
        python -m rider_recognition.body_recog /chemin/vers/frame.jpg
    """
    import time
    if len(sys.argv) < 2:
        print("Usage: python -m rider_recognition.body_recog <image>")
        return 1

    img_path = Path(sys.argv[1])
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Erreur : impossible de lire {img_path}")
        return 1

    print(f"Image {img.shape[1]}×{img.shape[0]}, init BodyRecognizer...")
    rec = BodyRecognizer()
    print("Inférence (verbose)...")
    t0 = time.time()
    persons = rec.detect(img, verbose=True)
    dt = (time.time() - t0) * 1000
    print(f"\n{len(persons)} personnes en {dt:.0f} ms")
    for p in persons:
        kp_ok = sum(1 for k in p.keypoints if k[2] >= _KP_MIN_CONF)
        print(f"  track_id={p.track_id} bbox={p.person_bbox} "
              f"face_kp={p.face_kp} ({kp_ok}/17 kp fiables)")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
