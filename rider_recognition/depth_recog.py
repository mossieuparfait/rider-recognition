"""Estimation de profondeur monoculaire via DepthAnything-V2-Small.

Sortie : depth map uint8 (0-255), même résolution que l'input. 255 =
proche caméra, 0 = arrière-plan lointain. Convention DepthAnything.

Perf RTX 4060 : ~125 ms par frame 1080p après warmup. Pour bullet-time
qui freeze une frame, une seule passe suffit ; en continu on tourne à
basse cadence (2 fps) pour avoir une depth fraîche dispo à tout moment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


class DepthRecognizer:
    """Wrapper DepthAnything-V2-Small via transformers pipeline."""

    def __init__(self, gpu_id: int = -1) -> None:
        from transformers import pipeline
        from PIL import Image as PILImage
        self._PILImage = PILImage
        device = gpu_id if gpu_id >= 0 else -1
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Retourne la depth map uint8 (h, w) pour la frame BGR."""
        img = self._PILImage.fromarray(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        res = self.pipe(img)
        return np.array(res["depth"], dtype=np.uint8)


def _cli() -> int:
    """Test standalone : depth sur 1 image, sauve une preview colorisée."""
    import time
    if len(sys.argv) < 2:
        print("Usage: python -m rider_recognition.depth_recog <image>")
        return 1
    img_path = Path(sys.argv[1])
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Erreur : impossible de lire {img_path}")
        return 1
    print(f"Image {img.shape[1]}×{img.shape[0]}, init DepthRecognizer...")
    rec = DepthRecognizer(gpu_id=0)
    for i in range(3):
        t0 = time.time()
        depth = rec.detect(img)
        dt = (time.time() - t0) * 1000
        print(f"  run {i+1}: {dt:.0f}ms shape={depth.shape} "
              f"range={depth.min()}-{depth.max()}")
    out = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    out_path = img_path.with_suffix(".depth.jpg")
    cv2.imwrite(str(out_path), out)
    print(f"Preview colorisée : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
