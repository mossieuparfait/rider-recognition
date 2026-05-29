#!/usr/bin/env python3
"""ndi_sender.py — capture V4L2 (Magewell SDI) → broadcast NDI HB.

Pourquoi : sépare le détenteur du device V4L2 (ce sender) de la
consommation pour reco/incrustation/monitoring (clients NDI). Permet à
plusieurs consommateurs (face-recog local, future new box, OBS sur le
LAN, etc.) de lire le même flux sans contention sur /dev/videoN.

Latence ajoutée : ~30-50 ms (encoding SpeedHQ intra-frame). Acceptable
pour monitoring + graphics broadcast, pas pour broadcast hot-path.

Variables d'env :
    SOURCE        v4l2:/dev/video0    source V4L2 Magewell
    NDI_NAME      AVtoWan-FaceRecog   nom du flux NDI visible sur le LAN
    WIDTH         1920                largeur output
    HEIGHT        1080                hauteur output
    FPS           60                  framerate output

Le flux NDI est discoverable via tout client NDI (OBS, vMix, mpv avec
plugin NDI, cyndilib Finder, NDI Studio Monitor, etc.). Pas
d'authentification, comme c'est la convention NDI sur LAN trust.
"""

import os
import sys
import time
from fractions import Fraction

# Bootstrap CUDA libs avant tout import lourd (cf face_recog_service.py).
# Pas strictement nécessaire pour le sender (pas de GPU) mais on garde
# le pattern par cohérence si on partage du code plus tard.
import ctypes, glob
_nv_glob = os.path.join(os.path.dirname(sys.executable),
                        "..", "lib", "python*", "site-packages", "nvidia",
                        "*", "lib", "lib*.so*")
for _lib in sorted(glob.glob(_nv_glob)):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import cv2
import numpy as np

from cyndilib import Sender
from cyndilib.video_frame import VideoSendFrame
from cyndilib.wrapper.ndi_structs import FourCC, FrameFormat


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


SOURCE   = env("SOURCE", "v4l2:/dev/video0")
NDI_NAME = env("NDI_NAME", "AVtoWan-FaceRecog")
WIDTH    = int(env("WIDTH", "1920"))
HEIGHT   = int(env("HEIGHT", "1080"))
FPS      = int(env("FPS", "60"))


def log(msg: str) -> None:
    print(f"[ndi-sender] {msg}", flush=True)


def open_v4l2(dev: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"FATAL: ouverture {dev} échec (signal SDI absent ? "
                 f"device tenu par un autre process ?)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = cap.get(cv2.CAP_PROP_FPS)
    log(f"V4L2 {dev} ouvert : {w}x{h} @ {f:.1f}fps")
    return cap


def main() -> None:
    log(f"démarrage SOURCE={SOURCE} NDI_NAME={NDI_NAME} "
        f"{WIDTH}x{HEIGHT}@{FPS}fps")

    if not SOURCE.startswith("v4l2:"):
        sys.exit(f"FATAL: SOURCE non supporté (attendu v4l2:/dev/...) : {SOURCE}")
    cap = open_v4l2(SOURCE[len("v4l2:"):])

    # NDI sender + VideoSendFrame. BGRA car OpenCV retourne BGR — on ajoute
    # un canal alpha = 255 partout. cyndilib réencode en SpeedHQ 4:2:0 en
    # interne donc le 4:4:4 du BGRA est downsampled, mais ça reste
    # visuellement lossless pour notre usage monitoring.
    sender = Sender(NDI_NAME)
    vframe = VideoSendFrame()
    vframe.set_resolution(WIDTH, HEIGHT)
    vframe.set_frame_rate(Fraction(FPS, 1))
    vframe.set_fourcc(FourCC.BGRA)
    vframe.set_frame_format(FrameFormat.PROGRESSIVE)
    sender.set_video_frame(vframe)
    sender.open()
    log(f"NDI sender '{NDI_NAME}' ouvert, visible sur le LAN")

    stats_last = time.monotonic()
    stats_count = 0
    bgra_buf = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        # BGR → BGRA in-place (alpha = 255). cv2.cvtColor crée un nouveau
        # array à chaque appel ; on évite en utilisant un buffer réutilisé
        # + assignations directes (plus rapide à 60 fps).
        bgra_buf[:, :, :3] = frame
        bgra_buf[:, :, 3]  = 255
        vframe.write_data(bgra_buf)
        sender.send_video()

        stats_count += 1
        now = time.monotonic()
        if now - stats_last > 5.0:
            dt = now - stats_last
            log(f"{stats_count/dt:.1f} fps envoyés, "
                f"{sender.get_num_connections()} client(s) NDI connecté(s)")
            stats_count = 0
            stats_last = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ndi-sender] SIGINT")
