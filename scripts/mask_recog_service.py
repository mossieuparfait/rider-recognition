#!/usr/bin/env python3
"""mask_recog_service.py — segmentation broadcast-grade pour face-recog.

Lit les frames RAW BGR via SHM /dev/shm/arbox-frame (publiées par
avtowan-face-recog.service), passe par RVM (Robust Video Matting,
MobileNetV3 FP16) sur cuda:0, et publie le mask alpha uint8 0..255
dans /dev/shm/avtowan-mask. Cohérence temporelle native via les
recurrent states (4 tensors r1-r4) propagés frame à frame.

Le consommateur (face-recog draw_tracks avec MASK_BEHIND_LABELS=1)
utilise le mask pour ré-blender les pixels original au-dessus des
labels où le mask dit "rider" → labels apparaissent derrière les
riders.

Tuning :
  DOWNSAMPLE_RATIO   0.4         pour 1080p ; mettre 0.25 si saturé GPU
  RVM_MODEL          chemin ONNX FP16
  SHM_FRAME_NAME     arbox-frame (publié par face-recog)
  SHM_MASK_NAME      avtowan-mask
  MAX_FPS            60          limite la cadence de lecture
"""

# CUDA bootstrap : preload .so nvidia avant import onnxruntime.
import ctypes, glob, os, sys
_sp = os.path.join(sys.prefix, f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia")
for sub_dir in sorted(glob.glob(_sp + "/*/lib")):
    for so in glob.glob(sub_dir + "/*.so*"):
        try: ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError: pass

import struct
import time
from multiprocessing import shared_memory, resource_tracker

import cv2  # noqa: F401  (importé pour valider que la stack est OK)
import numpy as np
import onnxruntime as ort


def env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None else default


RVM_MODEL = env("RVM_MODEL", "/opt/avtowan-mask-recog/rvm_mobilenetv3_fp16.onnx")
SHM_FRAME_NAME = env("SHM_FRAME_NAME", "arbox-frame")
SHM_MASK_NAME = env("SHM_MASK_NAME", "avtowan-mask")
DOWNSAMPLE_RATIO = float(env("DOWNSAMPLE_RATIO", "0.4"))
MAX_FPS = float(env("MAX_FPS", "60"))
GPU_ID = int(env("GPU_ID", "0"))

# Layout SHM frame (= celui de face-recog SHMFramePublisher) :
#   0  seq u64 (seqlock, impair = write in progress)
#   8  ts_ns u64
#   16 width u32
#   20 height u32
#   24 channels u32 (3 = BGR)
#   28 dtype u32 (0 = uint8)
#   32 raw uint8 BGR
HEADER_FRAME = 32

# Layout SHM mask (1 channel uint8 alpha 0..255) :
#   0  seq u64
#   8  ts_ns u64
#   16 width u32
#   20 height u32
#   24 format u32 (0 = uint8 alpha)
#   28 reserved u32 (0)
#   32 raw uint8 mask
HEADER_MASK = 32
MASK_MAX_W = 1920
MASK_MAX_H = 1080


def log(msg: str) -> None:
    print(f"[mask-recog] {msg}", flush=True)


class FrameReader:
    """Lecteur seqlock du SHM frame publié par face-recog. Renvoie la
    dernière frame stable ou None si rien de nouveau."""

    def __init__(self, name: str):
        self._name = name
        self._shm = None
        self._last_seq = 0

    def _attach(self) -> bool:
        if self._shm is not None:
            return True
        try:
            self._shm = shared_memory.SharedMemory(name=self._name)
        except FileNotFoundError:
            return False
        # Ne PAS auto-unlink (on n'est pas le owner)
        try:
            resource_tracker.unregister(self._shm._name, "shared_memory")
        except Exception:
            pass
        return True

    def read(self):
        if not self._attach():
            return None
        buf = self._shm.buf
        # Seqlock read : 2 lectures du seq, frame valide si pair et égal.
        for _ in range(8):  # retry budget si writer rapide
            seq_a = struct.unpack_from("<Q", buf, 0)[0]
            if seq_a & 1:
                continue  # write in progress
            if seq_a == self._last_seq:
                return None  # rien de nouveau
            ts_ns, w, h, c, dtype = struct.unpack_from("<QIIII", buf, 8)
            if c != 3 or dtype != 0 or w == 0 or h == 0:
                return None
            n = w * h * c
            if HEADER_FRAME + n > len(buf):
                return None
            frame = np.frombuffer(buf, dtype=np.uint8,
                                  count=n, offset=HEADER_FRAME).reshape(h, w, c)
            # Re-check seq pour valider qu'on a lu une frame stable.
            seq_b = struct.unpack_from("<Q", buf, 0)[0]
            if seq_b != seq_a:
                continue
            self._last_seq = seq_a
            # Copie pour ne pas dépendre du SHM buf pendant inference.
            return frame.copy(), ts_ns
        return None


class MaskPublisher:
    """Publisher SHM single-channel uint8 mask, seqlock identique à
    face-recog SHMFramePublisher mais channels implicite=1."""

    def __init__(self, name: str, max_w: int, max_h: int):
        size = HEADER_MASK + max_w * max_h
        try:
            existing = shared_memory.SharedMemory(name=name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        self._shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        try:
            resource_tracker.unregister(self._shm._name, "shared_memory")
        except Exception:
            pass
        self._buf = self._shm.buf
        self._seq = 0
        log(f"SHM publisher '{name}' ouvert : {size} bytes (max {max_w}x{max_h})")

    def publish(self, mask: np.ndarray, ts_ns: int) -> None:
        h, w = mask.shape[:2]
        n = h * w
        if HEADER_MASK + n > len(self._buf):
            return
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if not mask.flags["C_CONTIGUOUS"]:
            mask = np.ascontiguousarray(mask)
        self._seq += 1
        struct.pack_into("<Q", self._buf, 0, self._seq)
        target = np.frombuffer(self._buf, dtype=np.uint8,
                                count=n, offset=HEADER_MASK).reshape(h, w)
        np.copyto(target, mask)
        struct.pack_into("<QIIII", self._buf, 8, ts_ns, w, h, 0, 0)
        self._seq += 1
        struct.pack_into("<Q", self._buf, 0, self._seq)

    def close(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


def main() -> None:
    log(f"démarrage RVM_MODEL={RVM_MODEL} ds={DOWNSAMPLE_RATIO} gpu={GPU_ID}")
    sess = ort.InferenceSession(
        RVM_MODEL,
        providers=[("CUDAExecutionProvider", {"device_id": GPU_ID}),
                   "CPUExecutionProvider"],
    )
    log(f"providers: {sess.get_providers()}")

    reader = FrameReader(SHM_FRAME_NAME)
    pub = MaskPublisher(SHM_MASK_NAME, MASK_MAX_W, MASK_MAX_H)

    # Recurrent states (initiaux : tensors zéro shape [1,1,1,1], RVM les
    # auto-initialise). Propagés à chaque frame pour cohérence temporelle.
    r1 = np.zeros([1, 1, 1, 1], dtype=np.float16)
    r2 = np.zeros([1, 1, 1, 1], dtype=np.float16)
    r3 = np.zeros([1, 1, 1, 1], dtype=np.float16)
    r4 = np.zeros([1, 1, 1, 1], dtype=np.float16)
    ds = np.array([DOWNSAMPLE_RATIO], dtype=np.float32)

    min_period = 1.0 / max(1.0, MAX_FPS)
    last_t = 0.0
    last_stat = time.monotonic()
    n_inf = 0
    sum_ms = 0.0

    log("waiting for SHM frames...")
    while True:
        now = time.monotonic()
        if now - last_t < min_period:
            time.sleep(max(0.0, min_period - (now - last_t)))
            continue
        last_t = now

        got = reader.read()
        if got is None:
            time.sleep(0.005)
            continue
        frame_bgr, ts_ns = got

        # BGR uint8 → RGB float16 NCHW (0..1)
        t0 = time.monotonic()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src = rgb.astype(np.float16) / np.float16(255.0)
        src = src.transpose(2, 0, 1)[None]  # 1,3,H,W

        fgr, pha, r1, r2, r3, r4 = sess.run(
            None,
            {"src": src, "r1i": r1, "r2i": r2, "r3i": r3, "r4i": r4,
             "downsample_ratio": ds},
        )
        # pha : (1,1,H,W) float16 0..1
        mask = (pha[0, 0].astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        infer_ms = (time.monotonic() - t0) * 1000

        pub.publish(mask, ts_ns)
        n_inf += 1
        sum_ms += infer_ms

        if now - last_stat > 5.0:
            log(f"{n_inf / (now - last_stat):.1f} masks/s, "
                f"mean infer={sum_ms / max(1, n_inf):.1f} ms, "
                f"last shape={mask.shape}")
            n_inf = 0
            sum_ms = 0.0
            last_stat = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
