"""nvjpeg_encoder.py — Encoder JPEG GPU via libnvjpeg + libcuda (ctypes).

Drop-in pour remplacer `cv2.imencode(".jpg", bgr_frame, [JPEG_QUALITY, q])`
quand on veut décharger le CPU. Sur RTX 3080 1080p → ~1.5 ms vs ~10 ms
en cv2 CPU à Q=80.

API minimale :
    enc = NvJpegEncoder(width=1280, height=720, quality=80,
                       sampling="420")
    jpeg_bytes = enc.encode(bgr_frame)  # np.ndarray (h, w, 3) uint8 BGR
    enc.close()

Couvre uniquement les usages utiles ici : BGR interleaved input, 4:2:0 ou
4:2:2 ou 4:4:4 chroma subsampling, qualité entière 1-100. Pas de RGB, pas
de planar, pas de batch. Reuse le CUDA primary context (cohabite avec
onnxruntime CUDA dans face-recog sans conflit de contexte).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import threading
from typing import Optional

import numpy as np


# Symboles libcuda (Driver API). cuInit(0) une fois par process.
_libcuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
_libnvjpeg = ctypes.CDLL("libnvjpeg.so.12", mode=ctypes.RTLD_GLOBAL)


# ─────────────────────── Types ctypes ──────────────────────────
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_void_p
nvjpegStatus_t = ctypes.c_int
nvjpegHandle_t = ctypes.c_void_p
nvjpegEncoderState_t = ctypes.c_void_p
nvjpegEncoderParams_t = ctypes.c_void_p


# Enums nvjpeg
NVJPEG_INPUT_BGR     = 4   # planar BGR (chaque canal séparé)
NVJPEG_INPUT_BGRI    = 6   # BGR interleaved (ce qu'on a en numpy)
NVJPEG_CSS_444       = 0
NVJPEG_CSS_422       = 1
NVJPEG_CSS_420       = 2
NVJPEG_CSS_GRAY      = 6

_SAMPLING_MAP = {
    "444": NVJPEG_CSS_444,
    "422": NVJPEG_CSS_422,
    "420": NVJPEG_CSS_420,
    "gray": NVJPEG_CSS_GRAY,
}


class nvjpegImage_t(ctypes.Structure):
    """Image NVJPEG : jusqu'à 4 plans de pixels (channel[i]) + pitch en
    bytes par plan. Pour BGR interleaved, seul channel[0] est utilisé,
    pitch[0] = width * 3."""
    NVJPEG_MAX_COMPONENT = 4
    _fields_ = [
        ("channel", ctypes.c_void_p * NVJPEG_MAX_COMPONENT),
        ("pitch",   ctypes.c_size_t * NVJPEG_MAX_COMPONENT),
    ]


# ─────────────────────── Bindings ctypes ────────────────────────
def _bind(lib, name, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


cuInit                       = _bind(_libcuda, "cuInit",
                                     CUresult, [ctypes.c_uint])
cuDeviceGet                  = _bind(_libcuda, "cuDeviceGet",
                                     CUresult, [ctypes.POINTER(CUdevice),
                                                 ctypes.c_int])
cuDevicePrimaryCtxRetain     = _bind(_libcuda, "cuDevicePrimaryCtxRetain",
                                     CUresult, [ctypes.POINTER(CUcontext),
                                                 CUdevice])
cuDevicePrimaryCtxRelease    = _bind(_libcuda, "cuDevicePrimaryCtxRelease",
                                     CUresult, [CUdevice])
cuCtxSetCurrent              = _bind(_libcuda, "cuCtxSetCurrent",
                                     CUresult, [CUcontext])
cuStreamCreate               = _bind(_libcuda, "cuStreamCreate",
                                     CUresult, [ctypes.POINTER(CUstream),
                                                 ctypes.c_uint])
cuStreamDestroy              = _bind(_libcuda, "cuStreamDestroy_v2",
                                     CUresult, [CUstream])
cuStreamSynchronize          = _bind(_libcuda, "cuStreamSynchronize",
                                     CUresult, [CUstream])
cuMemAlloc                   = _bind(_libcuda, "cuMemAlloc_v2",
                                     CUresult, [ctypes.POINTER(CUdeviceptr),
                                                 ctypes.c_size_t])
cuMemFree                    = _bind(_libcuda, "cuMemFree_v2",
                                     CUresult, [CUdeviceptr])
cuMemcpyHtoDAsync            = _bind(_libcuda, "cuMemcpyHtoDAsync_v2",
                                     CUresult, [CUdeviceptr,
                                                 ctypes.c_void_p,
                                                 ctypes.c_size_t,
                                                 CUstream])

nvjpegCreateSimple           = _bind(_libnvjpeg, "nvjpegCreateSimple",
                                     nvjpegStatus_t,
                                     [ctypes.POINTER(nvjpegHandle_t)])
nvjpegDestroy                = _bind(_libnvjpeg, "nvjpegDestroy",
                                     nvjpegStatus_t, [nvjpegHandle_t])
nvjpegEncoderStateCreate     = _bind(_libnvjpeg, "nvjpegEncoderStateCreate",
                                     nvjpegStatus_t,
                                     [nvjpegHandle_t,
                                      ctypes.POINTER(nvjpegEncoderState_t),
                                      CUstream])
nvjpegEncoderStateDestroy    = _bind(_libnvjpeg, "nvjpegEncoderStateDestroy",
                                     nvjpegStatus_t, [nvjpegEncoderState_t])
nvjpegEncoderParamsCreate    = _bind(_libnvjpeg, "nvjpegEncoderParamsCreate",
                                     nvjpegStatus_t,
                                     [nvjpegHandle_t,
                                      ctypes.POINTER(nvjpegEncoderParams_t),
                                      CUstream])
nvjpegEncoderParamsDestroy   = _bind(_libnvjpeg, "nvjpegEncoderParamsDestroy",
                                     nvjpegStatus_t, [nvjpegEncoderParams_t])
nvjpegEncoderParamsSetQuality = _bind(_libnvjpeg,
                                       "nvjpegEncoderParamsSetQuality",
                                       nvjpegStatus_t,
                                       [nvjpegEncoderParams_t,
                                        ctypes.c_int, CUstream])
nvjpegEncoderParamsSetSamplingFactors = _bind(
    _libnvjpeg, "nvjpegEncoderParamsSetSamplingFactors",
    nvjpegStatus_t,
    [nvjpegEncoderParams_t, ctypes.c_int, CUstream],
)
nvjpegEncodeImage            = _bind(_libnvjpeg, "nvjpegEncodeImage",
                                     nvjpegStatus_t,
                                     [nvjpegHandle_t,
                                      nvjpegEncoderState_t,
                                      nvjpegEncoderParams_t,
                                      ctypes.POINTER(nvjpegImage_t),
                                      ctypes.c_int,  # input_format
                                      ctypes.c_int,  # image_width
                                      ctypes.c_int,  # image_height
                                      CUstream])
nvjpegEncodeRetrieveBitstream = _bind(
    _libnvjpeg, "nvjpegEncodeRetrieveBitstream",
    nvjpegStatus_t,
    [nvjpegHandle_t, nvjpegEncoderState_t,
     ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), CUstream],
)


# ─────────────────────── Helpers ────────────────────────
def _check_cu(rc: int, where: str) -> None:
    if rc != 0:
        raise RuntimeError(f"CUDA error {rc} in {where}")


def _check_nvj(rc: int, where: str) -> None:
    if rc != 0:
        raise RuntimeError(f"nvjpeg error {rc} in {where}")


# cuInit appelé une seule fois par process, protégé par lock.
_cuda_init_lock = threading.Lock()
_cuda_inited = False


def _ensure_cuda_init() -> None:
    global _cuda_inited
    with _cuda_init_lock:
        if not _cuda_inited:
            _check_cu(cuInit(0), "cuInit")
            _cuda_inited = True


# ─────────────────────── Class principale ────────────────────────
class NvJpegEncoder:
    """Encoder JPEG GPU. Reuse le primary context CUDA du device 0.

    Pré-alloue 1 buffer device pour la frame BGR (size = w*h*3). Encode
    et retrieve sont synchrones côté CUDA stream pour rester drop-in
    avec le call cv2.imencode équivalent.

    Thread-safe au niveau process (lock interne sur encode())."""

    def __init__(self, width: int, height: int, quality: int = 80,
                 sampling: str = "420", device_id: int = 0) -> None:
        if sampling not in _SAMPLING_MAP:
            raise ValueError(f"sampling invalide: {sampling} "
                             f"(attendu 420|422|444|gray)")
        self.width = width
        self.height = height
        self.quality = max(1, min(100, int(quality)))
        self.sampling = _SAMPLING_MAP[sampling]
        self.device_id = device_id

        self._lock = threading.Lock()
        self._closed = False

        _ensure_cuda_init()
        dev = CUdevice()
        _check_cu(cuDeviceGet(ctypes.byref(dev), device_id), "cuDeviceGet")
        self._dev = dev
        ctx = CUcontext()
        _check_cu(cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
                  "cuDevicePrimaryCtxRetain")
        self._ctx = ctx
        _check_cu(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

        stream = CUstream()
        _check_cu(cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")
        self._stream = stream

        handle = nvjpegHandle_t()
        _check_nvj(nvjpegCreateSimple(ctypes.byref(handle)),
                   "nvjpegCreateSimple")
        self._handle = handle

        state = nvjpegEncoderState_t()
        _check_nvj(nvjpegEncoderStateCreate(handle, ctypes.byref(state),
                                              stream),
                   "nvjpegEncoderStateCreate")
        self._state = state

        params = nvjpegEncoderParams_t()
        _check_nvj(nvjpegEncoderParamsCreate(handle, ctypes.byref(params),
                                               stream),
                   "nvjpegEncoderParamsCreate")
        self._params = params
        _check_nvj(nvjpegEncoderParamsSetQuality(params, self.quality,
                                                   stream),
                   "nvjpegEncoderParamsSetQuality")
        _check_nvj(nvjpegEncoderParamsSetSamplingFactors(
            params, self.sampling, stream),
                   "nvjpegEncoderParamsSetSamplingFactors")

        # Device buffer pré-alloué pour la frame BGR interleaved.
        size = width * height * 3
        dptr = CUdeviceptr()
        _check_cu(cuMemAlloc(ctypes.byref(dptr), size), "cuMemAlloc")
        self._dptr = dptr
        self._frame_size = size

        # Buffer host pour bitstream (max théorique = w*h*3 mais en
        # pratique 1/5 → 1/10). On alloue généreux pour rester safe.
        self._bitstream_buf_size = max(64 * 1024, width * height)
        self._bitstream_buf = ctypes.create_string_buffer(
            self._bitstream_buf_size,
        )

    def encode(self, bgr_frame: np.ndarray) -> bytes:
        """Encode une frame BGR (h, w, 3) uint8 en JPEG. Retourne les
        bytes prêts à pousser dans le HTTP multipart."""
        if self._closed:
            raise RuntimeError("NvJpegEncoder déjà fermé")
        h, w = bgr_frame.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(
                f"taille frame inattendue {w}x{h} "
                f"(attendu {self.width}x{self.height})"
            )
        if bgr_frame.dtype != np.uint8 or bgr_frame.ndim != 3 \
                or bgr_frame.shape[2] != 3:
            raise ValueError(
                f"frame doit être (h, w, 3) uint8 BGR, "
                f"reçu {bgr_frame.shape} {bgr_frame.dtype}"
            )
        if not bgr_frame.flags["C_CONTIGUOUS"]:
            bgr_frame = np.ascontiguousarray(bgr_frame)

        with self._lock:
            _check_cu(cuCtxSetCurrent(self._ctx), "cuCtxSetCurrent")
            # 1. Upload H→D.
            host_ptr = bgr_frame.ctypes.data
            _check_cu(
                cuMemcpyHtoDAsync(self._dptr, host_ptr,
                                   self._frame_size, self._stream),
                "cuMemcpyHtoDAsync",
            )

            # 2. Set up nvjpegImage_t pointant sur le buffer device.
            img = nvjpegImage_t()
            img.channel[0] = self._dptr
            img.pitch[0]   = w * 3
            for i in range(1, 4):
                img.channel[i] = None
                img.pitch[i]   = 0

            # 3. Encode (BGR interleaved).
            _check_nvj(
                nvjpegEncodeImage(self._handle, self._state, self._params,
                                   ctypes.byref(img),
                                   NVJPEG_INPUT_BGRI, w, h, self._stream),
                "nvjpegEncodeImage",
            )

            # 4. Récupère la taille du bitstream produit (length OUT).
            length = ctypes.c_size_t(0)
            _check_nvj(
                nvjpegEncodeRetrieveBitstream(
                    self._handle, self._state, None,
                    ctypes.byref(length), self._stream,
                ),
                "nvjpegEncodeRetrieveBitstream (probe)",
            )
            if length.value > self._bitstream_buf_size:
                # Grow buffer (rare, mais possible avec quality élevée).
                self._bitstream_buf_size = length.value * 2
                self._bitstream_buf = ctypes.create_string_buffer(
                    self._bitstream_buf_size,
                )

            # 5. Récupère le bitstream pour de vrai.
            _check_nvj(
                nvjpegEncodeRetrieveBitstream(
                    self._handle, self._state, self._bitstream_buf,
                    ctypes.byref(length), self._stream,
                ),
                "nvjpegEncodeRetrieveBitstream",
            )
            _check_cu(cuStreamSynchronize(self._stream),
                      "cuStreamSynchronize")
            return self._bitstream_buf.raw[: length.value]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            cuStreamSynchronize(self._stream)
        except Exception:
            pass
        try:
            nvjpegEncoderParamsDestroy(self._params)
            nvjpegEncoderStateDestroy(self._state)
            nvjpegDestroy(self._handle)
            cuMemFree(self._dptr)
            cuStreamDestroy(self._stream)
            cuDevicePrimaryCtxRelease(self._dev)
        except Exception:
            pass

    def __del__(self):
        self.close()


if __name__ == "__main__":
    # Test standalone : encode 100 frames synthétiques, mesure latence.
    import time
    enc = NvJpegEncoder(1280, 720, quality=80, sampling="420")
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    # Warmup.
    for _ in range(5):
        _ = enc.encode(frame)
    t0 = time.perf_counter()
    N = 100
    last = b""
    for _ in range(N):
        last = enc.encode(frame)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"NvJpeg encode {N}× 720p Q=80 : "
          f"{elapsed/N:.2f} ms/frame, "
          f"bitstream avg {len(last)/1024:.1f} KB")
    # Sauve un sample pour vérif visuelle.
    with open("/tmp/nvjpeg_sample.jpg", "wb") as f:
        f.write(last)
    print("sample → /tmp/nvjpeg_sample.jpg")
    enc.close()
