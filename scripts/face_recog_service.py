#!/usr/bin/env python3
"""face_recog_service.py — reconnaissance faciale temps-réel sur live Magewell.

Capture une entrée V4L2 (Magewell Eco SDI typiquement), détecte les visages
avec RetinaFace, extrait un embedding ArcFace par visage, matche contre
l'index pré-calculé (cf index_faces.py), dessine bbox + nom + score en
overlay, et sert le flux annoté en MJPEG sur HTTP.

Pas dans le hot path low-latency : c'est une page monitoring, ~5-10 fps
visible côté browser via <img>. Le service tourne dans son propre process,
GPU CUDA via onnxruntime-gpu, n'interfère pas avec l'encoder broadcast.

Variables d'env (override possibles dans la systemd unit) :
    SOURCE        v4l2:/dev/video0  source vidéo (préfix v4l2:)
    INDEX         /var/lib/avtowan/face-index.npz   embeddings BDD
    HTTP_PORT     8810              port MJPEG
    THRESHOLD     0.5               seuil cosine pour match (0=accept tout)
    DET_SIZE      640               taille image détecteur
    JPEG_QUALITY  80                qualité JPEG (0-100)
    GPU_ID        0                 -1 = CPU fallback
    TARGET_FPS    10                cadence cible MJPEG (downsample)
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket

# ────────────────── Bootstrap libs CUDA depuis le venv ──────────────────
# Les wheels nvidia-* (cuda-runtime, cublas, cudnn, cufft, curand, nvjitlink,
# cuda-nvrtc) installent les .so dans site-packages/nvidia/*/lib/ — pas dans
# un chemin que le loader Linux connaît par défaut. Sans intervention,
# onnxruntime CUDAExecutionProvider échoue à charger libcublasLt.so.12 et
# silently fallback sur CPU (~5× plus lent).
#
# Solution : on dlopen RTLD_GLOBAL chaque .so AVANT d'importer onnxruntime.
# Le runtime trouvera les symboles déjà résidents dans le process.
# Alternative : LD_LIBRARY_PATH dans la systemd unit, mais ce preload rend
# l'invocation directe (python face_recog_service.py) self-contained.
import ctypes
import glob
_nv_glob = os.path.join(os.path.dirname(sys.executable),
                        "..", "lib", "python*", "site-packages", "nvidia",
                        "*", "lib", "lib*.so*")
for _lib in sorted(glob.glob(_nv_glob)):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # symlinks dupliqués, ignore — au moins un des paths réussit

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from filterpy.kalman import KalmanFilter
from PIL import Image, ImageDraw, ImageFont


# ────────────────────────────── Config ──────────────────────────────
def env(name: str, default: str) -> str:
    return os.environ.get(name, default)

SOURCE       = env("SOURCE", "v4l2:/dev/video0")
INDEX        = env("INDEX", "/var/lib/avtowan/face-index.npz")
HTTP_PORT    = int(env("HTTP_PORT", "8810"))
THRESHOLD    = float(env("THRESHOLD", "0.5"))
DET_SIZE     = int(env("DET_SIZE", "640"))
JPEG_QUALITY = int(env("JPEG_QUALITY", "80"))
GPU_ID       = int(env("GPU_ID", "0"))
# Modèle InsightFace : buffalo_l (default, ~280 MB VRAM, rapide, précision
# correcte) ou antelopev2 (~600 MB VRAM, ~3× plus lent mais +30% précision
# sur conditions difficiles : profil, casque, lunettes, peloton lointain).
# Téléchargé auto au 1er run dans ~/.insightface/models/.
INSIGHTFACE_MODEL = env("INSIGHTFACE_MODEL", "buffalo_l")
TARGET_FPS   = float(env("TARGET_FPS", "10"))
# Downscale du frame AVANT encode JPEG (0 = pas de downscale, garde la
# résolution capture). Utile pour publier en 1080p60 sans saturer le
# décodage MJPEG navigateur. La détection garde la résolution native
# (downscale UNIQUEMENT à la publication). Largeur dérivée pour garder le
# ratio source.
PUBLISH_HEIGHT = int(env("PUBLISH_HEIGHT", "0"))
# Chemin du JSON écrit par bib_recog_service (cf rider-recognition).
# Si le fichier n'existe pas / pas frais, on désactive simplement
# l'affichage du bib (zéro régression côté face-only).
PARTANTS_JSON  = env("PARTANTS_JSON", "/var/lib/avtowan/partants.json")
# Nom NDI publié par le sender out (= flux annoté broadcast vers régie).
# Vide = sender NDI désactivé (mode local/dev sans NDI). Sur la box,
# régler à "FaceRecog-Out" ou similaire pour exposer la sortie.
NDI_OUT_NAME   = env("NDI_OUT_NAME", "")

# ─── Snapshot capture (active learning) ────────────────────────────────
# Sauvegarde automatique de crops face dans /var/lib/face-recog/snapshots/
# pour les riders reconnus avec très haute confiance + cross-check OCR
# bib. Pas d'auto-promote vers face-db (cf [[project_face_recog_split]]) —
# review humain via outil séparé.
SNAPSHOT_ENABLE = bool(int(env("SNAPSHOT_ENABLE", "0")))
SNAPSHOTS_DIR   = env("SNAPSHOTS_DIR", "/var/lib/face-recog/snapshots")
# Score cosine doit dépasser THRESHOLD + ce delta pour être candidat.
SNAPSHOT_MIN_SCORE_DELTA = float(env("SNAPSHOT_MIN_SCORE_DELTA", "0.15"))
# Margin top-2 doit dépasser ce seuil (= matching sans ambiguïté).
SNAPSHOT_MIN_MARGIN      = float(env("SNAPSHOT_MIN_MARGIN", "0.15"))
# Si True, requiert que bib_confirmed=True (= 2e modalité indépendante
# OCR confirme la face). C'est le gate le plus fort, recommandé.
SNAPSHOT_REQUIRE_BIB_CONFIRMED = bool(
    int(env("SNAPSHOT_REQUIRE_BIB_CONFIRMED", "1")))
# Laplacian variance min sur le crop pour considérer le face net.
SNAPSHOT_MIN_BLUR_VAR    = float(env("SNAPSHOT_MIN_BLUR_VAR", "150"))
# Taille min du crop face (px sur le plus petit côté).
SNAPSHOT_MIN_SIZE_PX     = int(env("SNAPSHOT_MIN_SIZE_PX", "120"))
# Padding autour de la bbox face avant crop (1.2 = +20% chaque côté).
SNAPSHOT_CROP_PADDING    = float(env("SNAPSHOT_CROP_PADDING", "1.2"))
# Rate limit : minimum N secondes entre 2 snapshots du même rider.
SNAPSHOT_RATE_LIMIT_S    = float(env("SNAPSHOT_RATE_LIMIT_S", "30"))
# Cap global snapshots/heure (anti-runaway sur close-up long).
SNAPSHOT_HOURLY_CAP      = int(env("SNAPSHOT_HOURLY_CAP", "200"))
# Déduplication : si cosine du nouveau embedding ≥ seuil avec un
# snapshot existant du même rider dans la session, skip.
SNAPSHOT_DEDUP_COSINE    = float(env("SNAPSHOT_DEDUP_COSINE", "0.95"))
FACE_DB_DIR    = env("FACE_DB_DIR", "/var/lib/avtowan/face-db")
# Manifest dataset rider-recognition : map UCI ID → {name, ...}. Sert à
# résoudre name → uciid → face-db/<uciid>/ pour les photos, indépendant
# du JSON partants TDF (lui-même limité aux 198 TDF 2024). Couvre les
# ~860 sportifs du dataset complet, ~234 avec photo locale.
RIDER_MANIFEST_JSON = env("RIDER_MANIFEST_JSON",
                          "/var/lib/avtowan/rider_manifest.json")
# Vue "skeleton-only" : fond noir + squelettes body + photos + noms.
# Exposée en parallèle du flux vidéo annoté sur un endpoint séparé
# (/stream-skeleton.mjpeg) — le flux principal /stream.mjpeg reste
# toujours la vidéo annotée, c'est lui que body_recog et bib_recog
# consomment (boucle circulaire évitée si on les coupait via le toggle).
# Le toggle UI bascule juste l'URL de l'<img>, pas le contenu du backend.
# Taille (px côté) de la vignette photo dans le mode skeleton-only.
SKELETON_PHOTO_SIZE = int(env("SKELETON_PHOTO_SIZE", "96"))
BIBS_JSON      = env("BIBS_JSON", "/tmp/avtowan-bibs.json")
# Âge max du JSON bibs au-delà duquel on considère le service bib mort
# et on ignore son contenu (sinon : labels figés avec vieux bibs).
BIBS_MAX_AGE_S = float(env("BIBS_MAX_AGE_S", "3.0"))
# Chemin du JSON écrit par body_recog_service (YOLOv8-pose + BoT-SORT).
BODIES_JSON      = env("BODIES_JSON", "/tmp/avtowan-bodies.json")
BODIES_MAX_AGE_S = float(env("BODIES_MAX_AGE_S", "3.0"))
# Distance max (px) entre position estimée body et dernière position connue
# du face track pour accepter un update. Au-delà = probable bad matching
# track_id, on skip pour éviter les sauts visuels.
BODY_KP_MAX_JUMP_PX = float(env("BODY_KP_MAX_JUMP_PX", "200"))
# Soft-pull du Kalman face vers la position body face_kp pendant les
# frames où le visage est missed. EMA douce (alpha = ~0.2) : pas de
# téléport instantané = pas de flicker, mais le label suit le mouvement
# du rider via le body même quand le visage est caché. 0 = désactive.
BODY_FACE_PULL_ALPHA = float(env("BODY_FACE_PULL_ALPHA", "0.20"))
# Distance max (px) entre Kalman face cx,cy et body face_kp pour accepter
# le soft-pull. Au-delà, l'écart est trop gros pour être une simple perte
# de visage — probablement une mauvaise association body. On laisse
# associate_bodies_to_tracks gérer le drop.
BODY_FACE_PULL_MAX_PX = float(env("BODY_FACE_PULL_MAX_PX", "300"))
# Nb min de frames "missed" avant de commencer à maintenir un face track
# via le body. En dessous, on laisse le Kalman face faire son travail
# normal entre 2 détections. Au-delà = visage perdu, on prend le relais.
BODY_MAINTAIN_MIN_MISSED = int(env("BODY_MAINTAIN_MIN_MISSED", "10"))
# Confiance min sur le keypoint nez/yeux pour utiliser face_kp directement.
# Si en dessous, on tombe sur les fallbacks (épaules → person_bbox).
BODY_KP_MIN_CONF = float(env("BODY_KP_MIN_CONF", "0.3"))
# Toggle affichage bbox des body persons sur le flux annoté (utile pour
# QA / debug visuel). 0 pour désactiver côté broadcast final.
DRAW_BODIES = bool(int(env("DRAW_BODIES", "1")))
# Chemins du depth_recog (DepthAnything publie ici à ~2 fps).
DEPTH_NPY        = env("DEPTH_NPY",  "/tmp/avtowan-depth.npy")
DEPTH_JSON       = env("DEPTH_JSON", "/tmp/avtowan-depth.json")
DEPTH_MAX_AGE_S  = float(env("DEPTH_MAX_AGE_S", "5.0"))
# Bullet time : durée totale de l'effet (secondes), amplitude max du yaw
# (degrés), shift max latéral d'un pixel proche (px sur frame native).
BULLET_TIME_DURATION_S = float(env("BULLET_TIME_DURATION_S", "2.5"))
BULLET_TIME_MAX_YAW    = float(env("BULLET_TIME_MAX_YAW", "20.0"))
BULLET_TIME_MAX_SHIFT  = float(env("BULLET_TIME_MAX_SHIFT", "80.0"))
# Cadence de détection (RetinaFace + ArcFace + matching). Découplée de
# TARGET_FPS pour permettre publish fluide (30 fps) avec détection lente
# (2-5 fps) — la GPU n'est sollicitée que pour les frames de détection.
# Entre 2 détections, les bboxes sont figées sur leur dernière position
# connue (pas d'interpolation Kalman dans cette version simple).
# Défaut = TARGET_FPS pour compat (= comportement avant).
DETECT_FPS   = float(env("DETECT_FPS", str(TARGET_FPS)))
# Filtre détections : skip les visages dont la dimension max (w ou h) est
# < MIN_FACE_PX. Utile pour ignorer le public en arrière-plan (typiquement
# 30-60 px) tout en gardant les sujets premiers plans (100-300 px).
# 0 = pas de filtre.
MIN_FACE_PX  = int(env("MIN_FACE_PX", "0"))
# Quality gate à la query : variance Laplacien minimale du face crop pour
# considérer la détection comme "sharp enough" pour produire un embedding
# fiable. 0 = filtre désactivé (default). Échelles indicatives sur
# Laplacien d'image grayscale :
#   < 30  : très flou (motion blur, défocus marqué)
#   30-80 : flou modéré
#   > 100 : net
# Sweet spot pour cyclisme broadcast = 30-50 (assez lâche pour pas
# perdre les frames de motion modérée, tue les vrais flous indéchiffrables).
FACE_BLUR_MIN_VAR = float(env("FACE_BLUR_MIN_VAR", "0"))

# Partage de la frame raw BGR via shared memory POSIX (/dev/shm). Permet à
# body_recog + bib_recog co-localisés de lire la même frame que face-recog
# vient de décoder du NDI, SANS passer par MJPEG/HTTP (économie ~50% CPU
# total dans la chaîne preview). Body et bib opt-in via env SOURCE_SHM=
# leur côté. 0 = SHM désactivé (legacy compat).
SHM_PUBLISH         = bool(int(env("SHM_PUBLISH", "1")))
SHM_NAME            = env("SHM_NAME", "arbox-frame")
# Resolution max alloctable. Si la frame source dépasse, on skip la
# publish SHM (= safety, ne fait pas crasher). 1080p BGR = 6.2 MB.
SHM_MAX_W           = int(env("SHM_MAX_W", "1920"))
SHM_MAX_H           = int(env("SHM_MAX_H", "1080"))
# Garde uniquement les N plus gros visages par frame (top-N par aire).
# 0 = pas de limite. Filet supplémentaire si MIN_FACE_PX ne suffit pas
# à élaguer (ex: public proche caméra).
MAX_FACES    = int(env("MAX_FACES", "0"))
# Tracker Norfair (Kalman + IoU/euclidean) : maintient track_id + bbox
# interpolée FRAME PAR FRAME même quand DETECT_FPS < TARGET_FPS.
# TRACK_MIN_HITS : nb de détections requises avant confirmation (réduit
# flicker faux positifs). À DETECT_FPS=2 → 1 = confirmation immédiate.
TRACK_MIN_HITS = int(env("TRACK_MIN_HITS", "3"))
# Buffer (publish frames) qu'un track survit sans détection avant d'être
# supprimé. À TARGET_FPS=10 + DETECT_FPS=2 → 10 = 1 sec.
TRACK_BUFFER = int(env("TRACK_BUFFER", "30"))
# Seuil distance euclidienne (en pixels) pour qu'une nouvelle détection
# soit associée à un track existant. Trop bas = nouveau track à chaque
# détection (motion trop ample), trop haut = associations fausses.
# 150-300 px convient à 1080p selon vitesse cible.
TRACK_DISTANCE_THRESHOLD = float(env("TRACK_DISTANCE_THRESHOLD", "200"))
# Seuil cosine similarity pour la ré-identification embedding ArcFace
# (Phase 2 du matching). Si une détection ne match pas par IoU mais que
# son embedding ressemble à un track en "lost" récent (cosine ≥ seuil),
# on restaure le track (même ID + nom préservé). Évite de perdre
# l'identité quand un visage est baissé puis remonte. 0.4 = permissif
# (re-id facile MAIS noms échangés entre personnes), 0.55-0.6 = strict
# (recommandé industrie pour éviter faux re-id qui corrompent les tracks).
TRACK_REID_THRESHOLD = float(env("TRACK_REID_THRESHOLD", "0.55"))

# Multi-frame voting : on attend ce nombre d'embeddings accumulés sur un
# track avant de matcher contre l'index. Match = sur la moyenne
# normalisée des samples (= prototype temporel, dénoise le bruit
# single-shot). À DETECT_FPS=6 → 5 samples ≈ 0.8s avant que le label
# apparaisse, trade-off latence/fiabilité acceptable.
_EMB_SAMPLES_MIN = int(env("EMB_SAMPLES_MIN", "5"))
# Fenêtre glissante max d'embeddings retenus par track. Sert au vote
# initial puis à la ré-id quand l'embedding change. > MIN pour avoir un
# peu de marge sur la robustesse.
_EMB_SAMPLES_MAX = int(env("EMB_SAMPLES_MAX", "10"))
# Margin minimal top - 2e best cosine pour valider un match. Sous ce
# seuil, on est dans la zone "sosies/incertitude" — le top match peut
# être faux. On retarde la résolution plutôt que d'afficher un faux nom.
# Échelle indicative cosine ArcFace 512-d : 0.03 = très ambigu,
# 0.05 = légèrement net, 0.10 = clair, > 0.15 = sans ambiguïté.
_MATCH_MIN_MARGIN = float(env("MATCH_MIN_MARGIN", "0.05"))
# Re-vote périodique : après la résolution initiale, on re-match tous les
# N nouveaux samples accumulés. Permet de corriger un mauvais nom locké
# trop tôt si les samples récents convergent ailleurs. À 5 = re-vote ~2x
# par seconde à DETECT_FPS=6.
_REVOTE_INTERVAL = int(env("REVOTE_INTERVAL", "5"))


def log(msg: str) -> None:
    print(f"[face-recog] {msg}", flush=True)


# ──────────────────────── Loading index ──────────────────────────
class FaceIndex:
    """Wrapper sur le .npz embeddings + reload sur mtime change."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.mtime = 0.0
        self.embeddings = np.zeros((0, 512), dtype=np.float32)
        self.names: list[str] = []
        self.reload_if_changed()

    def reload_if_changed(self) -> bool:
        if not self.path.is_file():
            return False
        m = self.path.stat().st_mtime
        if m <= self.mtime:
            return False
        try:
            data = np.load(self.path, allow_pickle=True)
            self.embeddings = data["embeddings"].astype(np.float32)
            self.names      = list(data["names"])
            self.mtime      = m
            log(f"index chargé : {len(self.names)} sportifs ({self.path})")
            return True
        except Exception as e:
            log(f"index reload échec : {e}")
            return False

    def match(self, emb: np.ndarray) -> tuple[str, float, float]:
        """Retourne (name, top_score, margin) du meilleur match.

        - top_score : cosine du meilleur match (∈ [-1, 1])
        - margin   : top_score - 2e meilleur match. Petit margin = zone
          ambiguë (sosies, mauvaise lumière, profil), le caller peut
          décider de retarder la résolution. 0 si l'index n'a qu'un seul
          rider ou est vide.
        """
        if self.embeddings.shape[0] == 0:
            return ("?", 0.0, 0.0)
        sims = self.embeddings @ emb
        idx = int(np.argmax(sims))
        top = float(sims[idx])
        if self.embeddings.shape[0] >= 2:
            # Trouve le 2e meilleur sans trier tout l'array : on masque le
            # top et on reprend l'argmax.
            sims2 = sims.copy()
            sims2[idx] = -2.0  # < min cosine possible
            second = float(np.max(sims2))
            margin = top - second
        else:
            margin = 0.0
        return (self.names[idx], top, margin)


# ──────────────────────── Bibs JSON (publié par bib_recog_service) ───
class BibsState:
    """Lit périodiquement /tmp/avtowan-bibs.json publié par le service
    bib_recog (venv séparé). Cache + reload sur mtime change.

    JSON attendu :
        {
          "ts": float epoch,
          "frame_w": int, "frame_h": int,
          "bibs": [
            {"bib": int, "name": str|null, "team": str|null,
             "person_bbox": [x1,y1,x2,y2], "bib_bbox": [...],
             "confidence": float},
            ...
          ]
        }

    Si le fichier est trop vieux (> BIBS_MAX_AGE_S) ou manquant, retourne
    une liste vide → face-recog continue à fonctionner sans bibs.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.mtime = 0.0
        self.payload: dict | None = None

    def _refresh(self) -> None:
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            self.payload = None
            return
        if m != self.mtime:
            try:
                self.payload = json.loads(self.path.read_text())
                self.mtime = m
            except (json.JSONDecodeError, OSError):
                self.payload = None

    def _fresh(self) -> bool:
        if self.payload is None:
            return False
        return time.time() - self.payload.get("ts", 0.0) <= BIBS_MAX_AGE_S

    def get(self) -> list[dict]:
        """Retourne la liste de bibs courante (vide si stale/absent)."""
        self._refresh()
        if not self._fresh():
            return []
        return self.payload.get("bibs", [])

    def get_persons(self) -> list[dict]:
        """Retourne la liste de personnes trackées (BoT-SORT) courante."""
        self._refresh()
        if not self._fresh():
            return []
        return self.payload.get("persons", [])


def _face_in_person(t, person_bbox) -> bool:
    """True si le centre de la bbox face est dans le person_bbox."""
    fx1, fy1, fx2, fy2 = t.bbox_xyxy()
    cx = (fx1 + fx2) * 0.5
    cy = (fy1 + fy2) * 0.5
    px1, py1, px2, py2 = person_bbox
    return px1 <= cx <= px2 and py1 <= cy <= py2


# ──────────────────────── Bodies JSON (publié par body_recog_service) ───
class BodiesState:
    """Lit /tmp/avtowan-bodies.json publié par body_recog_service (venv
    séparé). Cache + reload sur mtime. Si stale, retourne []."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.mtime = 0.0
        self.payload: dict | None = None

    def get_persons(self, target_w: int | None = None,
                     target_h: int | None = None) -> list[dict]:
        """Retourne la liste persons courante. Si target_w/h fournis et
        différents du frame_w/h du JSON (cas où body_recog consomme un
        MJPEG downscalé), rescale les coordonnées (bbox, keypoints,
        face_kp) pour matcher la frame native côté face_recog."""
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            return []
        if m != self.mtime:
            try:
                self.payload = json.loads(self.path.read_text())
                self.mtime = m
            except (json.JSONDecodeError, OSError):
                return []
        if self.payload is None:
            return []
        if time.time() - self.payload.get("ts", 0.0) > BODIES_MAX_AGE_S:
            return []
        persons = self.payload.get("persons", [])
        src_w = self.payload.get("frame_w")
        src_h = self.payload.get("frame_h")
        if (target_w is None or target_h is None
                or src_w is None or src_h is None
                or (src_w == target_w and src_h == target_h)):
            return persons
        sx = target_w / src_w
        sy = target_h / src_h
        scaled = []
        for p in persons:
            pb = p["person_bbox"]
            kps = p["keypoints"]
            fk = p.get("face_kp")
            scaled.append({
                **p,
                "person_bbox": [pb[0] * sx, pb[1] * sy,
                                pb[2] * sx, pb[3] * sy],
                "keypoints": [[k[0] * sx, k[1] * sy, k[2]] for k in kps],
                "face_kp": [fk[0] * sx, fk[1] * sy, fk[2]] if fk else None,
            })
        return scaled


# ──────────────────────── Depth + Bullet time ──────────────────────────
class DepthState:
    """Lit /tmp/avtowan-depth.npy + .json publiés par depth_recog_service.
    Cache + reload sur mtime du JSON. Si stale ou absent, retourne None."""

    def __init__(self, npy_path: str, json_path: str) -> None:
        self.npy_path = Path(npy_path)
        self.json_path = Path(json_path)
        self.mtime = 0.0
        self.depth: np.ndarray | None = None
        self.meta: dict | None = None

    def _refresh(self) -> None:
        try:
            m = self.json_path.stat().st_mtime
        except FileNotFoundError:
            self.depth = None
            self.meta = None
            return
        if m == self.mtime:
            return
        try:
            self.meta = json.loads(self.json_path.read_text())
            h, w = self.meta["shape"]
            raw = self.npy_path.read_bytes()
            self.depth = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
            self.mtime = m
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            self.depth = None
            self.meta = None

    def get(self) -> tuple[np.ndarray, dict] | None:
        """Retourne (depth, meta) si frais, sinon None."""
        self._refresh()
        if self.depth is None or self.meta is None:
            return None
        if time.time() - self.meta.get("ts", 0.0) > DEPTH_MAX_AGE_S:
            return None
        return self.depth, self.meta


def warp_3d_photo(frame_bgr: np.ndarray, depth: np.ndarray,
                   yaw_deg: float,
                   max_shift_px: float = BULLET_TIME_MAX_SHIFT) -> np.ndarray:
    """Layer-based parallax warp : chaque pixel shift horizontalement
    proportionnellement à sa depth (proche = grand shift, loin = nul).

    La depth est BLURRÉE (Gaussien) avant warp pour adoucir les bords
    abrupts. Sans ce blur, les pixels juste au bord d'un sujet ont une
    depth intermédiaire à cause du downscale → shift partiel → effet
    "ghosting / dédoublement" très visible.

    depth : uint8 (dh, dw), 255 = proche, 0 = loin (DepthAnything).
            Resized auto si dim ≠ frame.
    yaw_deg : angle caméra virtuel en degrés.
    """
    if abs(yaw_deg) < 0.5:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    # GaussianBlur sur la depth : adoucit les transitions abruptes aux
    # frontières d'objets pour réduire les artifacts de ghosting/dédou-
    # blement. Kernel 31×31 = environ 1% de l'image largeur.
    depth = cv2.GaussianBlur(depth, (31, 31), 0)
    depth_norm = depth.astype(np.float32) / 255.0
    shift_amount = math.sin(math.radians(yaw_deg)) * max_shift_px
    h_shift = depth_norm * shift_amount
    mapx, mapy = np.meshgrid(np.arange(w, dtype=np.float32),
                              np.arange(h, dtype=np.float32))
    mapx_new = mapx - h_shift
    return cv2.remap(frame_bgr, mapx_new, mapy,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


class BulletTimeState:
    """État global trigger bullet-time, partagé entre processing_loop
    (qui update la dernière frame live) et MJPEGHandler (qui reçoit
    POST /bullet-time du client). Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = False
        self.start_ts = 0.0
        self.freeze_frame: np.ndarray | None = None
        self.freeze_depth: np.ndarray | None = None
        # Mis à jour par processing_loop à chaque tick (ref, pas copy).
        self.last_native_frame: np.ndarray | None = None

    def update_live(self, frame: np.ndarray) -> None:
        """processing_loop appelle ça à chaque frame native (ref seule,
        pas de copy → coût négligeable)."""
        with self._lock:
            self.last_native_frame = frame

    def trigger(self, depth: np.ndarray | None) -> bool:
        """Démarre l'effet avec la dernière frame live + une depth.
        Retourne False si depth absente, frame pas encore prête ou
        bullet-time déjà actif."""
        if depth is None:
            return False
        with self._lock:
            if self.active or self.last_native_frame is None:
                return False
            self.active = True
            self.start_ts = time.monotonic()
            self.freeze_frame = self.last_native_frame.copy()
            self.freeze_depth = depth.copy()
        return True

    def get_state(self) -> tuple[bool, float, np.ndarray, np.ndarray]:
        """Retourne (active, elapsed_s, frame, depth). Auto-désactive
        quand la durée est dépassée."""
        with self._lock:
            if not self.active:
                return False, 0.0, None, None
            elapsed = time.monotonic() - self.start_ts
            if elapsed >= BULLET_TIME_DURATION_S:
                self.active = False
                self.freeze_frame = None
                self.freeze_depth = None
                return False, 0.0, None, None
            return True, elapsed, self.freeze_frame, self.freeze_depth


# Instances globales partagées entre processing_loop et MJPEGHandler.
bullet_state = BulletTimeState()
depth_state_global = DepthState(DEPTH_NPY, DEPTH_JSON)


def bullet_time_yaw(elapsed_s: float) -> float:
    """Courbe d'animation du yaw pendant le bullet-time : aller-retour
    sinus complet sur la durée. 0 → +MAX → -MAX → 0."""
    if BULLET_TIME_DURATION_S <= 0:
        return 0.0
    phase = elapsed_s / BULLET_TIME_DURATION_S  # [0, 1]
    return BULLET_TIME_MAX_YAW * math.sin(2.0 * math.pi * phase)


def associate_bodies_to_tracks(tracks, persons: list[dict]) -> None:
    """3 phases par face track :

    A. **Validation géo** d'une association existante. Si le body
       associé a bougé hors de la position du face Kalman (BoT-SORT a
       swappé son track_id entre 2 coureurs proches), on drop l'asso →
       phase B re-cherchera le bon body.

    B. **(Re-)association** si pas de body : on cherche un body dont le
       person_bbox contient le centre Kalman face.

    C. **Soft-pull** du Kalman face cx,cy vers le body face_kp pendant
       les frames missed. EMA douce (BODY_FACE_PULL_ALPHA) — pas de
       teleport, pas de flicker. Effet : le label suit le mouvement du
       rider via le body même quand le visage est caché. Quand le
       visage revient, l'update Kalman normal écrase doucement.
    """
    now_s = time.monotonic()
    persons_by_id = {p["track_id"]: p for p in persons
                     if p.get("track_id") is not None}

    for t in tracks:
        # Phase A : validation de l'association existante.
        if t.body_track_id is not None:
            p_cur = persons_by_id.get(t.body_track_id)
            if (p_cur is not None
                    and not _face_in_person(t, p_cur["person_bbox"])):
                # Le body bearer du track_id n'est plus à la position face.
                # BoT-SORT a probablement swappé → drop, on retentera en B.
                t.body_track_id = None

        # Phase B : (re-)association si pas de body.
        if t.body_track_id is None:
            for p in persons:
                tid = p.get("track_id")
                if tid is None:
                    continue
                if _face_in_person(t, p["person_bbox"]):
                    t.body_track_id = tid
                    t.last_body_maintain_ts = now_s
                    break
            continue

        # Phase C : confirmation présence + offset learning + soft-pull.
        p = persons_by_id.get(t.body_track_id)
        if p is None:
            # Body absent cette frame mais on garde le track_id (recovery
            # possible si BoT-SORT le re-publie au tick suivant).
            continue
        t.last_body_maintain_ts = now_s

        # Position body face_kp (= nez/yeux YOLO-pose) si confidence OK.
        face_kp = p.get("face_kp")
        body_kp_x: float | None = None
        body_kp_y: float | None = None
        if face_kp is not None and face_kp[2] >= 0.3:
            body_kp_x = float(face_kp[0])
            body_kp_y = float(face_kp[1])

        # Apprentissage de l'offset (face Kalman pos − body face_kp).
        # Conditions : visage détecté ce tick (Kalman frais) ET body
        # face_kp confiant. EMA lente (α=0.1) pour ignorer les outliers
        # tout en convergent vers le vrai offset moyen du rider en
        # ~10 détections (≈ 1s à DETECT_FPS=12).
        if t.missed == 0 and body_kp_x is not None:
            cur_off_x = float(t.kf.x[0, 0]) - body_kp_x
            cur_off_y = float(t.kf.x[1, 0]) - body_kp_y
            if t.body_kp_offset_x is None:
                t.body_kp_offset_x = cur_off_x
                t.body_kp_offset_y = cur_off_y
            else:
                t.body_kp_offset_x += (cur_off_x - t.body_kp_offset_x) * 0.1
                t.body_kp_offset_y += (cur_off_y - t.body_kp_offset_y) * 0.1

        # Soft-pull Kalman pendant missed, cible = body_face_kp + offset
        # appris → équivalent à la position face détectée. Plus de saut
        # systématique à l'alternance détection ↔ missed.
        if t.missed > 0 and BODY_FACE_PULL_ALPHA > 0:
            if body_kp_x is not None:
                target_x = body_kp_x
                target_y = body_kp_y
                if t.body_kp_offset_x is not None:
                    target_x += t.body_kp_offset_x
                    target_y += t.body_kp_offset_y
            else:
                # Fallback top du person_bbox si pas de face_kp confiant.
                pb = p["person_bbox"]
                target_x = float((pb[0] + pb[2]) * 0.5)
                target_y = float(pb[1] + (pb[3] - pb[1]) * 0.08)
            cur_cx = float(t.kf.x[0, 0])
            cur_cy = float(t.kf.x[1, 0])
            dx = target_x - cur_cx
            dy = target_y - cur_cy
            if (abs(dx) < BODY_FACE_PULL_MAX_PX
                    and abs(dy) < BODY_FACE_PULL_MAX_PX):
                t.kf.x[0, 0] += dx * BODY_FACE_PULL_ALPHA
                t.kf.x[1, 0] += dy * BODY_FACE_PULL_ALPHA


# COCO 17 keypoints — paires d'indices à relier pour le squelette.
_SKELETON_EDGES = [
    (0, 1), (0, 2),         # nez ↔ yeux
    (1, 3), (2, 4),         # yeux ↔ oreilles
    (5, 6),                 # épaules
    (5, 7), (7, 9),         # bras gauche
    (6, 8), (8, 10),        # bras droit
    (5, 11), (6, 12),       # épaules ↔ hanches (torse)
    (11, 12),               # hanches
    (11, 13), (13, 15),     # jambe gauche
    (12, 14), (14, 16),     # jambe droite
]
_KP_VIS_CONF = 0.3  # confiance min pour afficher un keypoint


import math


def _orbital_yaw_deg(now_s: float, phase_s: float) -> float:
    """Yaw courant pour l'orbite (sin oscillant entre -AMP et +AMP)."""
    t = (now_s + phase_s) % _ORBIT_PERIOD_S
    return _ORBIT_YAW_AMP_DEG * math.sin(2.0 * math.pi * t / _ORBIT_PERIOD_S)


def _orbital_warp(bgra: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Warp perspective d'un label BGRA pour simuler une rotation yaw 3D
    (caméra virtuelle qui tourne autour du label = effet broadcast pro).

    Modèle : le label est un quad plan Z=0 dans l'espace, caméra à
    distance d sur l'axe Z. On rotate les 4 coins autour de l'axe Y
    puis on projette avec une focale simple.
    """
    if abs(yaw_deg) < 0.5:
        return bgra  # transformation négligeable

    h, w = bgra.shape[:2]
    half_w = w * 0.5
    half_h = h * 0.5

    # 4 coins dans l'espace 3D, origine centre label.
    pts3d = np.array([
        [-half_w, -half_h, 0.0],
        [ half_w, -half_h, 0.0],
        [ half_w,  half_h, 0.0],
        [-half_w,  half_h, 0.0],
    ], dtype=np.float32)

    # Rotation autour de Y (yaw).
    a = math.radians(yaw_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    Ry = np.array([
        [cos_a, 0.0, sin_a],
        [0.0,   1.0, 0.0  ],
        [-sin_a, 0.0, cos_a],
    ], dtype=np.float32)
    rotated = pts3d @ Ry.T

    # Projection perspective simple : f = focale virtuelle, on place le
    # plan label à Z=0 et la caméra à Z=-f. Le point projeté = x*f/(f+z).
    f = w * 2.0
    pts2d_dst = []
    for p in rotated:
        denom = f + p[2]
        if denom <= 1e-3:
            return bgra  # dégénéré
        pts2d_dst.append([
            p[0] * f / denom + half_w,
            p[1] * f / denom + half_h,
        ])
    pts2d_dst = np.array(pts2d_dst, dtype=np.float32)

    # Bbox de sortie + translation pour avoir des coords positives.
    x_min, y_min = pts2d_dst.min(axis=0)
    x_max, y_max = pts2d_dst.max(axis=0)
    out_w = max(1, int(math.ceil(x_max - x_min)))
    out_h = max(1, int(math.ceil(y_max - y_min)))
    pts2d_dst_shifted = pts2d_dst - [x_min, y_min]

    pts2d_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts2d_src, pts2d_dst_shifted)
    return cv2.warpPerspective(
        bgra, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def draw_bodies(frame: np.ndarray, persons: list[dict]) -> None:
    """Trace le squelette COCO 17 keypoints + jointures + track_id de
    chaque body person détectée. Style fil de fer cyan, pour QA/debug."""
    color = (200, 180, 80)  # BGR cyan/turquoise
    for p in persons:
        kps = p["keypoints"]
        # Arêtes du squelette (lignes entre keypoints connectés).
        for a, b in _SKELETON_EDGES:
            ka, kb = kps[a], kps[b]
            if ka[2] < _KP_VIS_CONF or kb[2] < _KP_VIS_CONF:
                continue
            cv2.line(frame,
                     (int(ka[0]), int(ka[1])),
                     (int(kb[0]), int(kb[1])),
                     color, 2, cv2.LINE_AA)
        # Points aux jointures (kp visibles).
        for kp in kps:
            if kp[2] >= _KP_VIS_CONF:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3,
                           color, -1, cv2.LINE_AA)
        # Track ID près du nez (ou rien si nez pas détecté).
        tid = p.get("track_id")
        if tid is not None and kps[0][2] >= _KP_VIS_CONF:
            cv2.putText(frame, f"#{tid}",
                        (int(kps[0][0]) + 6, int(kps[0][1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)


def load_rider_photo(uciid: str, size: int) -> np.ndarray | None:
    """Charge la 1ère photo du dossier face-db/<uciid>/ et resize au
    carré `size`. Retourne BGR uint8 ou None si pas de photo trouvée.

    Pas de detection face / pas de crop intelligent : on prend l'image
    brute (assume que les photos face-db sont déjà des portraits cadrés).
    """
    if not uciid:
        return None
    folder = os.path.join(FACE_DB_DIR, uciid)
    if not os.path.isdir(folder):
        return None
    # Cherche n'importe quelle image — extensions courantes.
    for ext in ("jpg", "jpeg", "png", "webp", "JPG", "PNG"):
        matches = glob.glob(os.path.join(folder, f"*.{ext}"))
        if matches:
            try:
                img = cv2.imread(matches[0], cv2.IMREAD_COLOR)
                if img is None:
                    continue
                # Square crop centré puis resize.
                h, w = img.shape[:2]
                s = min(h, w)
                y0 = (h - s) // 2
                x0 = (w - s) // 2
                img = img[y0:y0 + s, x0:x0 + s]
                img = cv2.resize(img, (size, size),
                                 interpolation=cv2.INTER_AREA)
                return img
            except Exception:
                continue
    return None


_FONT_SKEL_NAME = ImageFont.truetype(_FONT_PATH, 18) if False else None  # placeholder, lazy


def draw_skeleton_view(canvas: np.ndarray, persons: list[dict],
                       tracks) -> None:
    """Compose le rendu mode skeleton-only sur `canvas` (déjà rempli de
    noir) : squelettes COCO + vignette photo + nom court par track
    résolu et associé à un body.

    Layout par rider : photo carrée au-dessus du nez, nom court (juste
    `t.name`) immédiatement sous la photo.
    """
    # Squelettes en premier (couche du fond).
    draw_bodies(canvas, persons)

    if not tracks:
        return

    # Index body_track_id → person pour positionner les vignettes.
    persons_by_id = {p["track_id"]: p for p in persons
                     if p.get("track_id") is not None}

    fh, fw = canvas.shape[:2]
    photo_size = SKELETON_PHOTO_SIZE
    name_h = 24  # hauteur de la bande nom sous la photo

    for t in tracks:
        if not t.name_resolved or not isinstance(t.photo_thumb, np.ndarray):
            continue
        p = persons_by_id.get(t.body_track_id) if t.body_track_id else None
        if p is None:
            continue
        kps = p.get("keypoints")
        if not kps:
            continue
        # Position de référence = nez (kp 0), fallback centre épaules.
        if kps[0][2] >= _KP_VIS_CONF:
            ax = int(kps[0][0])
            ay = int(kps[0][1])
        else:
            ls, rs = kps[5], kps[6]
            if ls[2] < _KP_VIS_CONF or rs[2] < _KP_VIS_CONF:
                continue
            ax = int((ls[0] + rs[0]) * 0.5)
            ay = int((ls[1] + rs[1]) * 0.5) - 30
        # Coin haut-gauche de la photo : centrée X sur ax, juste au-dessus
        # de ay avec marge.
        gap_above = 14
        px = ax - photo_size // 2
        py = ay - photo_size - gap_above - name_h
        # Clip au cadre (sinon la photo serait coupée hors frame).
        if py < 4:
            # Pas la place au-dessus → place sous la tête.
            py = ay + gap_above
        px = max(4, min(fw - photo_size - 4, px))
        py = max(4, min(fh - photo_size - name_h - 4, py))

        # Composite photo (BGR direct, pas d'alpha sur les vignettes).
        canvas[py:py + photo_size, px:px + photo_size] = t.photo_thumb

        # Bordure fine blanche autour pour détacher du fond.
        cv2.rectangle(canvas,
                      (px - 1, py - 1),
                      (px + photo_size, py + photo_size),
                      (255, 255, 255), 1, cv2.LINE_AA)

        # Nom court sous la photo, fond sombre + texte blanc.
        name_y0 = py + photo_size + 2
        name_y1 = name_y0 + name_h
        name_x0 = px
        name_x1 = px + photo_size
        cv2.rectangle(canvas,
                      (name_x0, name_y0),
                      (name_x1, name_y1),
                      (15, 18, 26), -1)
        # Texte centré (utilisation OpenCV simple, pas PIL pour rester
        # léger sur le hot path).
        text = t.name.replace("_", " ")
        # Auto-réduit la taille si le nom est trop long pour la largeur.
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        while tw > photo_size - 6 and font_scale > 0.32:
            font_scale -= 0.02
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, thickness)
        tx = name_x0 + (photo_size - tw) // 2
        ty = name_y0 + (name_h + th) // 2 - 2
        cv2.putText(canvas, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (235, 235, 240), thickness, cv2.LINE_AA)


def _estimate_face_pos_from_body(p: dict) -> tuple[float, float] | None:
    """Estime la position du visage depuis un body, en cascade :
    1. face_kp (nez/yeux) si conf >= BODY_KP_MIN_CONF
    2. Centre épaules + offset vertical si épaules fiables
    3. Haut du person_bbox + offset si rien d'autre

    Retourne (x, y) ou None si rien d'exploitable.
    """
    # 1. face_kp direct
    face_kp = p.get("face_kp")
    if face_kp is not None and face_kp[2] >= BODY_KP_MIN_CONF:
        return float(face_kp[0]), float(face_kp[1])

    # 2. Estimation depuis les épaules (kp COCO 5 = left, 6 = right).
    # Les épaules restent visibles même quand la tête est baissée.
    kps = p.get("keypoints")
    if kps and len(kps) >= 7:
        ls, rs = kps[5], kps[6]
        if ls[2] >= BODY_KP_MIN_CONF and rs[2] >= BODY_KP_MIN_CONF:
            cx_s = (ls[0] + rs[0]) * 0.5
            cy_s = (ls[1] + rs[1]) * 0.5
            shoulders_w = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
            # Face ~ 0.8 × largeur épaules au-dessus du centre épaules.
            face_x = cx_s
            face_y = cy_s - shoulders_w * 0.8
            return float(face_x), float(face_y)

    # 3. Fallback : haut du person_bbox + 8% pour approx centre tête.
    pb = p.get("person_bbox")
    if pb is not None:
        px1, py1, px2, py2 = pb
        face_x = (px1 + px2) * 0.5
        face_y = py1 + (py2 - py1) * 0.08
        return float(face_x), float(face_y)

    return None


def maintain_tracks_via_face_kp(tracks, persons: list[dict]) -> int:
    """Pour les face tracks dont le visage est perdu (missed > seuil)
    mais dont le body_track_id est encore présent dans `persons` :
    update directement le Kalman du face track vers la position face
    estimée (cascade nez/yeux → épaules → bbox).

    Sanity check : skip si saut > BODY_KP_MAX_JUMP_PX (mauvais matching).

    Retourne le nb de tracks maintenus.
    """
    persons_by_id = {p["track_id"]: p for p in persons
                     if p.get("track_id") is not None}
    n_maintained = 0
    for t in tracks:
        if t.body_track_id is None or t.missed < BODY_MAINTAIN_MIN_MISSED:
            continue
        p = persons_by_id.get(t.body_track_id)
        if p is None:
            continue
        pos = _estimate_face_pos_from_body(p)
        if pos is None:
            continue
        kx, ky = pos
        fx1, fy1, fx2, fy2 = t.bbox_xyxy()
        fw, fh = fx2 - fx1, fy2 - fy1
        last_cx, last_cy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
        if abs(kx - last_cx) > BODY_KP_MAX_JUMP_PX or \
           abs(ky - last_cy) > BODY_KP_MAX_JUMP_PX:
            continue
        new_bbox = np.array(
            [kx - fw / 2, ky - fh / 2, kx + fw / 2, ky + fh / 2],
            dtype=np.float32,
        )
        t.update(new_bbox, t.last_embedding)
        n_maintained += 1
    return n_maintained


def associate_bibs_to_tracks(tracks, bibs: list[dict]) -> bool:
    """Fusion bib OCR ↔ bib face-via-partants.

    Pour chaque bib OCR détecté (publié par bib_recog_service), on cherche
    le face track dont la bbox visage est contenue dans le person_bbox du
    bib. Trois cas, dans cet ordre :

      1. Track sans bib (face hors partants ou non résolue) → on accepte
         le bib OCR (t.bib = bib_ocr, bib_confirmed = False car aucune
         source indépendante ne valide).

      2. Track avec bib déjà set (issu du JSON partants au moment de la
         ré-id face) ET bib OCR == bib track → **CONFIRMÉ** : on marque
         t.bib_confirmed = True. Visuel : un puce avant le numéro dans le
         label (cf render_lower_third).

      3. Track avec bib déjà set ET bib OCR != bib track → conflit, face
         gagne (= partants reste autorité). OCR ignoré. Pas de changement.

    Retourne True si au moins un track a changé de bib ou de statut
    confirmé (= label à re-render côté processing_loop).
    """
    changed = False
    for b in bibs:
        person_bbox = b["person_bbox"]
        bib_val = b["bib"]
        tid = b.get("track_id")
        for t in tracks:
            if _face_in_person(t, person_bbox):
                if t.bib is None:
                    # Cas 1 : track sans bib → OCR fournit.
                    t.bib = bib_val
                    t.bib_name = b.get("name")
                    t.bib_confirmed = False
                    changed = True
                elif t.bib == bib_val:
                    # Cas 2 : match face↔OCR → confirmation.
                    if not t.bib_confirmed:
                        t.bib_confirmed = True
                        changed = True
                else:
                    # Cas 3 : conflit → face (= partants) gagne, OCR ignoré.
                    pass
                # Toujours mémoriser le body track_id si fourni (sert au
                # maintien face via body en cas de perte visage).
                if tid is not None:
                    t.body_track_id = tid
                break
    return changed


def associate_persons_to_tracks(tracks, persons: list[dict]) -> None:
    """Pour chaque face track sans body_track_id encore, tente d'en
    associer un via overlap géométrique. Permet ensuite de maintenir le
    track quand le visage disparaît (cf inject_phantom_detections)."""
    for t in tracks:
        if t.body_track_id is not None:
            continue
        for p in persons:
            if _face_in_person(t, p["person_bbox"]):
                t.body_track_id = p["track_id"]
                break


def phantom_detections_from_persons(
    tracks, persons: list[dict],
) -> tuple[list, list]:
    """Pour les face tracks dont le visage est perdu (missed > 0) mais
    dont le body_track_id est encore présent dans `persons`, injecte une
    "détection face virtuelle" centrée dans le person_bbox. Ça nourrit le
    TrackManager.update() au tick suivant pour qu'il maintienne le track
    actif au lieu de l'éteindre par max_missed.

    Retourne (bboxes, embeddings) à ajouter aux détections réelles.
    Embedding = None (on garde l'ancien stocké dans le track).
    """
    if not persons:
        return [], []
    persons_by_id = {p["track_id"]: p for p in persons}
    fake_bboxes = []
    fake_embeds = []
    for t in tracks:
        if t.body_track_id is None or t.missed == 0:
            continue
        p = persons_by_id.get(t.body_track_id)
        if p is None:
            continue
        # Construit une bbox face plausible : centrée horizontalement,
        # en haut du person_bbox (les visages sont au-dessus du corps).
        px1, py1, px2, py2 = p["person_bbox"]
        pw, ph = px2 - px1, py2 - py1
        # Taille face ≈ 25% de la largeur person, ~30% en haut du corps.
        fw = max(40, pw // 4)
        fh = fw
        cx = (px1 + px2) // 2
        fy1 = py1 + ph // 10  # 10% sous le sommet
        fake_bboxes.append(np.array(
            [cx - fw // 2, fy1, cx + fw // 2, fy1 + fh], dtype=np.float32
        ))
        # Embedding : on réutilise le dernier connu du track. Le tracker
        # va matcher cette bbox au track par proximité (IoU/distance).
        fake_embeds.append(t.last_embedding)
    return fake_bboxes, fake_embeds


# ──────────────────────── Capture source ──────────────────────────
class NDIReceiver:
    """Duck-type cv2.VideoCapture pour un flux NDI HB.

    Découvre les sources NDI sur le LAN via cyndilib.Finder, se connecte
    à la première dont le name contient `source_match` (substring,
    case-insensitive). Frame format BGRX (= BGRA opaque) reçu par
    cyndilib, on convertit en BGR pour rester compat avec le reste du
    pipeline OpenCV / InsightFace.

    Méthodes exposées (subset cv2.VideoCapture utilisé par le code) :
      - isOpened() → bool
      - read() → (ok: bool, frame: np.ndarray | None)
      - release() → None
      - get(prop), set(prop, val) → no-ops (NDI auto-négocie)

    NB : c'est un wrapper synchrone. CaptureWorker tournera comme avant
    en boucle .read() ; cyndilib bloque dans capture_video() jusqu'à la
    prochaine frame disponible (latence ~16ms à 60 fps).
    """

    # Délai max (sec) pour découvrir la source NDI au boot. Sans source,
    # on exit FATAL — le service systemd retentera via Restart=on-failure.
    DISCOVERY_TIMEOUT_S = 30.0

    def __init__(self, source_match: str) -> None:
        # Import différé pour ne pas obliger cyndilib si SOURCE=v4l2:...
        # API confirmée sur cyndilib v0.x (test box 2026-05-28).
        from cyndilib import (  # type: ignore
            Finder, Receiver, RecvBandwidth, RecvColorFormat, VideoFrameSync,
        )

        log(f"NDI : recherche source matching '{source_match}'...")
        finder = Finder()
        finder.open()
        deadline = time.monotonic() + self.DISCOVERY_TIMEOUT_S
        target = None
        match_lower = (source_match or "").lower()
        while time.monotonic() < deadline:
            finder.update_sources()
            for src in finder.iter_sources():
                name = src.name if hasattr(src, "name") else str(src)
                if not match_lower or match_lower in name.lower():
                    target = src
                    break
            if target is not None:
                break
            time.sleep(0.5)

        if target is None:
            sys.exit(
                f"FATAL: source NDI matching '{source_match}' introuvable "
                f"après {self.DISCOVERY_TIMEOUT_S:.0f}s "
                f"(NDI sender démarré ? même LAN sans firewall mDNS ?)"
            )
        log(f"NDI : source trouvée = {target.name}")

        self._receiver = Receiver(
            color_format=RecvColorFormat.BGRX_BGRA,
            bandwidth=RecvBandwidth.highest,
        )
        self._receiver.set_source(target)
        self._vframe = VideoFrameSync()
        # frame_sync = sous-objet sync wrapper du receiver. set_video_frame
        # binde notre vframe pour qu'il soit rempli à chaque capture_video.
        self._receiver.frame_sync.set_video_frame(self._vframe)
        self._finder = finder
        self._target = target
        self._opened = True
        log("NDI receiver connecté, en attente de la 1ère frame...")

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        # Blocking read d'une frame. cyndilib timeout interne ; si la
        # source disparaît, capture_video() retourne sans données et le
        # vframe est vide → on signale fail au CaptureWorker, qui sleep
        # puis retry.
        try:
            self._receiver.frame_sync.capture_video()
        except Exception as e:
            log(f"NDI capture err: {e}")
            return False, None
        arr = self._vframe.get_array()
        if arr is None or arr.size == 0:
            return False, None
        # cyndilib retourne typiquement un buffer plat ; reshape via
        # xres/yres si nécessaire. En BGRX_BGRA = 4 canaux uint8.
        if arr.ndim == 1:
            w = int(self._vframe.xres)
            h = int(self._vframe.yres)
            if w > 0 and h > 0:
                arr = arr.reshape((h, w, 4))
        if arr.ndim == 3 and arr.shape[2] == 4:
            # BGRA → BGR (RetinaFace + cv2 attendent BGR 3-channels).
            return True, cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return True, arr
        return False, None

    def release(self) -> None:
        self._opened = False
        try:
            if hasattr(self._receiver, "close"):
                self._receiver.close()
        except Exception:
            pass
        try:
            if hasattr(self._finder, "close"):
                self._finder.close()
        except Exception:
            pass

    # No-ops pour compat cv2.VideoCapture
    def set(self, prop, val) -> bool:  # noqa: ARG002
        return False

    def get(self, prop) -> float:  # noqa: ARG002
        return 0.0


class SnapshotManager:
    """Capture + staging des snapshots face pour l'active learning.

    Pour chaque rider reconnu avec très haute confiance (cf gates dans
    `should_save`), sauve un JPG du crop face dans
    SNAPSHOTS_DIR/<uciid>/<ts>_score_margin_tag.jpg. RIEN n'est promu
    automatiquement vers face-db — un outil de review séparé (humain in
    the loop) trie les candidats et déplace les bons vers face-db.

    Maintient en mémoire :
      - last_save_ts[uciid] : dernier save par rider (rate limit)
      - session_embeddings[uciid] : embeddings des snapshots déjà save
        cette session, pour la déduplication
      - hourly_window : timestamps des saves dans la dernière heure (cap)
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        try:
            os.makedirs(base_dir, exist_ok=True)
        except OSError as e:
            log(f"snapshot dir création échec ({base_dir}): {e}")
        self.last_save_ts: dict[str, float] = {}
        self.session_embeddings: dict[str, list[np.ndarray]] = {}
        self.hourly_window: list[float] = []
        self.total_saved = 0

    def _prune_hourly(self, now: float) -> None:
        cutoff = now - 3600.0
        # Truncate from the left (chronological).
        i = 0
        for ts in self.hourly_window:
            if ts > cutoff:
                break
            i += 1
        if i:
            self.hourly_window = self.hourly_window[i:]

    def can_save(self, uciid: str, now: float) -> bool:
        if not uciid:
            return False
        last = self.last_save_ts.get(uciid, 0.0)
        if now - last < SNAPSHOT_RATE_LIMIT_S:
            return False
        self._prune_hourly(now)
        if len(self.hourly_window) >= SNAPSHOT_HOURLY_CAP:
            return False
        return True

    def is_dedup(self, uciid: str, embedding: np.ndarray) -> bool:
        prior = self.session_embeddings.get(uciid)
        if not prior:
            return False
        # Cosine = dot product (embeddings normalisés des 2 côtés).
        for e in prior:
            if float(np.dot(embedding, e)) >= SNAPSHOT_DEDUP_COSINE:
                return True
        return False

    def save(self, uciid: str, name: str, crop_bgr: np.ndarray,
             embedding: np.ndarray, score: float, margin: float,
             bib_confirmed: bool, now: float) -> str | None:
        rider_dir = os.path.join(self.base_dir, uciid)
        try:
            os.makedirs(rider_dir, exist_ok=True)
        except OSError as e:
            log(f"snapshot mkdir échec ({rider_dir}): {e}")
            return None
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        tag = "confirmed" if bib_confirmed else "face"
        fname = f"{ts_str}_s{score:.2f}_m{margin:.2f}_{tag}.jpg"
        path = os.path.join(rider_dir, fname)
        try:
            ok = cv2.imwrite(path, crop_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return None
        except Exception as e:
            log(f"snapshot write échec ({path}): {e}")
            return None
        # Bookkeeping.
        self.last_save_ts[uciid] = now
        self.hourly_window.append(now)
        embs = self.session_embeddings.setdefault(uciid, [])
        embs.append(embedding.copy())
        if len(embs) > 50:
            del embs[:-50]
        self.total_saved += 1
        return path


class SHMFramePublisher:
    """Publie chaque frame BGR raw via multiprocessing.shared_memory.

    Layout (little-endian) :
        offset 0  : seq    u64  — incrémenté à chaque cycle write
                                  Impair = écriture en cours (seqlock,
                                  reader doit retry). Pair = stable.
        offset 8  : ts_ns  u64  — timestamp monotonic_ns au moment du write
        offset 16 : width  u32
        offset 20 : height u32
        offset 24 : channels u32 (3 = BGR)
        offset 28 : dtype  u32  (0 = uint8)
        offset 32 : raw BGR uint8 (width * height * 3 bytes)

    Taille totale = 32 + SHM_MAX_W * SHM_MAX_H * 3.

    Pattern seqlock :
      writer : seq |= 1 ; copy frame ; write meta ; seq += 1 (devient pair)
      reader : seq_a ; lit frame ; seq_b ; valid si seq_a == seq_b et pair

    Le reader peut polling le seq pour savoir s'il y a une nouvelle frame
    (seq plus grand que dernier vu).
    """

    HEADER_SIZE = 32

    def __init__(self, name: str, max_w: int, max_h: int) -> None:
        from multiprocessing import shared_memory  # stdlib
        import struct
        self._struct = struct
        size = self.HEADER_SIZE + max_w * max_h * 3
        # Force unlink d'un éventuel résidu d'un précédent run (un service
        # qui crash peut laisser le shm orphelin → SharedMemory(create=True)
        # lèverait FileExistsError).
        try:
            existing = shared_memory.SharedMemory(name=name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        self._shm = shared_memory.SharedMemory(
            create=True, size=size, name=name,
        )
        self._buf = self._shm.buf
        self._seq = 0
        log(f"SHM publisher '{name}' ouvert : {size} bytes "
            f"(max {max_w}x{max_h}x3)")

    def publish(self, frame_bgr: np.ndarray) -> None:
        """Écrit la frame dans le SHM avec seqlock. Skip si la frame
        dépasse la taille allouée (= sécurité, ne crash pas)."""
        h, w = frame_bgr.shape[:2]
        c = frame_bgr.shape[2] if frame_bgr.ndim == 3 else 1
        frame_size = h * w * c
        if self.HEADER_SIZE + frame_size > len(self._buf):
            return  # frame trop grande, skip
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        # Phase 1 : seq impair = "write in progress".
        self._seq += 1  # devient impair (était pair)
        self._struct.pack_into("<Q", self._buf, 0, self._seq)
        # Phase 2 : copy frame + meta. np.copyto sur une view numpy du
        # buffer = 1 memcpy direct (vs frame.tobytes() qui allouait un
        # Python bytes temporaire et faisait 2 memcpys). ~6ms gagnés par
        # frame à 1080p.
        target = np.frombuffer(self._buf, dtype=np.uint8,
                                count=frame_size,
                                offset=self.HEADER_SIZE).reshape(h, w, c)
        np.copyto(target, frame_bgr)
        self._struct.pack_into(
            "<QIIII", self._buf, 8,
            time.monotonic_ns(), w, h, c, 0,  # dtype 0 = uint8
        )
        # Phase 3 : seq pair = "write done".
        self._seq += 1  # redevient pair
        self._struct.pack_into("<Q", self._buf, 0, self._seq)

    def close(self) -> None:
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


class NDIOutSender:
    """Sender NDI HB pour la frame annotée sortie de face-recog.

    Init lazy à la première frame pour caler width/height/fps sur la
    source réelle. Si désactivé (NDI_OUT_NAME vide), aucune instance.

    Pattern identique au ndi_sender.py standalone : VideoSendFrame en
    BGRA progressive, buffer BGRA réutilisé pour limiter les allocs à
    60 fps.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._sender = None
        self._vframe = None
        self._bgra_buf: np.ndarray | None = None
        self._width = 0
        self._height = 0
        self._dropped_size_mismatch = 0

    def _lazy_init(self, height: int, width: int, fps: int) -> None:
        from fractions import Fraction
        from cyndilib import (  # type: ignore
            Sender, VideoSendFrame, FourCC, FrameFormat,
        )
        self._sender = Sender(self.name)
        self._vframe = VideoSendFrame()
        self._vframe.set_resolution(width, height)
        self._vframe.set_frame_rate(Fraction(max(1, int(fps)), 1))
        self._vframe.set_fourcc(FourCC.BGRA)
        self._vframe.set_frame_format(FrameFormat.progressive)
        self._sender.set_video_frame(self._vframe)
        self._sender.open()
        self._bgra_buf = np.empty((height, width, 4), dtype=np.uint8)
        # Alpha plein = la sortie NDI est opaque (BGRA composé sur fond
        # noir si quelqu'un veut key-out, sinon ignoré).
        self._bgra_buf[:, :, 3] = 255
        self._width = width
        self._height = height
        log(f"NDI sender out '{self.name}' ouvert : {width}x{height}@{fps}fps")

    def send(self, bgr_frame: np.ndarray, fps_hint: int) -> None:
        h, w = bgr_frame.shape[:2]
        if self._sender is None:
            self._lazy_init(h, w, fps_hint)
        elif h != self._height or w != self._width:
            # La frame a changé de taille en cours (bullet-time, downscale
            # intermédiaire). Pour v1 on skip plutôt que de réinitialiser
            # le sender (qui ferait un cut côté receiver). À monitorer.
            self._dropped_size_mismatch += 1
            return
        self._bgra_buf[:, :, :3] = bgr_frame
        self._vframe.write_data(self._bgra_buf.reshape(-1))
        self._sender.send_video()

    def num_connections(self) -> int:
        if self._sender is None:
            return 0
        try:
            return int(self._sender.get_num_connections())
        except Exception:
            return 0

    def close(self) -> None:
        if self._sender is not None:
            try:
                self._sender.close()
            except Exception:
                pass


def open_source(spec: str):
    """Ouvre la source vidéo. Supporte :
      - 'v4l2:/dev/videoN'   → cv2.VideoCapture (Magewell, webcam, etc.)
      - 'ndi://<match>'      → NDIReceiver duck-typé (LAN NDI HB stream)

    Le receiver NDI cherche une source dont le name CONTIENT <match>
    (case-insensitive). 'ndi://AVtoWan' match 'AVtoWan-FaceRecog' publié
    par notre ndi_sender. 'ndi://' (vide) prend la première dispo.

    On laisse le format pixel à la négociation du driver (Magewell Eco
    expose YU12 = YUV 4:2:0 planar par défaut ; OpenCV le convertit en
    BGR pour cv2.read()).
    """
    if spec.startswith("v4l2:"):
        dev = spec[len("v4l2:"):]
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            sys.exit(f"FATAL: ouverture {dev} échec (pas de signal SDI ? "
                     f"v4l2-ctl -d {dev} --all pour debug)")
        # Force 1080p 60 fps. OpenCV par défaut négocie souvent 640×480.
        # Échec silencieux si la source SDI ne supporte pas.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS,          60)
        # BUFFERSIZE=1 : driver V4L2 ne garde qu'une frame en queue.
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log(f"source v4l2 {dev} ouverte ({w}x{h} @ {fps:.1f}fps)")
        return cap
    if spec.startswith("ndi://"):
        match = spec[len("ndi://"):]
        return NDIReceiver(match)
    if spec.startswith("udp:"):
        # Source HEVC MPEG-TS UDP — utilisée sur les boxes dédiées
        # (ex : arbox) qui consomment le flux NVENC distant plutôt
        # que de capturer en V4L2 local. CAP_FFMPEG décode HEVC en
        # software via libavcodec (i3-12100 supporte 1080p60).
        port = spec[len("udp:"):]
        uri = (f"udp://0.0.0.0:{port}"
               f"?fifo_size=8192000&overrun_nonfatal=1"
               f"&buffer_size=16777216")
        cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            sys.exit(f"FATAL: ouverture udp:{port} échec")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log(f"source udp:{port} ouverte ({w}x{h} @ {fps:.1f}fps)")
        return cap
    sys.exit(f"FATAL: source non supportée : {spec} "
             f"(attendu v4l2:/dev/..., ndi://<match> ou udp:<port>)")


# ──────────────────────── Frame processing ────────────────────────
class FrameBuffer:
    """Dernier JPEG annoté, accessible depuis les threads MJPEG (read-only).
    Lock court : juste pour swap atomic du buffer bytes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        # Event signal pour réveiller les MJPEG clients en attente d'un
        # nouveau frame (au lieu de polling).
        self.cond = threading.Condition(self.lock)
        # Compteur de clients actifs (= MJPEG handlers en attente). Sert
        # au producteur pour skip le render quand personne ne regarde.
        self.subscribers: int = 0

    def push(self, jpeg: bytes) -> None:
        with self.cond:
            self.jpeg = jpeg
            self.frame_id += 1
            self.cond.notify_all()

    def wait_new(self, last_frame_id: int, timeout: float = 1.0) -> tuple[bytes, int] | None:
        with self.cond:
            if not self.cond.wait_for(lambda: self.frame_id > last_frame_id, timeout=timeout):
                return None
            return (self.jpeg, self.frame_id)


# ──────────────────────── Capture worker (thread) ──────────────────────
class CaptureWorker:
    """Thread dédié à la lecture V4L2. Lit en continu (~60 fps source) et
    publie la dernière frame dans un slot 1-emplacement (drop-oldest).

    Évite que le display thread se fasse "coller" par la queue V4L2 quand
    le throttle TARGET_FPS skip des frames : le display thread sleep
    précisément entre publish, puis prend la dernière frame disponible.
    Sans ce pattern, le display thread vide la queue V4L2 via cap.read()
    bloquant → throttle défait → fps publish irrégulier.
    """

    def __init__(self, cap, shm_pub: "SHMFramePublisher | None" = None):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frames_grabbed = 0
        self._shm_pub = shm_pub
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                          name="face-recog-capture")
        self._thread.start()

    def get_latest(self) -> np.ndarray | None:
        """Renvoie la dernière frame disponible (ou None si pas encore)."""
        with self._lock:
            f = self._frame
        return f

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        log("capture worker thread started")
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            # Publish RAW frame en SHM avant tout traitement → body_recog
            # et bib_recog co-localisés ont la même frame que face-recog
            # va processer, sans passer par MJPEG/HTTP.
            if self._shm_pub is not None:
                try:
                    self._shm_pub.publish(frame)
                except Exception as e:
                    log(f"SHM publish err: {e}")
            with self._lock:
                self._frame = frame
            self._frames_grabbed += 1


# ──────────────────────── Detection worker (thread) ──────────────────────
class DetectionWorker:
    """Thread dédié à la détection lourde (RetinaFace + ArcFace embedding).

    Le main thread (display) submit() une frame récente, le worker la
    consomme à son rythme (DETECT_FPS) et publie les bboxes + embeddings.
    Le main thread fait get_latest() pour récupérer le dernier résultat
    sans bloquer.

    GIL : onnxruntime CUDA libère le GIL pendant l'inférence → la détection
    tourne réellement en parallèle du display loop côté CPU Python.

    Args :
        app          : insightface FaceAnalysis pré-initialisé
        detect_fps   : cadence max de détection (Hz)
        min_face_px  : filtre size (cf MIN_FACE_PX env)
        max_faces    : filtre nombre (cf MAX_FACES env)
    """

    def __init__(self, app: FaceAnalysis, detect_fps: float,
                 min_face_px: int = 0, max_faces: int = 0,
                 blur_min_var: float = 0.0):
        self.app          = app
        self.period       = 1.0 / max(0.1, detect_fps)
        self.min_face_px  = min_face_px
        self.max_faces    = max_faces
        self.blur_min_var = blur_min_var
        # Slot d'entrée : dernière frame submise par le display thread.
        self._in_lock     = threading.Lock()
        self._in_frame: np.ndarray | None = None
        self._in_event    = threading.Event()
        # Slot de sortie : (bboxes_list, embeddings_list, version).
        self._out_lock    = threading.Lock()
        self._out_bboxes: list = []
        self._out_embeds: list = []
        self._out_version = 0
        self._last_consumed_version = 0
        # Telemetry simple.
        self.detections_count = 0
        # Thread.
        self._stop_event  = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True,
                                              name="face-recog-detect")
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Soumet une frame pour détection async (replace l'ancienne si
        worker pas encore consommé — drop-oldest)."""
        with self._in_lock:
            self._in_frame = frame
        self._in_event.set()

    def get_latest(self) -> tuple[list, list, bool]:
        """Retourne (bboxes, embeddings, is_new). is_new=True si ces
        résultats n'ont pas encore été consommés par le main thread."""
        with self._out_lock:
            is_new = self._out_version > self._last_consumed_version
            bboxes = self._out_bboxes
            embeds = self._out_embeds
            self._last_consumed_version = self._out_version
        return bboxes, embeds, is_new

    def stop(self) -> None:
        self._stop_event.set()
        self._in_event.set()  # réveille le wait

    def _run(self) -> None:
        log("detection worker thread started")
        last_run = 0.0
        while not self._stop_event.is_set():
            # Throttle au DETECT_FPS.
            now = time.monotonic()
            if (now - last_run) < self.period:
                # Wait avec timeout = temps restant — réveille tôt si
                # submit() arrive avant.
                remaining = self.period - (now - last_run)
                self._in_event.wait(timeout=remaining)
                self._in_event.clear()
                continue
            # Snapshot input frame.
            with self._in_lock:
                if self._in_frame is None:
                    self._in_event.wait(timeout=0.05)
                    self._in_event.clear()
                    continue
                frame = self._in_frame  # vue partagée, OK car lecture seule
            last_run = time.monotonic()

            # Détection RetinaFace + ArcFace embed (GPU).
            faces = self.app.get(frame)
            if self.min_face_px > 0 and faces:
                faces = [f for f in faces
                         if max(f.bbox[2] - f.bbox[0],
                                f.bbox[3] - f.bbox[1]) >= self.min_face_px]
            # Quality gate blur : drop les detections dont le face crop
            # est trop flou (variance Laplacien sous le seuil). Évite
            # d'alimenter le multi-frame voting avec des embeddings issus
            # de crops indéchiffrables (motion blur peloton lointain,
            # défocus). Coût : 1 Laplacian par face crop, négligeable.
            if self.blur_min_var > 0.0 and faces:
                fh, fw = frame.shape[:2]
                sharp = []
                for f in faces:
                    x1, y1, x2, y2 = (int(v) for v in f.bbox)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(fw, x2); y2 = min(fh, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    gray = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            if crop.ndim == 3 else crop)
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if lap_var >= self.blur_min_var:
                        sharp.append(f)
                faces = sharp
            if self.max_faces > 0 and len(faces) > self.max_faces:
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                          (f.bbox[3] - f.bbox[1]),
                           reverse=True)
                faces = faces[:self.max_faces]
            bboxes = [f.bbox for f in faces]
            embeds = [f.normed_embedding for f in faces]
            self.detections_count += len(faces)
            # Publish.
            with self._out_lock:
                self._out_bboxes  = bboxes
                self._out_embeds  = embeds
                self._out_version += 1


# ──────────────────────── Rendu lower-third PIL ──────────────────────
# Font globale chargée 1× (PIL.ImageFont.truetype est cher à instancier).
# DejaVuSans-Bold est garanti sur Ubuntu (dejavu fonts paquet par défaut).
_FONT_PATH      = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_SIZE      = 26
_FONT           = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
# Police plus petite pour bib + code nationalité (chips secondaires).
_FONT_SMALL_SIZE = 20
_FONT_SMALL     = ImageFont.truetype(_FONT_PATH, _FONT_SMALL_SIZE)


# ─────────────────────── Métadonnées riders (bib + nation) ─────────────
# Map "Firstname LASTNAME" → {"bib": int, "nationality": str3, "team_code":
# str3, "team_name": str}. Alimenté au boot depuis le JSON partants ASO
# pointé par PARTANTS_JSON. Vide si le fichier n'existe pas → labels
# rendus sans bib/nation (fallback gracieux au comportement historique).
_RIDERS_META: dict[str, dict] = {}
# Map "Firstname LASTNAME" → uciid pour tous les sportifs du manifest
# dataset (~860 entrées). Sert UNIQUEMENT au lookup photo. Bib +
# nationalité restent côté _RIDERS_META (partants TDF only).
_NAME_TO_UCIID: dict[str, str] = {}


def load_name_to_uciid(manifest_path: str | None) -> dict[str, str]:
    """Charge le manifest rider-recognition (UCI ID → {name}) et inverse
    en name → UCI ID. Robust aux entrées sans name (skip)."""
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[face-recog] manifest rider: échec lecture: {e}",
              file=sys.stderr, flush=True)
        return {}
    out: dict[str, str] = {}
    for uciid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name and name not in out:
            out[name] = uciid
    return out


def load_riders_meta(json_path: str | None) -> dict[str, dict]:
    """Parse le JSON partants ASO et retourne {name → meta}.

    Format attendu : top-level {teams: [{code, name, riders: [{bib,
    firstname, lastname, nationality, ...}]}]}. La clé du dict de retour
    est `f"{firstname} {lastname}"` (cohérent avec le format des noms
    dans face-index.npz : "Firstname LASTNAME"). On indexe aussi sous
    `lastnameshort` en fallback (utile si le folder face-db ne contient
    pas le nom composé complet).
    """
    if not json_path or not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[face-recog] partants JSON: échec lecture: {e}",
              file=sys.stderr, flush=True)
        return {}
    out: dict[str, dict] = {}
    for team in data.get("teams", []) or []:
        team_code = team.get("code") or ""
        team_name = team.get("name") or ""
        for r in team.get("riders", []) or []:
            firstname = r.get("firstname") or ""
            lastname = r.get("lastname") or ""
            lastnameshort = r.get("lastnameshort") or lastname
            meta = {
                "bib": r.get("bib"),
                "nationality": r.get("nationality"),
                "team_code": team_code,
                "team_name": team_name,
                "uciid": r.get("uciid") or "",
            }
            for key in {f"{firstname} {lastname}",
                        f"{firstname} {lastnameshort}"}:
                if key.strip() and key not in out:
                    out[key] = meta
    return out


def render_lower_third(name: str, score: float,
                       bib: int | None = None,
                       nationality: str | None = None,
                       bib_confirmed: bool = False) -> np.ndarray:
    """Rend un label broadcast lower-third pour un nom donné.

    Layout horizontal, neutre (pas de couleur dépendante du score) :
      [ bib  │  NAME  │  NAT ]

    - bib : entier en texte clair sur le fond sombre, visible uniquement
      si fourni.
    - NAME : texte blanc gras (police principale).
    - NAT : code pays 3 lettres en texte clair, visible uniquement si
      fourni.
    - Séparateurs verticaux fins entre sections présentes.

    Format display name : "Firstname LASTNAME" tel quel (l'index face-db
    stocke déjà sous cette forme). Si underscores présents (cas folders
    legacy), on les convertit en espaces.

    Renvoie un BGRA numpy array (h, w, 4) prêt à composite sur frame BGR.
    """
    display = name.replace("_", " ")

    # Mesures du texte principal (nom).
    bbox_txt = _FONT.getbbox(display)
    text_w = bbox_txt[2] - bbox_txt[0]
    text_h = bbox_txt[3] - bbox_txt[1]

    # Mesures bib + nationalité (police petite). Puce "•" avant le bib
    # uniquement si confirmé par OCR cross-check (signal subtil de
    # match face↔OCR vérifié, pas de couleur).
    bib_prefix = "• " if (bib is not None and bib_confirmed) else ""
    bib_str = f"{bib_prefix}{bib}" if bib is not None else ""
    nat_str = nationality.upper() if nationality else ""
    bib_text_w = (_FONT_SMALL.getbbox(bib_str)[2]
                  - _FONT_SMALL.getbbox(bib_str)[0]) if bib_str else 0
    nat_text_w = (_FONT_SMALL.getbbox(nat_str)[2]
                  - _FONT_SMALL.getbbox(nat_str)[0]) if nat_str else 0

    pad_x          = 16
    pad_y_top      = 8
    pad_y_bottom   = 8
    radius         = 14
    arrow_w        = 12
    arrow_h        = 10
    section_gap    = 12  # espace de chaque côté d'un séparateur
    sep_w          = 1   # largeur du séparateur vertical

    # Composition horizontale (largeurs additionnées des sections présentes).
    sections_w = text_w
    if bib_text_w:
        sections_w += bib_text_w + section_gap + sep_w + section_gap
    if nat_text_w:
        sections_w += section_gap + sep_w + section_gap + nat_text_w

    w = sections_w + 2 * pad_x
    h_label = pad_y_top + text_h + pad_y_bottom
    h = h_label + arrow_h

    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fond principal (ombre + carte).
    draw.rounded_rectangle((1, 2, w - 1, h_label),
                           radius=radius, fill=(0, 0, 0, 60))      # ombre
    draw.rounded_rectangle((0, 0, w - 2, h_label - 1),
                           radius=radius, fill=(15, 18, 26, 140))  # fond

    cursor_x = pad_x
    # Y aligné sur le texte du nom (les chips secondaires sont centrés
    # verticalement sur la même ligne de texte).
    name_text_y = pad_y_top - 2
    small_text_y = pad_y_top + (text_h - _FONT_SMALL_SIZE) // 2 - 1
    sep_top = pad_y_top + 2
    sep_bot = h_label - pad_y_bottom - 2

    # 1. Bib à gauche (texte simple + séparateur après).
    if bib_text_w:
        draw.text(
            (cursor_x, small_text_y),
            bib_str, font=_FONT_SMALL, fill=(235, 235, 240, 255),
        )
        cursor_x += bib_text_w + section_gap
        draw.rectangle(
            (cursor_x, sep_top, cursor_x + sep_w, sep_bot),
            fill=(255, 255, 255, 90),
        )
        cursor_x += sep_w + section_gap

    # 2. Nom (centre).
    draw.text((cursor_x + 1, name_text_y + 1), display, font=_FONT,
              fill=(0, 0, 0, 200))                                # drop shadow
    draw.text((cursor_x,     name_text_y),     display, font=_FONT,
              fill=(255, 255, 255, 255))
    cursor_x += text_w

    # 3. Séparateur + code pays à droite.
    if nat_text_w:
        cursor_x += section_gap
        draw.rectangle(
            (cursor_x, sep_top, cursor_x + sep_w, sep_bot),
            fill=(255, 255, 255, 90),
        )
        cursor_x += sep_w + section_gap
        draw.text(
            (cursor_x, small_text_y),
            nat_str, font=_FONT_SMALL, fill=(220, 220, 230, 240),
        )

    # Flèche fine vers le visage, centrée sous la carte.
    arrow_top = h_label - 1
    cx_arrow = (w - 2) // 2
    draw.polygon([
        (cx_arrow - arrow_w // 2, arrow_top),
        (cx_arrow + arrow_w // 2, arrow_top),
        (cx_arrow,                arrow_top + arrow_h),
    ], fill=(15, 18, 26, 140))

    # `score` reste dans la signature pour compat (peut alimenter une
    # bordure discrète plus tard), mais n'influe plus sur la couleur des
    # éléments visibles.
    _ = score

    arr = np.array(img)
    return arr[:, :, [2, 1, 0, 3]].copy()


def composite_bgra(frame_bgr: np.ndarray, label_bgra: np.ndarray,
                    x: int, y: int) -> None:
    """Alpha-blend in-place du label_bgra sur frame_bgr à position (x, y).
    Clip aux limites du frame."""
    lh, lw = label_bgra.shape[:2]
    fh, fw = frame_bgr.shape[:2]
    # Clip rect destination.
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(fw, x + lw); y1 = min(fh, y + lh)
    if x1 <= x0 or y1 <= y0:
        return
    # Sous-image label correspondante.
    lx0 = x0 - x; ly0 = y0 - y
    lx1 = lx0 + (x1 - x0); ly1 = ly0 + (y1 - y0)
    roi   = frame_bgr[y0:y1, x0:x1]
    label = label_bgra[ly0:ly1, lx0:lx1]
    alpha = label[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    roi[:] = (alpha * label[:, :, :3] + (1.0 - alpha) * roi).astype(np.uint8)


# ──────────────────────── Tracker Kalman + IoU matching ──────────────────
def _iou(a, b) -> float:
    """IoU entre 2 bboxes [x1,y1,x2,y2]. 0 si pas de chevauchement."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


class FaceTrack:
    """Un track unique : Kalman 8D (cx,cy,w,h,vx,vy,vw,vh), bbox lissée +
    métadonnées nom/score/age. Predict appelé sur les frames sans détection,
    update appelé quand une nouvelle détection matche ce track."""

    _next_id = 1

    def __init__(self, bbox_xyxy: np.ndarray, embedding: np.ndarray):
        self.id = FaceTrack._next_id
        FaceTrack._next_id += 1
        # État Kalman : centre x/y, w/h, et leurs vitesses.
        kf = KalmanFilter(dim_x=8, dim_z=4)
        # F : transition à vélocité constante (dt=1 frame).
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.F[i, i + 4] = 1.0
        # H : observation = état[0:4] (cx, cy, w, h directement mesurés).
        kf.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            kf.H[i, i] = 1.0
        # Covariances. Q (process noise) faible sur position/taille, plus
        # forte sur vitesses qui peuvent changer rapidement. R (measurement
        # noise) modéré — RetinaFace est précis mais bbox jitter ~5 px.
        kf.P *= 10.0
        kf.R *= 4.0
        kf.Q[4:, 4:] *= 5.0  # vitesses bougent plus librement
        # Init état avec la 1ère mesure, vitesses à 0.
        x1, y1, x2, y2 = bbox_xyxy
        kf.x[:4, 0] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        self.kf = kf
        self.last_embedding = embedding
        # Multi-frame voting : on accumule les embeddings successifs (un
        # par detection update) jusqu'à _EMB_SAMPLES_MAX. Le match contre
        # l'index se fait sur la MOYENNE normalisée de ces embeddings
        # (= prototype temporel du rider, dénoise les single-shots
        # bruités). name_resolved ne flip qu'après _EMB_SAMPLES_MIN
        # samples ; jusque-là le track existe mais sans label.
        self.embedding_samples: list[np.ndarray] = [embedding]
        # Samples reçus depuis le dernier vote (initial ou re-vote).
        # Incrémenté dans update(), remis à 0 quand on (re-)vote. Sert au
        # re-vote périodique sans dépendre de la longueur de la fenêtre.
        self.samples_since_last_vote: int = 1
        self.name: str = "?"
        self.score: float = 0.0
        self.name_resolved = False  # True après matching ArcFace
        self.age = 0       # nb total frames ce track a vécu
        self.missed = 0    # nb frames consécutives sans détection
        # Cache du label rendu (BGRA numpy). Calculé 1× quand name_resolved
        # passe à True (cf processing_loop), évite de re-rendre à chaque
        # frame (PIL text draw ~5-10ms/appel, prohibitif à 60 fps × N visages).
        self.label_img: np.ndarray | None = None
        # Position d'affichage lissée par EMA dans draw_tracks (None = pas
        # encore initialisée). Indépendant du Kalman bbox : on lisse en plus
        # côté rendu pour absorber les micro-jitters résiduels.
        self.display_x: float | None = None
        self.display_y: float | None = None
        # Dossard associé : rempli en priorité par lookup partants JSON
        # (au moment de la ré-id ArcFace) — pas par bib_recog OCR qui est
        # désactivé actuellement. None = pas de dossard connu (rider hors
        # partants ou JSON non chargé). bib_recog peut encore l'écraser si
        # réactivé un jour.
        self.bib: int | None = None
        self.bib_name: str | None = None
        # Code nationalité 3-letter ISO (ex "DEN", "FRA") issu du JSON
        # partants. None si rider hors liste.
        self.nationality: str | None = None
        # True quand le bib a été confirmé par cross-check OCR ↔ face/partants
        # (associate_bibs_to_tracks cas 2). Visuel : puce avant le numéro
        # dans le label rendu.
        self.bib_confirmed: bool = False
        # ID stable du tracker body BoT-SORT (publié par body_recog).
        # Quand set, sert à 2 choses :
        # 1. TrackManager.cleanup étend max_missed × 4 tant que ce body
        #    est encore vu (last_body_maintain_ts récent) → le track ne
        #    meurt pas pendant que la tête est baissée.
        # 2. draw_tracks utilise la position du body (top du person_bbox)
        #    pour placer le label quand le visage est perdu → label suit
        #    le coureur visuellement, sans toucher au Kalman face (donc
        #    pas de doublons quand l'embedding ré-id au retour du visage).
        self.body_track_id: int | None = None
        # Timestamp monotonic de la dernière fois où on a vu le body
        # associé à ce track (= associate_bodies_to_tracks l'a confirmé).
        # Sert au cleanup étendu et au rendu en mode "maintained".
        self.last_body_maintain_ts: float = 0.0
        # Offset systématique (px) entre le centre bbox face détectée et
        # le body face_kp (nez YOLO-pose) pour CE rider spécifiquement.
        # Appris par EMA lente quand le visage est détecté ET le body
        # face_kp confiant ; appliqué au pull pendant les missed pour
        # que le label ne saute pas à chaque alternance détection ↔ body.
        # None tant que pas appris.
        self.body_kp_offset_x: float | None = None
        self.body_kp_offset_y: float | None = None
        # Vignette photo (BGR, SKELETON_PHOTO_SIZE × SKELETON_PHOTO_SIZE)
        # chargée 1× depuis face-db/<uciid>/ quand le rider est résolu.
        # Utilisée uniquement par le rendu mode skeleton-only.
        # `False` = on a déjà essayé et il n'y a pas de photo (cache neg).
        self.photo_thumb: np.ndarray | None | bool = None
        # Flip latché : label au-dessus (False) ou en-dessous (True) du
        # visage. Décide UNIQUEMENT quand la condition opposée est tenue
        # sur _LABEL_FLIP_HYST_FRAMES consécutives, pour éviter les
        # bascules nerveuses quand le visage est proche du bord haut.
        self.label_flipped: bool = False
        self.flip_change_countdown: int = 0
        # Phase d'orbite (sec) pour le pseudo-3D : aléatoire à l'init pour
        # que chaque label orbite avec un décalage temporel différent
        # (sinon tous en sync = visuellement faux).
        import random
        self.orbit_phase_s: float = random.uniform(0, _ORBIT_PERIOD_S)
        # Niveau d'empilement vertical (0 = au-dessus du visage à la
        # distance naturelle, +1 = 1 cran plus haut, etc.). Latché avec
        # hystérésis : on monte vite quand une collision apparaît, on
        # descend après _LABEL_DETACH_FRAMES frames consécutives sans
        # besoin. X du label reste TOUJOURS centré sur le visage propre,
        # seul Y bouge — chaque label reste lisiblement "sur son visage".
        self.stack_level: int = 0
        self.detach_countdown: int = 0
        # Dernière cible utilisée (avant EMA). Sert au dead-zone : tant que
        # la nouvelle cible reste à <_LABEL_TARGET_DEADZONE_PX de la
        # précédente, on garde la précédente pour ne pas réveiller l'EMA
        # sur du jitter Kalman pur.
        self.last_target_x: float | None = None
        self.last_target_y: float | None = None

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.missed += 1
        # Amortir progressivement la vélocité estimée tant que la face reste
        # missed : sinon la prédiction Kalman runaway dans la direction de
        # la dernière vélocité (bruit ou vrai mouvement) et la bbox glisse
        # hors du cadre après quelques secondes (cas typique : track étendu
        # via body, missed peut atteindre 120 frames = 12s). Au-delà de 2
        # frames missed on coast vers l'arrêt avec un facteur 0.85/frame
        # → vélocité ≈ 5% au bout de 20 frames, label se fige.
        if self.missed > 2:
            self.kf.x[4:8, 0] *= 0.85

    def update(self, bbox_xyxy: np.ndarray, embedding: np.ndarray):
        x1, y1, x2, y2 = bbox_xyxy
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
                     dtype=np.float32)
        self.kf.update(z)
        self.last_embedding = embedding
        self.missed = 0
        # Accumule l'embedding pour le multi-frame voting. Drop le plus
        # ancien quand la fenêtre est pleine (fenêtre glissante).
        self.embedding_samples.append(embedding)
        if len(self.embedding_samples) > _EMB_SAMPLES_MAX:
            self.embedding_samples.pop(0)
        self.samples_since_last_vote += 1

    def bbox_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.kf.x[:4, 0]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                        dtype=np.float32)


class TrackManager:
    """Gère un set de FaceTracks. update() en 2 phases :
        Phase 1 (IoU)    : match les détections aux tracks par bbox Kalman.
        Phase 2 (re-id)  : pour les détections orphelines, cherche un match
                           cosine vs embeddings de tracks (incl. en "lost") →
                           restaure l'identité si visage revient après baisse.
    Drop les tracks après max_missed frames sans détection."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30,
                 reid_threshold: float = 0.45, display_buffer: int = 2):
        self.tracks: dict[int, FaceTrack] = {}
        self.iou_threshold  = iou_threshold
        self.max_missed     = max_missed
        self.reid_threshold = reid_threshold
        # display_buffer = nb max de frames sans détection où un track reste
        # encore affiché à l'écran (bbox Kalman extrapolée). Évite le
        # clignotement quand TARGET_FPS > DETECT_FPS (frames predict-only
        # entre détections). Au-delà, le track reste vivant en interne pour
        # la ré-id mais n'apparaît plus tant qu'une détection ne le récupère.
        self.display_buffer = display_buffer

    def update(self, det_bboxes: list, det_embeddings: list):
        # Predict tous les tracks (avance Kalman d'1 frame).
        for t in self.tracks.values():
            t.predict()

        track_ids = list(self.tracks.keys())
        used_tracks = set()
        used_dets   = set()

        # Phase 1 : matching IoU greedy. Tolère mouvement modéré entre
        # détections (Kalman extrapole le bbox).
        for di, dbb in enumerate(det_bboxes):
            best_iou = self.iou_threshold
            best_tid = None
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                iou = _iou(dbb, self.tracks[tid].bbox_xyxy())
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                self.tracks[best_tid].update(dbb, det_embeddings[di])
                used_tracks.add(best_tid)
                used_dets.add(di)

        # Phase 2 : ré-identification embedding pour les détections
        # orphelines après IoU. On compare cosine similarity vs TOUS les
        # tracks restants (incl. ceux en "lost" depuis quelques frames =
        # visage baissé ou occlusion brève). Si match > seuil → restaure
        # l'identité (même track_id, même nom déjà résolu).
        for di, dbb in enumerate(det_bboxes):
            if di in used_dets:
                continue
            det_emb = det_embeddings[di]
            best_sim = self.reid_threshold
            best_tid = None
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                t = self.tracks[tid]
                if t.last_embedding is None:
                    continue
                # Cosine = dot product (embeddings normalisés des 2 côtés).
                sim = float(np.dot(det_emb, t.last_embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_tid = tid
            if best_tid is not None:
                self.tracks[best_tid].update(dbb, det_embeddings[di])
                used_tracks.add(best_tid)
                used_dets.add(di)

        # Nouveaux tracks pour les détections toujours orphelines.
        for di, dbb in enumerate(det_bboxes):
            if di not in used_dets:
                t = FaceTrack(dbb, det_embeddings[di])
                self.tracks[t.id] = t

        # Drop les tracks trop longtemps sans détection (au-delà du window
        # de ré-id). max_missed plus large que sans ré-id (~30-60 frames =
        # 5-10s à 6 fps), donne du temps à la ré-id de récupérer.
        # EXCEPTION : si un track est "maintained" via body (= un body
        # track_id encore actif dans le JSON pose) on étend sa vie à 4×
        # max_missed. Le visage perdu (casque baissé, profil) peut durer
        # longtemps tant que le coureur est physiquement présent ; quand
        # le visage revient, l'embedding ArcFace ré-id le match.
        # 3 régimes de durée de vie selon l'état body :
        # - body actif (vu < 1s) → max_missed × 4 (coureur présent, visage
        #   caché)
        # - body PERDU (avait un body mais plus vu > 1s) → kill rapide
        #   (0.5s) : le coureur est sorti du champ, pas la peine de garder
        #   un label fantôme pendant 6s
        # - jamais de body associé → max_missed standard
        now_s = time.monotonic()
        stale = []
        for tid, t in self.tracks.items():
            had_body = getattr(t, "last_body_maintain_ts", 0.0) > 0.0
            body_recent = had_body and now_s - t.last_body_maintain_ts < 0.3
            if body_recent:
                mm = self.max_missed * 4
            elif had_body:
                mm = min(self.max_missed, 15)  # ~0.25s à 60fps
            else:
                mm = self.max_missed
            if t.missed > mm:
                stale.append(tid)
        for tid in stale:
            del self.tracks[tid]

    def active(self) -> list:
        """Tracks affichables : missed <= display_buffer. Couvre les frames
        predict-only entre 2 détections (bbox Kalman extrapolée). Au-delà,
        track invisible à l'écran mais survit en interne (re-id, max_missed).

        EXCEPTION : un track "maintained via body" (last_body_maintain_ts
        récent) reste affichable même missed > display_buffer — sa
        position visuelle viendra du body, pas du Kalman face."""
        now_s = time.monotonic()
        out = []
        for t in self.tracks.values():
            if t.missed <= self.display_buffer:
                out.append(t)
            elif (getattr(t, "last_body_maintain_ts", 0.0) > 0.0
                    and now_s - t.last_body_maintain_ts < 0.3):
                out.append(t)
        return out


# Facteur EMA sur la position d'affichage du label. Plus haut = label
# suit plus vite (moins d'inertie), plus bas = plus lisse (absorbe les
# micro-jitters). À 60 fps publish :
#   0.03 → ~1.5s rattrapage : très tassé, beaucoup d'inertie visible
#   0.15 → ~0.2s rattrapage : réactif, micro-jitters absorbés par la
#          dead-zone (cf _LABEL_TARGET_DEADZONE_PX)
#   0.30 → ~0.1s rattrapage : ultra-réactif, risque de tremblement.
_LABEL_SMOOTH_ALPHA = float(env("LABEL_SMOOTH_ALPHA", "0.15"))
# Snap immédiat si la cible saute de plus que ce seuil. 500 = on absorbe
# largement les transitions de mode (Kalman face ↔ position body) via
# EMA, snap seulement sur ré-id réelle entre 2 personnes différentes.
_LABEL_SNAP_PX = 500
# Décalage vertical entre 2 labels empilés (gap visuel entre eux).
_LABEL_STACK_GAP = 4
# Max niveaux d'empilement quand des visages alignés se chevauchent (peloton
# cycliste typique). Au-delà on accepte la collision.
_LABEL_STACK_MAX = 6
# Hystérésis de descente de niveau : un label monté en stack attend ce
# nombre de frames consécutives "niveau plus bas serait OK" avant de
# redescendre. À 10 fps = 3s. Évite les ascenseurs frame-à-frame quand
# les visages oscillent autour du seuil de collision.
_LABEL_DETACH_FRAMES = 30
# Marge (px) ajoutée au rect du label lors du test de collision côté
# descente : on n'accepte de descendre que si le niveau plus bas serait
# clair avec cette marge — sinon ça re-monterait à la première oscillation.
_LABEL_ANCHOR_MARGIN_PX = 6
# Dead-zone (px) sur la cible : tant que la nouvelle cible (X et Y) reste
# à moins de ce delta de la cible précédente, on ne change pas la cible
# (= EMA tire vers une position figée). Absorbe les micro-jitters bbox
# Kalman et body face_kp qui sinon font onduler le label.
# 24 = tolérance confortable : le label ne se recentre pas sur la bbox
# tant que celle-ci n'a pas vraiment bougé. Tue le sentiment de "trop
# précis" sur le centrage face. Bouge le label seulement sur de vraies
# translations.
_LABEL_TARGET_DEADZONE_PX = int(env("LABEL_TARGET_DEADZONE_PX", "24"))
# Frames consécutives où la condition opposée doit être tenue avant de
# basculer le flip au-dessus ↔ en-dessous. À 60 fps publish = 1s de
# stabilité requise. Évite les flips nerveux quand le visage longe le
# bord haut du frame.
_LABEL_FLIP_HYST_FRAMES = int(env("LABEL_FLIP_HYST_FRAMES", "60"))

# ── Pseudo-3D billboard (titrage 3D qui orbite, effet broadcast pro) ──
# Toggle env var : 0 = label 2D classique.
PSEUDO_3D = bool(int(env("PSEUDO_3D", "1")))
# Période d'un cycle complet d'orbite (sec). Chaque label a sa phase
# aléatoire pour ne pas tous bouger en sync.
_ORBIT_PERIOD_S = float(env("ORBIT_PERIOD_S", "6.0"))
# Amplitude max du yaw (degrés). 15-25 = effet visible, > 40 = caricature.
_ORBIT_YAW_AMP_DEG = float(env("ORBIT_YAW_AMP_DEG", "20.0"))


def _rect_collide(a, b) -> bool:
    """Intersection rectangulaire stricte. a, b = (x0, y0, x1, y1)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_tracks(frame: np.ndarray, tracks,
                bodies: list[dict] | None = None) -> None:
    """Composite le lower-third (nom + flèche) pour chaque track actif.

    Politique de placement :
      - X du label = TOUJOURS centré horizontalement sur le visage propre,
        jamais décalé latéralement. Chaque label reste lisiblement "sur son
        visage" (= identification visuelle immédiate qui est qui).
      - Y = au-dessus du visage à distance naturelle (niveau 0). Si
        chevauchement avec un label déjà placé cette frame, on monte d'un
        cran (`step = lh + _LABEL_STACK_GAP`). On itère jusqu'à trouver un
        niveau libre, ou jusqu'à `_LABEL_STACK_MAX`.
      - Le niveau de stack est **latché par track** : on monte vite à un
        niveau plus haut quand une collision apparaît, mais on n'en
        redescend qu'après `_LABEL_DETACH_FRAMES` frames consécutives où
        le niveau plus bas resterait clair (test avec marge
        `_LABEL_ANCHOR_MARGIN_PX`). Évite les ascenseurs frame-à-frame
        quand 2 visages oscillent autour d'un seuil de collision.
      - Cible final passée par un dead-zone (`_LABEL_TARGET_DEADZONE_PX`) :
        si la nouvelle cible bouge moins que ça par rapport à la précédente,
        on garde la précédente → EMA absorbe les jitters Kalman pour de bon.
    """
    # Le placement du label utilise UNIQUEMENT la bbox Kalman face (qui
    # continue à prédire même sur les frames où la détection a manqué).
    # On n'utilise PAS la position estimée par body_kp ici : ça créait
    # des flickers à chaque transition missed↔détecté quand body et face
    # Kalman ne s'accordaient pas (cas typique : tête tournée, body_kp
    # s'aligne sur l'arrière du crâne, face Kalman garde la dernière
    # position frontale). Le body reste utilisé ailleurs pour étendre la
    # durée de vie du track et l'aide à la ré-id (cf TrackManager), mais
    # n'intervient plus dans la position du label affiché.
    _ = bodies  # paramètre conservé pour compat appelants

    placeable = [t for t in tracks if t.label_img is not None]

    # Dédoublonnage par nom : un même sportif ne doit jamais apparaître
    # deux fois à l'écran. Cause typique : un track fantôme survit après
    # une ré-id sur un autre track, ou 2 faces du dataset matchent le même
    # nom (sosies, embedding bruyant). On garde celui dont la bbox est la
    # plus proche du centre de l'image — c'est lui qui est censé être le
    # "vrai" sujet.
    if placeable:
        fh_frame, fw_frame = frame.shape[:2]
        cx_frame = fw_frame * 0.5
        cy_frame = fh_frame * 0.5

        def _dist_sq_to_center(tr) -> float:
            bx1, by1, bx2, by2 = tr.bbox_xyxy()
            bcx = (bx1 + bx2) * 0.5
            bcy = (by1 + by2) * 0.5
            return (bcx - cx_frame) ** 2 + (bcy - cy_frame) ** 2

        best_by_name: dict[str, "Track"] = {}
        for tr in placeable:
            prev = best_by_name.get(tr.name)
            if prev is None or _dist_sq_to_center(tr) < _dist_sq_to_center(prev):
                best_by_name[tr.name] = tr
        placeable = list(best_by_name.values())

    # Dédoublonnage spatial : un visage = un seul label, point. Si deux
    # tracks ont des bbox qui s'overlap fortement (IoU ≥ 0.5), ils
    # désignent le même visage physique avec des identités différentes
    # (typiquement : un nouveau track né sur un visage déjà tracké avant
    # que le manager ne fusionne, ou un ghost en cours de cleanup). On
    # garde celui au meilleur score embedding (= la meilleure ré-id), et
    # on retire les autres du rendu cette frame.
    if len(placeable) > 1:
        by_score = sorted(placeable,
                          key=lambda tr: tr.score, reverse=True)
        kept: list = []
        for tr in by_score:
            tr_bbox = tr.bbox_xyxy()
            if any(_iou(tr_bbox, k.bbox_xyxy()) >= 0.5 for k in kept):
                continue
            kept.append(tr)
        placeable = kept

    # Tri par track_id (= ordre d'apparition, STABLE d'une frame à l'autre).
    # Les tracks anciens placent en premier, prennent les niveaux bas ; les
    # nouveaux s'empilent au-dessus.
    placeable.sort(key=lambda t: t.id)

    now_s = time.monotonic()
    # Display rects des tracks déjà placés cette frame, indexés par track id.
    # Sert au test de collision contre la position visible (post-EMA), pas
    # pré-EMA.
    placed_by_id: dict[int, tuple[int, int, int, int]] = {}

    for t in placeable:
        x1, y1, x2, y2 = (int(v) for v in t.bbox_xyxy())

        # Pseudo-3D : oscille en yaw selon la phase d'orbite du track.
        if PSEUDO_3D:
            yaw_deg = _orbital_yaw_deg(now_s, t.orbit_phase_s)
            label_to_draw = _orbital_warp(t.label_img, yaw_deg)
        else:
            label_to_draw = t.label_img
        lh, lw = label_to_draw.shape[:2]

        # Position naturelle (X centré sur le visage, Y au-dessus de la
        # tête à distance fixe ; flip dessous si hors cadre en haut).
        cx = (x1 + x2) // 2
        nat_x = cx - lw // 2
        # Gap vertical entre la flèche du label et le haut/bas du visage.
        # 18 px = breathing room confortable, le label ne mord pas le haut
        # de la tête / casque.
        face_gap = 18

        # Décision flip avec hystérésis latchée par track.
        # - Souhait courant : True (en-dessous) si la position au-dessus
        #   dépasse en haut du frame (y1 < lh + face_gap), sinon False.
        # - On ne change le t.label_flipped persistant qu'après
        #   _LABEL_FLIP_HYST_FRAMES frames consécutives où le souhait
        #   diffère du state courant. Évite les bascules nerveuses quand
        #   le visage est proche du seuil.
        want_flipped = (y1 - lh - face_gap) < 0
        if want_flipped == t.label_flipped:
            t.flip_change_countdown = 0
        else:
            t.flip_change_countdown += 1
            if t.flip_change_countdown >= _LABEL_FLIP_HYST_FRAMES:
                t.label_flipped = want_flipped
                t.flip_change_countdown = 0
        flipped = t.label_flipped
        nat_y = (y2 + face_gap) if flipped else (y1 - lh - face_gap)
        step = lh + _LABEL_STACK_GAP
        step_sign = +1 if flipped else -1  # +1 = descendre, -1 = monter

        # 1. Calcul du needed_level : niveau minimum qui évite la collision
        #    avec les display rects déjà placés cette frame.
        def _ty_at(level: int) -> int:
            return nat_y + step_sign * level * step

        def _collides_at(level: int, margin: int = 0) -> bool:
            ty = _ty_at(level)
            r = (nat_x - margin, ty - margin,
                 nat_x + lw + margin, ty + lh + margin)
            return any(_rect_collide(r, pr) for pr in placed_by_id.values())

        needed_level = 0
        while needed_level <= _LABEL_STACK_MAX and _collides_at(needed_level):
            needed_level += 1
        if needed_level > _LABEL_STACK_MAX:
            # Saturé : on accepte la collision au max level (rare).
            needed_level = _LABEL_STACK_MAX

        # 2. Hystérésis sur le stack_level latché :
        #    - si needed > stack_level → on monte tout de suite.
        #    - si needed == stack_level → niveau OK, reset countdown.
        #    - si needed < stack_level → on teste avec MARGE qu'un niveau
        #      intermédiaire (current_level - 1) serait clair ; si oui,
        #      on incrémente le countdown ; à _LABEL_DETACH_FRAMES on
        #      descend d'un cran. Ainsi la descente est progressive et
        #      vraiment justifiée.
        if needed_level > t.stack_level:
            t.stack_level = needed_level
            t.detach_countdown = 0
        elif needed_level == t.stack_level:
            t.detach_countdown = 0
        else:
            # needed < stack_level : tester avec marge le niveau juste
            # en-dessous du courant. Si ça collide encore avec marge, on
            # reset le countdown (la descente n'est pas vraiment sûre).
            if _collides_at(t.stack_level - 1,
                            margin=_LABEL_ANCHOR_MARGIN_PX):
                t.detach_countdown = 0
            else:
                t.detach_countdown += 1
                if t.detach_countdown >= _LABEL_DETACH_FRAMES:
                    t.stack_level -= 1
                    t.detach_countdown = 0

        # 3. Position cible finale + dead-zone.
        target_x = nat_x
        target_y = _ty_at(t.stack_level)
        if (t.last_target_x is not None
                and abs(target_x - t.last_target_x)
                    < _LABEL_TARGET_DEADZONE_PX
                and abs(target_y - t.last_target_y)
                    < _LABEL_TARGET_DEADZONE_PX):
            # Variation trop faible → on garde la cible précédente, EMA
            # tire vers une position figée plutôt que de courir un jitter.
            target_x = t.last_target_x
            target_y = t.last_target_y
        else:
            t.last_target_x = float(target_x)
            t.last_target_y = float(target_y)

        # 4. EMA vers la cible (snap si saut énorme, e.g. ré-id entre 2
        #    personnes très éloignées).
        if (t.display_x is None
                or abs(target_x - t.display_x) > _LABEL_SNAP_PX
                or abs(target_y - t.display_y) > _LABEL_SNAP_PX):
            t.display_x = float(target_x)
            t.display_y = float(target_y)
        else:
            t.display_x += (target_x - t.display_x) * _LABEL_SMOOTH_ALPHA
            t.display_y += (target_y - t.display_y) * _LABEL_SMOOTH_ALPHA

        dx, dy = int(t.display_x), int(t.display_y)
        placed_by_id[t.id] = (dx, dy, dx + lw, dy + lh)
        composite_bgra(frame, label_to_draw, dx, dy)


_stats_last_det_count = 0  # snapshot précédent du cumul détections worker

def processing_loop(cap, app: FaceAnalysis,
                    index: FaceIndex, fbuf: FrameBuffer,
                    skel_buf: FrameBuffer,
                    ndi_out: "NDIOutSender | None",
                    snapshot_mgr: "SnapshotManager | None",
                    stop: threading.Event) -> None:
    """Boucle display thread : capture → tracker.update → overlay → push JPEG.

    Architecture multi-thread :
      - Ce thread (main/display) tourne à TARGET_FPS strict, ne bloque
        jamais sur la détection.
      - DetectionWorker (thread séparé) tourne à DETECT_FPS, fait la
        détection RetinaFace+ArcFace en parallèle (onnxruntime CUDA
        libère le GIL). Submit/get_latest lock-free pour le main.
      - tracker.update() chaque frame : avec nouvelles détections si dispo,
        predict-only sinon. Kalman lisse les bboxes entre détections.
    """
    publish_period = 1.0 / TARGET_FPS
    last_publish = 0.0
    last_index_check = 0.0
    stats_last = time.monotonic()
    stats_count = 0

    # Worker de détection en thread séparé. Le main loop n'appelle JAMAIS
    # app.get() lui-même → pas de blocage par les ~50-100ms de détection
    # quand le peloton est dense.
    det_worker = DetectionWorker(app, DETECT_FPS,
                                  min_face_px=MIN_FACE_PX,
                                  max_faces=MAX_FACES,
                                  blur_min_var=FACE_BLUR_MIN_VAR)
    # SHM publisher : publie chaque frame raw BGR pour body_recog +
    # bib_recog co-localisés (économise leur HTTP+JPEG decode).
    shm_pub: "SHMFramePublisher | None" = None
    if SHM_PUBLISH:
        try:
            shm_pub = SHMFramePublisher(SHM_NAME, SHM_MAX_W, SHM_MAX_H)
        except Exception as e:
            log(f"WARN: SHM publisher échec init ({e}) — body/bib en "
                f"fallback MJPEG")
    # Worker de capture séparé : alimente un slot frame, le main thread
    # sleep entre les publish au lieu de cap.read()-bloquer. Publie aussi
    # en SHM pour les consumers co-localisés.
    cap_worker = CaptureWorker(cap, shm_pub=shm_pub)

    # Tracker custom : 1 Kalman par face (filterpy) + matching IoU + ré-id
    # par embedding ArcFace. Quand un visage est baissé (RetinaFace perd la
    # détection), le track est "lost" mais survit max_missed frames. Si le
    # visage remonte ET que l'embedding match (cosine ≥ reid_threshold),
    # l'identité est restaurée — même track_id, même nom.
    # display_buffer auto = nb publish frames entre 2 détections + 1 buffer.
    # À 30/6 → 6 ; à 60/6 → 11 ; à 30/30 → 2. Évite que la bbox clignote
    # entre 2 détections.
    _display_buf = max(2, int(round(TARGET_FPS / max(1, DETECT_FPS))) + 1)
    track_mgr = TrackManager(
        iou_threshold=0.3,
        max_missed=TRACK_BUFFER,
        reid_threshold=TRACK_REID_THRESHOLD,
        display_buffer=_display_buf,
    )

    # bib_recog : lecture du JSON publié par bib_recog_service (venv
    # séparé, YOLOv8 person + PaddleOCR dossard). Fusion avec les bibs
    # déjà set depuis le JSON partants (via face match) : si OCR confirme
    # le bib partants → track.bib_confirmed = True (puce visuelle), si
    # conflit → face gagne, si pas de bib face → OCR comble.
    bibs_state = BibsState(BIBS_JSON)

    # body_recog actif : lecteur du JSON pose (publié par
    # body_recog_service venv séparé, ~10 fps CUDA). Permet de maintenir
    # un face track quand le visage disparaît temporairement (casque
    # baissé, profil), via la position du keypoint nez/yeux du squelette.
    bodies_state = BodiesState(BODIES_JSON)

    while not stop.is_set():
        now = time.monotonic()

        # Throttle publish au TARGET_FPS via sleep précis (pas de busy-loop,
        # pas de blocage cap.read).
        sleep_for = publish_period - (now - last_publish)
        if sleep_for > 0:
            time.sleep(sleep_for)
            now = time.monotonic()
        last_publish = now

        # Reload index si fichier a changé.
        if now - last_index_check > 2.0:
            if index.reload_if_changed():
                for t in track_mgr.tracks.values():
                    t.name_resolved = False
                    t.label_img     = None
            last_index_check = now

        # Pull la dernière frame du capture worker (drop-oldest).
        frame = cap_worker.get_latest()
        if frame is None:
            continue  # pas encore de frame, on retentera au tick suivant

        # Submit la frame courante au worker (copie nécessaire pour qu'il
        # ait sa propre vue stable pendant qu'on enchaîne sur la suivante).
        det_worker.submit(frame.copy())

        # Récupère la dernière détection disponible. is_new=True seulement
        # si le worker a publié de nouveaux résultats depuis le dernier
        # appel. Sinon on fait du predict-only (tracks avancent en Kalman).
        det_bboxes, det_embeds, is_new = det_worker.get_latest()
        if is_new:
            track_mgr.update(det_bboxes, det_embeds)
        else:
            track_mgr.update([], [])

        # Pour chaque track actif : matching ArcFace 1×, label rendu si match
        # confirmé (score ≥ THRESHOLD). En-dessous du seuil : pas de label
        # affiché — on évite d'afficher des noms inventés pour des visages
        # non-reconnus (public, riders hors BDD, etc.).
        # Body tracking : on lit l'état pose juste pour l'association
        # body_track_id (info utile pour le rendu pseudo-3D ou future
        # exploitation). PAS d'update Kalman face depuis le body : les
        # tentatives précédentes (phantom detection puis cascade face_kp
        # → épaules → bbox) créaient toutes des doublons ou sautillements
        # quand le tracker face re-détecte au mauvais endroit. Mieux :
        # laisser le tracker face faire son taf seul, juste lui donner
        # plus de marge via TRACK_BUFFER (= max_missed).
        bodies_now = bodies_state.get_persons(
            target_w=frame.shape[1], target_h=frame.shape[0],
        )
        active_tracks = track_mgr.active()
        associate_bodies_to_tracks(active_tracks, bodies_now)

        for t in active_tracks:
            # Décide s'il faut (re-)voter :
            #   - jamais résolu ET assez de samples → vote initial
            #   - déjà résolu ET assez de nouveaux samples accumulés
            #     depuis le dernier vote → re-vote (correction possible).
            need_vote = False
            if (not t.name_resolved
                    and len(t.embedding_samples) >= _EMB_SAMPLES_MIN):
                need_vote = True
            elif (t.name_resolved
                    and t.samples_since_last_vote >= _REVOTE_INTERVAL):
                need_vote = True
            if not need_vote:
                continue

            # Multi-frame voting : moyenne des embeddings, re-normalise
            # (les embeddings ArcFace vivent sur l'hypersphère unité).
            avg = np.mean(t.embedding_samples, axis=0)
            norm = float(np.linalg.norm(avg))
            if norm > 1e-6:
                avg = avg / norm
            new_name, new_score, margin = index.match(avg)
            t.samples_since_last_vote = 0

            # Garde-fou top-2 margin : sous le seuil, le match est ambigu
            # (le 2e best est trop proche). On retarde la résolution
            # initiale plutôt que d'afficher un faux nom. Pour un re-vote
            # déjà résolu, on garde le nom actuel (pas de "downgrade"
            # vers ? sur ambiguïté ponctuelle).
            if margin < _MATCH_MIN_MARGIN:
                if not t.name_resolved:
                    continue
                # Re-vote ambigu → on ne change rien, on retentera plus tard.
                continue

            name_changed = (new_name != t.name)
            t.name = new_name
            t.score = new_score
            t.name_resolved = True

            # Si le nom a changé (corruption corrigée ou nouvelle id),
            # on rafraîchit bib + nationalité depuis partants et on
            # invalide le bib_confirmed (l'OCR devra re-confirmer).
            if name_changed or t.nationality is None:
                meta = _RIDERS_META.get(t.name)
                if meta is not None:
                    # Au re-vote, on écrase le bib uniquement si le nom
                    # a réellement changé (sinon, on garde un éventuel
                    # bib OCR comblé pour un rider hors partants).
                    if name_changed:
                        t.bib = meta.get("bib")
                        t.bib_confirmed = False
                    elif t.bib is None:
                        t.bib = meta.get("bib")
                    t.nationality = meta.get("nationality")
                elif name_changed:
                    # Nouveau nom hors partants : on jette les meta.
                    t.bib = None
                    t.nationality = None
                    t.bib_confirmed = False

            # Charge la vignette photo 1× depuis face-db/<uciid>/ pour le
            # rendu skeleton-only. Cache négatif = False si pas trouvée
            # (ne re-tente pas à chaque re-vote). Si le nom change, on
            # invalide pour re-tenter avec le nouveau uciid.
            # UCI ID lookup : priorité manifest rider-recognition (couvre
            # tous les sportifs du dataset), fallback partants TDF.
            if name_changed:
                t.photo_thumb = None
            if t.photo_thumb is None:
                uciid = _NAME_TO_UCIID.get(t.name, "")
                if not uciid:
                    meta = _RIDERS_META.get(t.name)
                    uciid = meta.get("uciid") if meta else ""
                photo = load_rider_photo(uciid, SKELETON_PHOTO_SIZE)
                t.photo_thumb = photo if photo is not None else False

            # (Re-)render label avec hystérésis sur la visibilité :
            #   - SHOW si score >= THRESHOLD * 0.7 (initial trigger)
            #   - KEEP visible tant que score >= THRESHOLD * 0.4 (= déjà
            #     visible, on tolère un score qui retombe sans effacer)
            #   - HIDE seulement si score < THRESHOLD * 0.4
            # Évite le flicker on/off à chaque re-vote quand le score
            # moyen oscille autour du seuil unique. Au-delà de la zone
            # rouge stricte (0.4), on cache vraiment.
            show_thresh = THRESHOLD * 0.7
            keep_thresh = THRESHOLD * 0.4
            currently_shown = t.label_img is not None
            should_show = (t.score >= show_thresh) or (
                currently_shown and t.score >= keep_thresh
            )
            if should_show:
                t.label_img = render_lower_third(
                    t.name, t.score,
                    bib=t.bib, nationality=t.nationality,
                    bib_confirmed=t.bib_confirmed,
                )
            else:
                t.label_img = None

        # Fusion bib OCR : applique le cross-check après résolution face.
        # Si bibs OCR ont changé l'état d'un track (nouveau bib OU passage
        # confirmé), on re-render les labels affectés. La fonction met
        # aussi à jour t.body_track_id en passant (info utile association).
        bibs_now = bibs_state.get()
        if bibs_now:
            bibs_changed = associate_bibs_to_tracks(active_tracks, bibs_now)
            if bibs_changed:
                for t in active_tracks:
                    if t.label_img is None or t.score < THRESHOLD * 0.7:
                        continue
                    t.label_img = render_lower_third(
                        t.name, t.score,
                        bib=t.bib, nationality=t.nationality,
                        bib_confirmed=t.bib_confirmed,
                    )

        # Snapshot capture (active learning) : sauve les crops face des
        # tracks reconnus avec très haute confiance + bib_confirmed dans
        # staging pour review humain. Gate cumulatifs (score, margin,
        # bib, blur, size) + rate limit + dedup. Frame est encore RAW à
        # ce stade (annotations dessinées juste après) — c'est ce qu'on
        # veut sauvegarder.
        if SNAPSHOT_ENABLE and snapshot_mgr is not None and active_tracks:
            now_t = time.time()
            for t in active_tracks:
                if not t.name_resolved:
                    continue
                if t.score < THRESHOLD + SNAPSHOT_MIN_SCORE_DELTA:
                    continue
                if (SNAPSHOT_REQUIRE_BIB_CONFIRMED
                        and not t.bib_confirmed):
                    continue
                uciid = _NAME_TO_UCIID.get(t.name, "")
                if not uciid:
                    continue
                if not snapshot_mgr.can_save(uciid, now_t):
                    continue
                # Crop face avec padding autour de la bbox Kalman.
                fx1, fy1, fx2, fy2 = (float(v) for v in t.bbox_xyxy())
                cx = (fx1 + fx2) * 0.5
                cy = (fy1 + fy2) * 0.5
                hw = (fx2 - fx1) * 0.5 * SNAPSHOT_CROP_PADDING
                hh = (fy2 - fy1) * 0.5 * SNAPSHOT_CROP_PADDING
                fh, fw = frame.shape[:2]
                px1 = max(0, int(cx - hw))
                py1 = max(0, int(cy - hh))
                px2 = min(fw, int(cx + hw))
                py2 = min(fh, int(cy + hh))
                if (px2 - px1) < SNAPSHOT_MIN_SIZE_PX or (
                        py2 - py1) < SNAPSHOT_MIN_SIZE_PX:
                    continue
                crop = frame[py1:py2, px1:px2]
                if crop.size == 0:
                    continue
                # Sharpness check (Laplacian variance sur grayscale).
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if blur_var < SNAPSHOT_MIN_BLUR_VAR:
                    continue
                # Dedup vs snapshots déjà save cette session.
                avg_emb = np.mean(t.embedding_samples, axis=0)
                avg_norm = float(np.linalg.norm(avg_emb))
                if avg_norm > 1e-6:
                    avg_emb = avg_emb / avg_norm
                if snapshot_mgr.is_dedup(uciid, avg_emb):
                    continue
                # Save (toutes les gates passées).
                # On utilise le margin du dernier vote ; il a été reset à
                # 0 par samples_since_last_vote=0 mais on l'a recalculé
                # plus haut. Pour simplicité, on ré-évalue ici (l'avg
                # embedding est encore frais).
                _, _, m_now = index.match(avg_emb)
                if m_now < SNAPSHOT_MIN_MARGIN:
                    continue
                snapshot_mgr.save(
                    uciid, t.name, crop, avg_emb,
                    t.score, m_now, t.bib_confirmed, now_t,
                )

        # Rendu : pipeline normal (vidéo annotée + labels + body opt.).
        # Bbox body en arrière-plan (toggleable). Labels face dessinés
        # APRÈS pour rester au-dessus visuellement.
        if DRAW_BODIES and bodies_now:
            draw_bodies(frame, bodies_now)

        if active_tracks:
            draw_tracks(frame, active_tracks, bodies_now)

        # Met à jour la dernière frame native dispo (utile si un POST
        # /bullet-time arrive : on freeze ce contenu). Avant warp.
        bullet_state.update_live(frame)

        # Bullet time : si actif, on REMPLACE la frame courante par la
        # frame freezée warpée selon l'angle yaw animé. Tous les overlays
        # (labels, etc.) déjà dessinés sur la frame live → on les écrase
        # mais c'est volontaire (effet cinéma : freeze pur + parallax).
        bt_active, bt_elapsed, bt_frame, bt_depth = bullet_state.get_state()
        if bt_active and bt_frame is not None and bt_depth is not None:
            yaw = bullet_time_yaw(bt_elapsed)
            frame = warp_3d_photo(bt_frame, bt_depth, yaw)

        # NDI out : pousse la frame annotée à RES NATIVE (= avant
        # PUBLISH_HEIGHT downscale) vers la régie broadcast. Désactivé si
        # NDI_OUT_NAME vide.
        if ndi_out is not None:
            try:
                ndi_out.send(frame, int(TARGET_FPS))
            except Exception as e:
                log(f"NDI out send err: {e}")

        # Downscale optionnel avant encode JPEG (allège le décodage browser à
        # haut fps). Garde le ratio source. INTER_AREA = bonne qualité downscale.
        if PUBLISH_HEIGHT and frame.shape[0] > PUBLISH_HEIGHT:
            ratio = PUBLISH_HEIGHT / frame.shape[0]
            new_w = int(round(frame.shape[1] * ratio))
            frame = cv2.resize(frame, (new_w, PUBLISH_HEIGHT),
                               interpolation=cv2.INTER_AREA)

        # Encode + push MJPEG /stream.mjpeg UNIQUEMENT si au moins un
        # client browser regarde le preview. body_recog + bib_recog ne
        # consomment plus le MJPEG (ils lisent en SHM), donc 0 subscribers
        # = personne ne regarde = pas d'encode = ~30% CPU libérée.
        if fbuf.subscribers > 0:
            ok, buf = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                fbuf.push(bytes(buf))

        # Si quelqu'un consomme le flux skeleton, on rend la vue
        # séparée et on push. Skip render quand 0 client (économie CPU).
        if skel_buf.subscribers > 0:
            skel = np.zeros_like(frame)
            draw_skeleton_view(skel, bodies_now, active_tracks)
            ok2, sbuf = cv2.imencode(".jpg", skel,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok2:
                skel_buf.push(bytes(sbuf))

        # Stats périodiques. Le compteur de détections vient du worker
        # (cumulé) — on garde la diff sur la fenêtre 5s.
        stats_count += 1
        if now - stats_last > 5.0:
            dt = now - stats_last
            fps = stats_count / dt
            global _stats_last_det_count
            cur_dets = det_worker.detections_count
            dets_per_sec = (cur_dets - _stats_last_det_count) / dt
            _stats_last_det_count = cur_dets
            n_tracks = len(track_mgr.tracks)
            log(f"{fps:.1f} fps published, {dets_per_sec:.1f} détections/s, "
                f"tracks={n_tracks}, index={len(index.names)} sportifs")
            stats_count = 0
            stats_last = now


# ──────────────────────── HTTP MJPEG server ──────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    # Références injectées via attributs classe par main().
    fbuf: FrameBuffer = None
    skel_buf: FrameBuffer = None

    def log_message(self, format, *args):  # noqa: D401 - silence default logs
        pass  # already logging via processing thread

    def _serve_mjpeg(self, buf: "FrameBuffer") -> None:
        """Serve une boucle MJPEG depuis un FrameBuffer donné, en
        incrémentant le compteur de subscribers pour activer le rendu
        côté producteur."""
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=avtowan-mjpeg")
        self.end_headers()
        self.connection.settimeout(2.0)
        with buf.lock:
            buf.subscribers += 1
        last_id = 0
        try:
            while True:
                res = buf.wait_new(last_id, timeout=2.0)
                if res is None:
                    continue
                jpeg, last_id = res
                chunk  = b"--avtowan-mjpeg\r\n"
                chunk += b"Content-Type: image/jpeg\r\n"
                chunk += f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                chunk += jpeg
                chunk += b"\r\n"
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, socket.timeout,
                TimeoutError, OSError):
            return
        finally:
            with buf.lock:
                buf.subscribers = max(0, buf.subscribers - 1)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            # Wrapper HTML fullscreen : <img> auto-scale viewport, fond
            # noir, curseur masqué. Le browser décode le MJPEG natif.
            self._serve_fullscreen("/stream.mjpeg", "face-recog")
            return
        elif self.path == "/skeleton" or self.path == "/skeleton.html":
            self._serve_fullscreen("/stream-skeleton.mjpeg",
                                   "face-recog skeleton")
            return
        elif self.path == "/stream.mjpeg":
            self._serve_mjpeg(self.fbuf)
            return
        elif self.path == "/stream-skeleton.mjpeg":
            self._serve_mjpeg(self.skel_buf)
            return
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_fullscreen(self, stream_path: str, title: str) -> None:
        html = (
            "<!DOCTYPE html><html><head>"
            f"<title>{title}</title>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,"
            "initial-scale=1\">"
            "<style>"
            "html,body{margin:0;padding:0;height:100%;background:#000;"
            "overflow:hidden;cursor:none;}"
            "img{display:block;width:100vw;height:100vh;"
            "object-fit:contain;}"
            "</style></head><body>"
            f"<img src=\"{stream_path}\" alt=\"\">"
            "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(html)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if self.path == "/bullet-time":
            # Trigger bullet-time : lit la dernière depth dispo + freeze
            # la dernière frame live. Réponse JSON : {ok, reason}.
            d = depth_state_global.get()
            if d is None:
                body = b'{"ok":false,"reason":"depth not ready"}'
                code = 503
            else:
                ok = bullet_state.trigger(d[0])
                if ok:
                    body = b'{"ok":true}'
                    code = 200
                else:
                    body = b'{"ok":false,"reason":"already active or no frame yet"}'
                    code = 409
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


# ──────────────────────── Main ──────────────────────────
def main() -> None:
    log(f"démarrage SOURCE={SOURCE} INDEX={INDEX} HTTP_PORT={HTTP_PORT} "
        f"THRESHOLD={THRESHOLD} GPU_ID={GPU_ID}")

    log(f"init détecteur {INSIGHTFACE_MODEL}...")
    t0 = time.monotonic()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if GPU_ID >= 0
                 else ["CPUExecutionProvider"])
    app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
    app.prepare(ctx_id=GPU_ID, det_size=(DET_SIZE, DET_SIZE))
    log(f"détecteur ready en {time.monotonic() - t0:.1f}s")

    index = FaceIndex(INDEX)
    if len(index.names) == 0:
        log(f"WARN: index vide ou introuvable ({INDEX}) — bbox seront affichés "
            f"sans nom (run index_faces.py d'abord)")

    # Méta riders (bib + nationalité) depuis le JSON partants ASO.
    global _RIDERS_META, _NAME_TO_UCIID
    _RIDERS_META = load_riders_meta(PARTANTS_JSON)
    if _RIDERS_META:
        log(f"partants chargés : {len(_RIDERS_META)} riders depuis "
            f"{PARTANTS_JSON}")
    else:
        log(f"partants JSON absent/vide ({PARTANTS_JSON}) — labels sans "
            f"bib ni nationalité")

    # Map name → uciid (pour le lookup photo, plus large que partants).
    _NAME_TO_UCIID = load_name_to_uciid(RIDER_MANIFEST_JSON)
    if _NAME_TO_UCIID:
        log(f"manifest rider chargé : {len(_NAME_TO_UCIID)} name→uciid "
            f"depuis {RIDER_MANIFEST_JSON}")
    else:
        log(f"manifest rider absent/vide ({RIDER_MANIFEST_JSON}) — photos "
            f"limitées aux UCI IDs des partants")

    cap = open_source(SOURCE)
    fbuf = FrameBuffer()
    skel_buf = FrameBuffer()

    # NDI sender out (optionnel via NDI_OUT_NAME). Init lazy à la 1ère
    # frame côté processing_loop pour caler sur la résolution réelle.
    ndi_out: NDIOutSender | None = None
    if NDI_OUT_NAME:
        ndi_out = NDIOutSender(NDI_OUT_NAME)
        log(f"NDI sender out armé : '{NDI_OUT_NAME}' (init lazy à 1ère frame)")

    # Snapshot capture (active learning), opt-in via SNAPSHOT_ENABLE.
    snapshot_mgr: SnapshotManager | None = None
    if SNAPSHOT_ENABLE:
        snapshot_mgr = SnapshotManager(SNAPSHOTS_DIR)
        log(f"snapshot capture actif : dir={SNAPSHOTS_DIR} "
            f"score≥{THRESHOLD + SNAPSHOT_MIN_SCORE_DELTA:.2f} "
            f"margin≥{SNAPSHOT_MIN_MARGIN} "
            f"bib_confirmed_req={SNAPSHOT_REQUIRE_BIB_CONFIRMED}")

    stop = threading.Event()
    proc = threading.Thread(target=processing_loop,
                            args=(cap, app, index, fbuf, skel_buf,
                                  ndi_out, snapshot_mgr, stop),
                            daemon=True)
    proc.start()

    MJPEGHandler.fbuf = fbuf
    MJPEGHandler.skel_buf = skel_buf
    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), MJPEGHandler)
    log(f"HTTP MJPEG sur :{HTTP_PORT}/stream.mjpeg "
        f"+ /stream-skeleton.mjpeg")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("SIGINT — shutdown")
    finally:
        stop.set()
        server.server_close()
        cap.release()
        if shm_pub is not None:
            shm_pub.close()
        if ndi_out is not None:
            ndi_out.close()


if __name__ == "__main__":
    main()
