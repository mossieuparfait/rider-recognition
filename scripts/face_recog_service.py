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
# Encode JPEG GPU (NVJPEG via ctypes, cf scripts/nvjpeg_encoder.py).
# 0 = cv2.imencode CPU (~10 ms à 720p), 1 = NvJpegEncoder GPU (~1.5 ms).
# Fallback automatique CPU si l'import ou l'init échoue.
NVJPEG_ENABLED = bool(int(env("NVJPEG_ENABLED", "1")))

# Compositing labels DERRIÈRE les riders via mask alpha publié par
# avtowan-mask-recog.service (RVM matting GPU). Si actif :
#   1. snapshot frame avant draw_tracks (= frame propre sans labels)
#   2. draw_tracks dessine les labels sur frame
#   3. ré-blend frame_orig au-dessus là où mask dit "rider" → label
#      apparaît visuellement DERRIÈRE le rider.
# Si mask absent ou stale → fallback no-op (labels au-dessus).
MASK_BEHIND_LABELS = bool(int(env("MASK_BEHIND_LABELS", "0")))
MASK_SHM_NAME = env("MASK_SHM_NAME", "avtowan-mask")
MASK_STALE_MAX_AGE_S = float(env("MASK_STALE_MAX_AGE_S", "0.5"))

# Rendu logo équipe derrière chaque coureur reconnu. Le logo est
# composité semi-transparent sur la frame avant les labels, puis le
# mask alpha (RVM) le cache là où le corps du coureur est devant →
# logo apparaît comme posé sur le fond gris derrière le rider.
# Scene logo (générique, indépendant des riders) — affiche UN logo à
# position fixe sur l'écran, semi-transparent, derrière les personnes
# (mask RVM cache les zones humaines). Cas type : logo course / partenaire
# sur fond de plateau, à incruster sous des intervenants en présentation.
SCENE_LOGO_BEHIND = bool(int(env("SCENE_LOGO_BEHIND", "0")))
SCENE_LOGO_PATH = env("SCENE_LOGO_PATH", "/var/lib/avtowan/scene-logos/TDF.png")
SCENE_LOGO_ALPHA = float(env("SCENE_LOGO_ALPHA", "0.45"))
SCENE_LOGO_HEIGHT = int(env("SCENE_LOGO_HEIGHT", "500"))
SCENE_LOGO_X_FRACTION = float(env("SCENE_LOGO_X_FRACTION", "0.5"))
SCENE_LOGO_Y_FRACTION = float(env("SCENE_LOGO_Y_FRACTION", "0.45"))

# Mode chroma-key : au lieu du mask RVM (qui rate les personnes en costume
# / hors-rider), on calcule un mask par proximité de couleur au fond.
# Le logo n'apparaît que là où le pixel courant est proche de la couleur
# clé (= fond gris). Tout le reste reste visuellement devant le logo.
SCENE_LOGO_CHROMA_KEY = bool(int(env("SCENE_LOGO_CHROMA_KEY", "0")))
# Couleur clé en BGR (format "B,G,R"). 60,60,60 = gris foncé typique
# backdrop studio. Adjust selon la scène.
SCENE_LOGO_KEY_BGR = env("SCENE_LOGO_KEY_BGR", "60,60,60")
# Distance euclidienne max au-delà de laquelle le pixel est considéré
# "pas le fond". Soft band de 20 unités pour bords lisses.
SCENE_LOGO_KEY_TOLERANCE = float(env("SCENE_LOGO_KEY_TOLERANCE", "35"))
SCENE_LOGO_KEY_SOFTNESS = float(env("SCENE_LOGO_KEY_SOFTNESS", "20"))

TEAM_LOGO_BEHIND = bool(int(env("TEAM_LOGO_BEHIND", "0")))
TEAM_LOGOS_DIR = env("TEAM_LOGOS_DIR", "/var/lib/avtowan/team-logos")
TEAM_LOGO_ALPHA = float(env("TEAM_LOGO_ALPHA", "0.5"))
# Hauteur du logo en pixels (sur le frame natif, pré-resize).
TEAM_LOGO_HEIGHT = int(env("TEAM_LOGO_HEIGHT", "400"))
# Position FIXE du logo sur l'écran (pas par rider) : centré horizontalement
# par défaut, fraction de la hauteur frame pour Y. Plusieurs équipes →
# alignées en row centré.
TEAM_LOGO_Y_FRACTION = float(env("TEAM_LOGO_Y_FRACTION", "0.4"))
TEAM_LOGO_SPACING_PX = int(env("TEAM_LOGO_SPACING_PX", "60"))

# Tableau "compo équipe" affiché au-dessus de la tête d'un rider reconnu
# (mode podium). Toggle au boot via SHOW_TEAM_ROSTER, mutable à chaud via
# POST /podium-mode {"on": bool} ou POST /podium-mode/toggle.
SHOW_TEAM_ROSTER = bool(int(env("SHOW_TEAM_ROSTER", "0")))
TEAM_ROSTER_FONT_SIZE = int(env("TEAM_ROSTER_FONT_SIZE", "22"))
TEAM_ROSTER_HEADER_FONT_SIZE = int(env("TEAM_ROSTER_HEADER_FONT_SIZE", "24"))
TEAM_ROSTER_GAP_PX = int(env("TEAM_ROSTER_GAP_PX", "30"))
TEAM_ROSTER_MARGIN_PX = int(env("TEAM_ROSTER_MARGIN_PX", "18"))
TEAM_ROSTER_BG_ALPHA = float(env("TEAM_ROSTER_BG_ALPHA", "0.78"))
TEAM_ROSTER_ACCENT_HEX = env("TEAM_ROSTER_ACCENT_HEX", "#a371f7")
# Position fixe du tableau : fraction de la hauteur frame pour le bord
# haut de la carte. 0.04 = ~4% du haut. X est toujours centré sur le
# frame (pas attaché à un rider).
TEAM_ROSTER_TOP_Y_FRACTION = float(env("TEAM_ROSTER_TOP_Y_FRACTION", "0.04"))
# Sticky court : on ne change l'équipe affichée qu'après ce délai de
# stabilité d'une équipe challenger majoritaire. Tue le ping-pong quand
# le compte d'apparitions oscille frame-à-frame.
TEAM_ROSTER_STICKY_S = float(env("TEAM_ROSTER_STICKY_S", "2.0"))
# Si aucun rider reconnu n'apparaît pendant ce délai (s), le tableau
# disparaît. Permet de cacher la card en off-shot sans toucher au
# toggle global. Mettre à 0 pour ne jamais cacher (la dernière équipe
# vue reste à l'écran tant que le mode est ON).
TEAM_ROSTER_IDLE_HIDE_S = float(env("TEAM_ROSTER_IDLE_HIDE_S", "10.0"))
# Chroma-key fond plateau : le tableau ne s'incruste QUE là où le pixel
# de la frame est proche de la couleur clé (= fond gris). Sur les
# personnes / micros / mobilier qui dépassent, le tableau s'efface
# naturellement → effet "fondu plateau". Réutilise les paramètres
# SCENE_LOGO_KEY_BGR / _TOLERANCE / _SOFTNESS (même scène, même fond).
# Si un mask RVM est dispo (MASK_BEHIND_LABELS=1), on prend l'UNION
# des deux : chroma capte les objets non-humains, RVM capte les
# vêtements de couleur proche du fond.
TEAM_ROSTER_CHROMA_KEY = bool(int(env("TEAM_ROSTER_CHROMA_KEY", "1")))
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
# Stickiness du nom : au re-vote d'un track DÉJÀ résolu, on ne remplace
# t.name que si new_score dépasse t.score d'au moins ce delta. Empêche
# les riders pile au seuil de basculer entre 2 identités proches à chaque
# re-vote (le margin_min filtre l'ambiguïté du match courant, pas la
# concurrence vs le résultat précédent). 0.05 = exige une vraie domination
# du nouveau, 0 = pas de stickiness (comportement historique).
_NAME_SWITCH_MARGIN = float(env("NAME_SWITCH_MARGIN", "0.05"))
# Lissage EMA du t.score affiché après résolution. Au re-vote :
#   t.score = (1 - alpha) * t.score + alpha * new_score
# 1.0 = pas de lissage (comportement historique, overwrite direct).
# 0.4 = lisse ~3 re-votes, stabilise la zone autour du show_thresh pour
# les borderline. Note : on lisse seulement si le nom ne change pas, sinon
# on snap (changement d'identité = vrai jump, pas un lissage).
_SCORE_SMOOTH_ALPHA = float(env("SCORE_SMOOTH_ALPHA", "0.4"))


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

    def match_batch(self, embs: np.ndarray) -> list[tuple[str, float, float]]:
        """Batch version : embs shape (K, 512), retourne K (name, top, margin).
        Un seul np.dot pour tous → ~K× plus rapide que K appels individuels."""
        if self.embeddings.shape[0] == 0 or len(embs) == 0:
            return [("?", 0.0, 0.0)] * len(embs)
        # sims shape (N_index, K)
        sims = self.embeddings @ embs.T
        idx = np.argmax(sims, axis=0)               # (K,)
        top = sims[idx, np.arange(len(idx))]        # (K,)
        out: list[tuple[str, float, float]] = []
        if self.embeddings.shape[0] >= 2:
            # Masque le top par colonne pour trouver le 2e best.
            sims_masked = sims.copy()
            sims_masked[idx, np.arange(len(idx))] = -2.0
            second = sims_masked.max(axis=0)         # (K,)
            margins = top - second
        else:
            margins = np.zeros(len(embs), dtype=np.float32)
        for k in range(len(embs)):
            out.append((self.names[int(idx[k])], float(top[k]),
                         float(margins[k])))
        return out


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

        # Format réception NDI :
        #   BGRX_BGRA  : 4 bytes/pixel (alpha + BGR). Toujours dispo, mais
        #                ~9% CPU en cv2.cvtColor(BGRA2BGR) sur la frame
        #                (cf profile py-spy 2026-05-29).
        #   UYVY_BGRA  : 2 bytes/pixel (YUV 4:2:2 packed) pour les sources
        #                opaques (HB SpeedHQ est YUV natif). Memcpy moitié,
        #                UYVY→BGR cv2 hautement vectorisé.
        #   fastest    : laisse le SDK décider — typiquement UYVY pour HB.
        # Env override pour bench/back-out facile.
        recv_color_name = env("NDI_RECV_COLOR", "BGRX_BGRA")
        recv_color = getattr(RecvColorFormat, recv_color_name,
                              RecvColorFormat.BGRX_BGRA)
        self._receiver = Receiver(
            color_format=recv_color,
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
        # Workaround Python multiprocessing resource_tracker bug
        # (https://bugs.python.org/issue38119) : SharedMemory(create=True)
        # registre le segment auprès du resource_tracker daemon qui
        # l'unlink agressivement (parfois immédiatement, parfois à l'exit
        # d'un sous-process). Résultat : /dev/shm/<name> disparaît alors
        # qu'on a encore le fd ouvert → les readers ne peuvent plus
        # attacher. On unregister manuellement : le publisher gère son
        # propre cleanup via close() explicite.
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(self._shm._name, "shared_memory")
        except Exception:
            pass
        self._buf = self._shm.buf
        self._seq = 0
        # Permet la lecture cross-user (face-recog tourne en root, des
        # consommateurs comme avtowan-mask-recog.service tournent en ben).
        # 0666 plutôt que 0664 car Python `shared_memory.SharedMemory(name=...)`
        # ouvre toujours en O_RDWR, donc le reader a besoin de write perm
        # même s'il ne fait que lire. SHM local sur machine dédiée → OK.
        try:
            os.chmod(f"/dev/shm/{name}", 0o666)
        except OSError:
            pass
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


class SHMMaskReader:
    """Reader seqlock du SHM mask alpha uint8 publié par avtowan-mask-recog.

    Layout (cf mask_recog_service.py) :
      0  seq u64    (impair = write in progress)
      8  ts_ns u64
      16 width u32
      20 height u32
      24 format u32 (0 = uint8 alpha)
      28 reserved u32
      32 raw uint8 alpha (w*h bytes)

    Attache lazy au SHM ; renvoie None si segment absent (service mask
    pas démarré) ou si frame trop ancienne (stale_max_age_s).
    """

    HEADER_SIZE = 32

    def __init__(self, name: str, stale_max_age_s: float = 0.5) -> None:
        self._name = name
        self._shm = None
        self._struct = __import__("struct")
        self._stale_ns = int(stale_max_age_s * 1e9)
        self._last_seq = 0

    def _attach(self) -> bool:
        if self._shm is not None:
            return True
        try:
            from multiprocessing import shared_memory
            self._shm = shared_memory.SharedMemory(name=self._name)
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(self._shm._name, "shared_memory")
            except Exception:
                pass
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def latest(self) -> "np.ndarray | None":
        """Renvoie le dernier mask uint8 (H,W) ou None. Copie le contenu
        pour ne pas dépendre du SHM buf après retour."""
        if not self._attach():
            return None
        buf = self._shm.buf
        for _ in range(8):
            seq_a = self._struct.unpack_from("<Q", buf, 0)[0]
            if seq_a & 1:
                continue
            if seq_a == 0:
                return None  # publisher pas encore écrit
            ts_ns, w, h, fmt, _ = self._struct.unpack_from(
                "<QIIII", buf, 8)
            if fmt != 0 or w == 0 or h == 0:
                return None
            n = w * h
            if self.HEADER_SIZE + n > len(buf):
                return None
            mask = np.frombuffer(buf, dtype=np.uint8,
                                  count=n, offset=self.HEADER_SIZE).reshape(h, w)
            seq_b = self._struct.unpack_from("<Q", buf, 0)[0]
            if seq_b != seq_a:
                continue
            # Stale check : si timestamp trop vieux, skip (= mask-recog
            # ne suit plus le rythme, on préfère ne pas l'utiliser).
            if self._stale_ns > 0:
                if time.monotonic_ns() - ts_ns > self._stale_ns:
                    return None
            self._last_seq = seq_a
            return mask.copy()
        return None


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


class _LoopingFileCapture:
    """Duck-type cv2.VideoCapture pour un fichier vidéo en boucle.

    Auto-rewind à EOF + throttle à la cadence native du fichier (sinon
    read() retourne à la vitesse de décodage = des centaines de fps,
    inutile et confondant pour le tracker). Sert au mode test studio
    où on substitue un MP4 à la place de la capture SDI live.

    Optionnel : start_s/end_s définissent une fenêtre [start, end[ dans
    le fichier (en secondes). Hors fenêtre on rewind à start_s. Utile
    pour boucler sur un segment intéressant en évitant intros/outros.
    """

    # Persistance position pour reprendre au même endroit après un
    # restart service (workflow itératif : modif env → restart → on
    # veut pas re-attendre la scène de test). État valide < 5 min,
    # même path, dans la fenêtre [start, end[.
    _STATE_PATH = "/var/lib/avtowan/playback_state.json"
    _STATE_MAX_AGE_S = 300

    def __init__(self, cap, path: str, fps: float,
                 start_s: float = 0.0, end_s: float | None = None):
        self._cap = cap
        self._path = path
        self._period = 1.0 / max(1.0, fps)
        self._next_t = time.monotonic()
        self._start_ms = max(0.0, start_s) * 1000.0
        self._end_ms = (end_s * 1000.0) if (end_s is not None and end_s > 0) else None
        self._last_save_t = time.monotonic()

        resumed = self._try_resume()
        if not resumed and self._start_ms > 0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, self._start_ms)

    def _try_resume(self) -> bool:
        import json as _json
        try:
            with open(self._STATE_PATH) as f:
                state = _json.load(f)
            if state.get("path") != self._path:
                return False
            if time.time() - float(state.get("saved_at", 0)) > self._STATE_MAX_AGE_S:
                return False
            pos_ms = float(state["pos_ms"])
            if pos_ms < self._start_ms:
                return False
            if self._end_ms is not None and pos_ms >= self._end_ms:
                return False
            self._cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
            log(f"resumed playback at {pos_ms/1000:.1f}s "
                f"(state file < {self._STATE_MAX_AGE_S}s old)")
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def _save_state(self) -> None:
        import json as _json, tempfile, os as _os
        try:
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            _os.makedirs(_os.path.dirname(self._STATE_PATH), exist_ok=True)
            # Écriture atomique : write tmp + rename.
            tmp = self._STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                _json.dump({"path": self._path,
                             "pos_ms": pos_ms,
                             "saved_at": time.time()}, f)
            _os.replace(tmp, self._STATE_PATH)
        except OSError:
            pass

    def isOpened(self) -> bool: return self._cap.isOpened()
    def set(self, *args, **kw): return self._cap.set(*args, **kw)
    def get(self, *args, **kw): return self._cap.get(*args, **kw)
    def release(self): return self._cap.release()

    def _rewind(self) -> None:
        self._cap.set(cv2.CAP_PROP_POS_MSEC, self._start_ms)

    def read(self):
        now = time.monotonic()
        if self._next_t > now:
            time.sleep(self._next_t - now)
        self._next_t = max(self._next_t + self._period, time.monotonic())
        if self._end_ms is not None:
            cur_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if cur_ms >= self._end_ms:
                self._rewind()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._rewind()
            ok, frame = self._cap.read()
        # Persistance pos ~1×/s.
        if now - self._last_save_t > 1.0:
            self._last_save_t = now
            self._save_state()
        return ok, frame


def open_source(spec: str):
    """Ouvre la source vidéo. Supporte :
      - 'v4l2:/dev/videoN'   → cv2.VideoCapture (Magewell, webcam, etc.)
      - 'file:<path>'        → fichier vidéo en boucle (mode test)
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
    if spec.startswith("file:"):
        path = spec[len("file:"):]
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            sys.exit(f"FATAL: ouverture {path} échec (fichier absent ?)")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        start_s = float(env("LOOP_START_S", "0"))
        end_s_raw = env("LOOP_END_S", "")
        end_s = float(end_s_raw) if end_s_raw else None
        win_msg = (f", loop [{start_s:.1f}s, {end_s:.1f}s["
                   if end_s is not None else
                   (f", loop [{start_s:.1f}s, EOF[" if start_s > 0 else ", loop full"))
        log(f"source file {path} ouverte ({w}x{h} @ {fps:.1f}fps{win_msg}+throttle)")
        return _LoopingFileCapture(cap, path, fps, start_s=start_s, end_s=end_s)
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
        # Pause : si set, le worker arrête d'avancer la source (= freeze
        # de la dernière frame). Le display thread continue à recevoir la
        # même frame, donc détecteur + tracker tournent sur image figée.
        # Utile pour tester en stop-action sur un fichier vidéo.
        self._paused = threading.Event()
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

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def _run(self) -> None:
        log("capture worker thread started")
        while not self._stop.is_set():
            if self._paused.is_set():
                # Pause : on n'avance pas la source, mais on continue
                # à re-publier la dernière frame dans le SHM à ~20Hz pour
                # que les consommateurs (mask-recog notamment) conservent
                # leur cadence et ne tombent pas en stale. Sans ça, mask
                # devient stale > MASK_STALE_MAX_AGE_S → TEAM_LOGO_BEHIND
                # et MASK_BEHIND_LABELS tombent en no-mask en pause.
                if self._shm_pub is not None:
                    with self._lock:
                        f = self._frame
                    if f is not None:
                        try:
                            self._shm_pub.publish(f)
                        except Exception as e:
                            log(f"SHM publish err: {e}")
                time.sleep(0.05)
                continue
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
# Chrono overlay (bottom-right) — police monospace pour stabilité visuelle
# (les digits ne dansent pas en largeur quand l'heure change).
_FONT_CLOCK_SIZE = int(env("CLOCK_FONT_SIZE", "40"))
_FONT_CLOCK_PATH = env("CLOCK_FONT_PATH",
                       "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
try:
    _FONT_CLOCK = ImageFont.truetype(_FONT_CLOCK_PATH, _FONT_CLOCK_SIZE)
except Exception:
    _FONT_CLOCK = ImageFont.truetype(_FONT_PATH, _FONT_CLOCK_SIZE)

# Cache chrono base (text → BGRA premul NON warpé). Re-rendu uniquement au
# changement de seconde. Le warp dynamique se fait par-dessus avec cache
# par bucket d'angle.
_CLOCK_BASE_CACHE: tuple[str, np.ndarray] | None = None
# Cache warp par bucket d'angle yaw (1° bucket = ~11 entrées pour une
# oscillation ±5°). Invalidé quand la base change (= changement de seconde).
_CLOCK_WARP_CACHE: dict[int, np.ndarray] = {}
# Toggle + position + style 3D + animation.
_CLOCK_ENABLED         = bool(int(env("CLOCK_ENABLED", "1")))
_CLOCK_YAW_DEG         = float(env("CLOCK_YAW_DEG", "5.0"))
_CLOCK_MARGIN_PX       = int(env("CLOCK_MARGIN_PX", "32"))
# Animation continue (osc yaw + bob) — désactivée par défaut.
_CLOCK_YAW_AMP_DEG     = float(env("CLOCK_YAW_AMP_DEG", "0.0"))
_CLOCK_ORBIT_PERIOD_S  = float(env("CLOCK_ORBIT_PERIOD_S", "5.0"))
_CLOCK_Y_BOB_AMP_PX    = int(env("CLOCK_Y_BOB_AMP_PX", "0"))
# Mini-animation au tick seconde : la plaque arrive du dessus en
# slide-down + fade-in, ease-out cubic pour qu'elle se pose doucement.
# Durée typique 200-300ms ; 0 = pas d'animation.
_CLOCK_TICK_DURATION_S = float(env("CLOCK_TICK_DURATION_S", "0.25"))
_CLOCK_TICK_SLIDE_PX   = int(env("CLOCK_TICK_SLIDE_PX", "14"))
# Instant (monotonic) où le texte courant est apparu pour la 1ère fois.
# Utilisé pour calculer la progression de l'anim tick.
_CLOCK_TICK_INSTANT: float | None = None
_CLOCK_LAST_TEXT: str | None = None


def _render_clock_card(text: str) -> Image.Image:
    """Rend la plaque chrono PIL (RGBA) — gradient dark + bevel haut/bas
    + drop shadow externe + texte blanc avec ombre portée interne. Pas de
    warp ici, fait après par _orbital_warp."""
    # Mesures texte.
    bbox = _FONT_CLOCK.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = 22
    pad_y_top = 12
    pad_y_bot = 14
    radius = 16
    shadow_off = 5  # offset du drop shadow externe

    inner_w = text_w + 2 * pad_x
    inner_h = pad_y_top + text_h + pad_y_bot

    # Canvas avec marge pour le drop shadow.
    canvas_w = inner_w + 2 * shadow_off
    canvas_h = inner_h + 2 * shadow_off
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Drop shadow externe — rect flouté approximé via 2 passes
    # alpha décroissantes (PIL n'a pas de blur direct sans Filter import).
    shadow_x = shadow_off + shadow_off
    shadow_y = shadow_off + shadow_off
    draw.rounded_rectangle(
        (shadow_x, shadow_y,
         shadow_x + inner_w, shadow_y + inner_h),
        radius=radius, fill=(0, 0, 0, 90),
    )
    draw.rounded_rectangle(
        (shadow_x - 1, shadow_y - 1,
         shadow_x + inner_w + 1, shadow_y + inner_h + 1),
        radius=radius + 1, fill=(0, 0, 0, 40),
    )

    # 2. Fond principal — empilage 2 rects en couleurs proches pour faux
    # gradient (haut un poil plus clair, bas plus sombre = profondeur).
    x0 = shadow_off
    y0 = shadow_off
    x1 = x0 + inner_w
    y1 = y0 + inner_h
    # Couche bas (dark base).
    draw.rounded_rectangle(
        (x0, y0, x1, y1), radius=radius, fill=(20, 24, 34, 235),
    )
    # Couche haut (50% top, plus claire) — masquée par radius via clip
    # implicite du rounded_rectangle (le coin bas-droit reste droit, OK
    # car couvert visuellement par la couche bas dessous).
    half_h = inner_h // 2
    draw.rounded_rectangle(
        (x0, y0, x1, y0 + half_h), radius=radius, fill=(38, 44, 56, 235),
    )
    # Trait fin séparation diffuse (alpha bas) — adoucit la transition
    # entre les 2 demi-rects pour qu'on perçoive un gradient continu.
    for k in range(4):
        a = 60 - k * 12
        if a <= 0:
            break
        draw.line(
            (x0 + radius, y0 + half_h + k, x1 - radius, y0 + half_h + k),
            fill=(28, 32, 42, a), width=1,
        )

    # 3. Bevel haut : ligne 1px brillante juste sous le bord supérieur,
    # à l'intérieur de la carte → "lumière au-dessus".
    draw.line(
        (x0 + radius, y0 + 1, x1 - radius, y0 + 1),
        fill=(255, 255, 255, 90), width=1,
    )
    # 4. Bevel bas : ligne 1px sombre au ras du bord inférieur → "ombre
    # interne en bas". Ensemble = effet de plaque convexe.
    draw.line(
        (x0 + radius, y1 - 2, x1 - radius, y1 - 2),
        fill=(0, 0, 0, 130), width=1,
    )

    # 5. Texte — drop shadow d'abord (offset 2px sombre), puis blanc.
    tx = x0 + pad_x - bbox[0]
    ty = y0 + pad_y_top - bbox[1]
    draw.text((tx + 2, ty + 2), text, font=_FONT_CLOCK,
              fill=(0, 0, 0, 220))
    draw.text((tx, ty), text, font=_FONT_CLOCK,
              fill=(245, 248, 255, 255))
    return img


def _clock_base(text: str) -> np.ndarray:
    """Rend la plaque chrono BGRA prémultipliée NON warpée. Cache par
    text (seconde). Invalide le cache de warp quand la base change."""
    global _CLOCK_BASE_CACHE
    if _CLOCK_BASE_CACHE is not None and _CLOCK_BASE_CACHE[0] == text:
        return _CLOCK_BASE_CACHE[1]
    card = _render_clock_card(text)
    arr = np.array(card)
    bgra = arr[:, :, [2, 1, 0, 3]].copy()
    a = bgra[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    bgra[:, :, :3] = (bgra[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    _CLOCK_BASE_CACHE = (text, bgra)
    _CLOCK_WARP_CACHE.clear()
    return bgra


def render_clock_overlay(now_s: float) -> tuple[np.ndarray, int]:
    """Retourne (BGRA warpée, y_offset_anim) pour le timestamp donné.
    yaw oscille en sin autour de CLOCK_YAW_DEG ± CLOCK_YAW_AMP_DEG sur la
    période CLOCK_ORBIT_PERIOD_S. Bob vertical sin en quadrature pour ne
    pas être en phase avec le yaw → mouvement organique non-mécanique."""
    text = time.strftime("%H:%M:%S", time.localtime(now_s))
    base = _clock_base(text)
    # Phase orbite (s) au sein de la période courante.
    phase = (now_s % _CLOCK_ORBIT_PERIOD_S) / _CLOCK_ORBIT_PERIOD_S
    yaw = _CLOCK_YAW_DEG + _CLOCK_YAW_AMP_DEG * math.sin(
        2.0 * math.pi * phase
    )
    # Bob vertical en quadrature (cos = sin décalé π/2).
    y_off = int(round(_CLOCK_Y_BOB_AMP_PX * math.cos(
        2.0 * math.pi * phase
    )))
    if abs(yaw) < 0.5:
        return base, y_off
    bucket = int(round(yaw))
    cached = _CLOCK_WARP_CACHE.get(bucket)
    if cached is None:
        cached = _orbital_warp(base, float(bucket))
        _CLOCK_WARP_CACHE[bucket] = cached
    return cached, y_off


def overlay_clock(frame_bgr: np.ndarray) -> None:
    """Composite la plaque chrono animée dans le coin de la frame
    in-place. Position contrôlée par CLOCK_POSITION env :
    bl (default), br, tl, tr. Mini-animation slide-down + fade-in à
    chaque tick seconde (ease-out cubic)."""
    if not _CLOCK_ENABLED:
        return
    global _CLOCK_TICK_INSTANT, _CLOCK_LAST_TEXT
    now_s = time.time()
    text = time.strftime("%H:%M:%S", time.localtime(now_s))
    now_mono = time.monotonic()
    if text != _CLOCK_LAST_TEXT:
        _CLOCK_LAST_TEXT = text
        _CLOCK_TICK_INSTANT = now_mono
    card, y_bob = render_clock_overlay(now_s)
    h, w = frame_bgr.shape[:2]
    ch, cw = card.shape[:2]
    pos = env("CLOCK_POSITION", "bl").lower()
    if pos == "br":
        x = w - cw - _CLOCK_MARGIN_PX
        y = h - ch - _CLOCK_MARGIN_PX
    elif pos == "tl":
        x = _CLOCK_MARGIN_PX
        y = _CLOCK_MARGIN_PX
    elif pos == "tr":
        x = w - cw - _CLOCK_MARGIN_PX
        y = _CLOCK_MARGIN_PX
    else:  # bl par défaut
        x = _CLOCK_MARGIN_PX
        y = h - ch - _CLOCK_MARGIN_PX

    # Calcul progression anim tick. p ∈ [0,1] ; ease-out cubic.
    if (_CLOCK_TICK_DURATION_S > 0 and _CLOCK_TICK_INSTANT is not None):
        elapsed = now_mono - _CLOCK_TICK_INSTANT
        if elapsed < _CLOCK_TICK_DURATION_S:
            p = elapsed / _CLOCK_TICK_DURATION_S
            ease = 1.0 - (1.0 - p) ** 3  # ease-out cubic
            # Slide-down : démarre au-dessus de y_target, atterrit en y_target.
            y += -int(round(_CLOCK_TICK_SLIDE_PX * (1.0 - ease)))
            # Fade-in : RGB et alpha multipliés par ease (card premul donc
            # multiplie les 4 channels uniformément).
            faded = (card.astype(np.float32) * ease).astype(np.uint8)
            composite_bgra(frame_bgr, faded, x, y + y_bob)
            return

    composite_bgra(frame_bgr, card, x, y + y_bob)

# Drapeau nationalité (PNG pré-rendus par scripts/download_flags.py dans
# <FLAGS_DIR>/<IOC3>.png). Cache module-level, lazy-load, downscale à
# _FLAG_TARGET_HW dans le label. Si manquant pour un code → pas de
# drapeau, juste les autres sections (bib/nom/team) rendus normalement.
_FLAGS_DIR        = Path(env("FLAGS_DIR", "/var/lib/face-recog/flags"))
# Source des portraits riders. Priorité PORTRAIT_PHOTOS_DIR (dataset
# face-recog dont l'index .npz a été construit — 100% des riders) puis
# fallback FACE_DB_DIR (ASO subset partiel ~37%).
_PORTRAIT_PHOTOS_DIR = Path(env("PORTRAIT_PHOTOS_DIR",
                                 "/home/ben/rider_photos"))
# Pré-warm cache portraits au boot pour éviter les spikes de disk-read
# lors du 1er render de chaque label. Coût boot ~3-5 s, gain runtime
# significatif.
PORTRAIT_PREWARM = bool(int(env("PORTRAIT_PREWARM", "1")))
# Pre-render TOUS les labels au boot (~15 ms × 198 = ~3 s boot, gain
# runtime énorme — la revote loop devient lookup-only).
LABEL_PRERENDER = bool(int(env("LABEL_PRERENDER", "1")))
# Cache global name → BGRA premul des labels pré-rendus. Vide tant que
# main() n'a pas tourné. NAME → ndarray.
_LABEL_PRERENDER_CACHE: dict = {}
_FLAG_TARGET_H    = int(env("FLAG_TARGET_H", "22"))  # px dans le label
_FLAG_CACHE: dict[str, "Image.Image | None"] = {}


_PORTRAIT_CACHE: dict[str, "Image.Image | None"] = {}
_PORTRAIT_TARGET_H = int(env("PORTRAIT_TARGET_H", "38"))  # px (carré)


def _load_portrait(uciid: str | None,
                   target_h: int | None = None) -> "Image.Image | None":
    """Retourne la photo portrait PIL Image RGBA carrée pour le rider, ou
    None si manquante. Charge en PIL pur (pas cv2.imread qui jette l'alpha)
    pour préserver le fond transparent des PNG ASO. target_h paramètre la
    taille de sortie ; cache par (uciid, target_h)."""
    if not uciid:
        return None
    h = target_h if target_h is not None else _PORTRAIT_TARGET_H
    key = f"{uciid}@{h}"
    if key in _PORTRAIT_CACHE:
        return _PORTRAIT_CACHE[key]
    # Cherche dans le dataset face-recog (100% des riders) puis fallback
    # face-db ASO (subset partiel ~37%). Renvoie la 1ère trouvée.
    folder = None
    for candidate_root in (_PORTRAIT_PHOTOS_DIR, Path(FACE_DB_DIR)):
        f = candidate_root / uciid
        if f.is_dir():
            folder = f
            break
    if folder is None:
        _PORTRAIT_CACHE[key] = None
        return None
    candidate = None
    # Priorité PNG (souvent avec alpha) puis fallback formats classiques.
    for ext in ("png", "PNG", "webp", "jpg", "jpeg", "JPG"):
        matches = sorted(folder.glob(f"*.{ext}"))
        if matches:
            candidate = matches[0]
            break
    if candidate is None:
        _PORTRAIT_CACHE[key] = None
        return None
    try:
        pil = Image.open(candidate).convert("RGBA")
        # Crop carré : center horizontal, TOP pour les photos portrait
        # (rider_photos ASO 660×1000 = full-body avec tête en haut → un
        # center-crop coupait la tête). Pour les photos paysage (rare)
        # on center-crop normalement.
        s = min(pil.width, pil.height)
        left = (pil.width - s) // 2
        if pil.height > pil.width:
            # Top-biased crop : on garde la tête + buste, on jette les jambes.
            top = 0
        else:
            top = (pil.height - s) // 2
        pil = pil.crop((left, top, left + s, top + s))
        pil = pil.resize((h, h), Image.LANCZOS)
        _PORTRAIT_CACHE[key] = pil
        return pil
    except Exception:
        _PORTRAIT_CACHE[key] = None
        return None


def _placeholder_rank_text(name: str) -> tuple[str, str]:
    """Placeholder rang + gap temps tant qu'on n'a pas de feed live. Stable
    par rider (hash du nom) pour des valeurs cohérentes frame-à-frame.
    Format : ("#16", "+12'55\""). À remplacer par une vraie source GC
    (JSON live ASO/PCS) quand dispo."""
    import hashlib
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    rank = (h % 198) + 1  # 1..198 (taille peloton TDF)
    if rank == 1:
        return "#1", "+0\""
    gap_total_s = (h >> 8) % 7200  # 0..2h, déterministe
    gap_m = gap_total_s // 60
    gap_s = gap_total_s % 60
    return f"#{rank}", f"+{gap_m}'{gap_s:02d}\""


def _load_flag(iso3: str | None) -> "Image.Image | None":
    """Retourne le drapeau PIL Image RGBA pour le code IOC, ou None.
    Cache positif et négatif (un manque ne re-tente pas chaque frame)."""
    if not iso3:
        return None
    key = iso3.upper()
    if key in _FLAG_CACHE:
        return _FLAG_CACHE[key]
    path = _FLAGS_DIR / f"{key}.png"
    if not path.is_file():
        _FLAG_CACHE[key] = None
        return None
    try:
        img = Image.open(path).convert("RGBA")
        # Downscale au target height en gardant ratio (ex: 40x30 → ~30x22).
        ratio = _FLAG_TARGET_H / img.height
        new_w = max(1, int(round(img.width * ratio)))
        img = img.resize((new_w, _FLAG_TARGET_H), Image.LANCZOS)
        _FLAG_CACHE[key] = img
        return img
    except Exception:
        _FLAG_CACHE[key] = None
        return None


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


# Map team_code → {name, riders: [{bib, lastname, firstname, lastnameshort}]}
# trié par bib. Alimenté depuis le JSON partants ASO au boot.
_TEAM_ROSTERS: dict[str, dict] = {}


def load_team_rosters(json_path: str | None) -> dict[str, dict]:
    """Parse le JSON partants ASO et renvoie {team_code → {name, riders}}.

    `riders` est trié par bib (asc), chaque entrée garde firstname,
    lastname, lastnameshort et bib (int|None). Sert au rendu du tableau
    "compo équipe" en mode podium.
    """
    if not json_path or not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[face-recog] partants JSON (rosters): échec lecture: {e}",
              file=sys.stderr, flush=True)
        return {}
    out: dict[str, dict] = {}
    for team in data.get("teams", []) or []:
        team_code = team.get("code") or ""
        if not team_code:
            continue
        team_name = team.get("name") or team_code
        riders: list[dict] = []
        for r in team.get("riders", []) or []:
            try:
                bib_val = int(r.get("bib")) if r.get("bib") is not None else None
            except (TypeError, ValueError):
                bib_val = None
            riders.append({
                "bib": bib_val,
                "firstname": r.get("firstname") or "",
                "lastname": r.get("lastname") or "",
                "lastnameshort": r.get("lastnameshort") or r.get("lastname") or "",
                "uciid": r.get("uciid") or "",
            })
        riders.sort(key=lambda x: (x["bib"] is None, x["bib"] or 0))
        out[team_code] = {"name": team_name, "riders": riders}
    return out


class PodiumState:
    """Toggle thread-safe pour l'affichage du tableau compo équipe.

    Lu par draw_tracks (publish loop) et écrit par les requêtes HTTP
    POST /podium-mode. Pas d'EMA / d'historique : le seul état est le
    bool. La sélection du rider central (et son sticky) vit dans une
    structure séparée côté draw_tracks pour ne pas mélanger les
    préoccupations.
    """

    def __init__(self, initial: bool = False) -> None:
        self._lock = threading.Lock()
        self._enabled = bool(initial)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, v: bool) -> bool:
        with self._lock:
            self._enabled = bool(v)
            return self._enabled

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled


podium_state = PodiumState(initial=SHOW_TEAM_ROSTER)


class LayoutState:
    """État du layout des lower-thirds rider (toggle on/off).

    Modes :
      - "solo" : labels rider au-dessus de chaque visage (comportement
        par défaut, identique à avant l'introduction du toggle).
      - "none" : aucun label rider rendu sur la frame. Le tracking et
        la reco restent actifs (sélection rider central pour le tableau
        compo équipe continue de fonctionner).
    """

    _VALID = ("solo", "none")

    def __init__(self, initial: str = "solo") -> None:
        self._lock = threading.Lock()
        self._mode = initial if initial in self._VALID else "solo"

    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, v: str) -> str:
        with self._lock:
            if v in self._VALID:
                self._mode = v
            return self._mode

    def toggle(self) -> str:
        with self._lock:
            self._mode = "none" if self._mode == "solo" else "solo"
            return self._mode


LAYOUT_MODE_INITIAL = env("LAYOUT_MODE", "solo")
layout_state = LayoutState(initial=LAYOUT_MODE_INITIAL)


_LABEL_SIMPLE = bool(int(env("LABEL_SIMPLE", "0")))


def _render_lower_third_simple(name: str,
                                bib: int | None,
                                team_or_nat: str | None,
                                bib_confirmed: bool) -> np.ndarray:
    """Version studio-like : [bib | NAME | team/nat] 1 ligne, pas de
    portrait ni flag. Beaucoup moins de composite/render = équivalent au
    code studio mai 27. Sert quand LABEL_SIMPLE=1."""
    display = name.replace("_", " ")
    bbox_txt = _FONT.getbbox(display)
    text_w = bbox_txt[2] - bbox_txt[0]
    text_h = bbox_txt[3] - bbox_txt[1]

    bib_prefix = "• " if (bib is not None and bib_confirmed) else ""
    bib_str = f"{bib_prefix}{bib}" if bib is not None else ""
    nat_str = team_or_nat.upper() if team_or_nat else ""
    bib_text_w = (_FONT_SMALL.getbbox(bib_str)[2]
                  - _FONT_SMALL.getbbox(bib_str)[0]) if bib_str else 0
    nat_text_w = (_FONT_SMALL.getbbox(nat_str)[2]
                  - _FONT_SMALL.getbbox(nat_str)[0]) if nat_str else 0

    pad_x = 16
    pad_y_top = 8
    pad_y_bottom = 8
    radius = 14
    arrow_w = 12
    arrow_h = 10
    section_gap = 12
    sep_w = 1

    sections_w = text_w
    if bib_text_w:
        sections_w += bib_text_w + section_gap + sep_w + section_gap
    if nat_text_w:
        sections_w += section_gap + sep_w + section_gap + nat_text_w

    w = sections_w + 2 * pad_x
    h_label = pad_y_top + text_h + pad_y_bottom
    h = h_label + arrow_h

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((1, 2, w - 1, h_label),
                           radius=radius, fill=(0, 0, 0, 60))
    draw.rounded_rectangle((0, 0, w - 2, h_label - 1),
                           radius=radius, fill=(15, 18, 26, 140))

    cursor_x = pad_x
    name_text_y = pad_y_top - 2
    small_text_y = pad_y_top + (text_h - _FONT_SMALL_SIZE) // 2 - 1
    sep_top = pad_y_top + 2
    sep_bot = h_label - pad_y_bottom - 2

    if bib_text_w:
        draw.text((cursor_x, small_text_y), bib_str, font=_FONT_SMALL,
                  fill=(235, 235, 240, 255))
        cursor_x += bib_text_w + section_gap
        draw.rectangle((cursor_x, sep_top, cursor_x + sep_w, sep_bot),
                       fill=(255, 255, 255, 90))
        cursor_x += sep_w + section_gap

    draw.text((cursor_x + 1, name_text_y + 1), display, font=_FONT,
              fill=(0, 0, 0, 200))
    draw.text((cursor_x, name_text_y), display, font=_FONT,
              fill=(255, 255, 255, 255))
    cursor_x += text_w

    if nat_text_w:
        cursor_x += section_gap
        draw.rectangle((cursor_x, sep_top, cursor_x + sep_w, sep_bot),
                       fill=(255, 255, 255, 90))
        cursor_x += sep_w + section_gap
        draw.text((cursor_x, small_text_y), nat_str, font=_FONT_SMALL,
                  fill=(220, 220, 230, 240))

    arrow_top = h_label - 1
    cx_arrow = (w - 2) // 2
    draw.polygon([
        (cx_arrow - arrow_w // 2, arrow_top),
        (cx_arrow + arrow_w // 2, arrow_top),
        (cx_arrow, arrow_top + arrow_h),
    ], fill=(15, 18, 26, 140))

    arr = np.array(img)
    bgra = arr[:, :, [2, 1, 0, 3]].copy()
    a = bgra[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    bgra[:, :, :3] = (bgra[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    return bgra


def render_lower_third(name: str, score: float,
                       bib: int | None = None,
                       nationality: str | None = None,
                       bib_confirmed: bool = False,
                       country_iso3: str | None = None,
                       uciid: str | None = None,
                       team_name: str | None = None) -> np.ndarray:
    if _LABEL_SIMPLE:
        return _render_lower_third_simple(
            name, bib,
            nationality,  # = team_code passé par le caller
            bib_confirmed,
        )
    """Rend un label broadcast lower-third 2 lignes pour un nom donné.

    Layout :
      ┌─────────┬──────────────────────────┐
      │         │ bib  NAME            🇫🇷 │
      │ PHOTO   ├──────────────────────────┤
      │         │ Team Name      #16 +0'30"│
      └─────────┴──────────────────────────┘
              ▼ (flèche vers le visage)

    Ligne 1 : bib (petit) + NAME (gros) + drapeau (à droite)
    Ligne 2 : nom équipe complet (gauche) + classement+gap (droite, or)
    Photo : pleine hauteur du label, carrée.

    `nationality` est sémantiquement le team_code dans les appels actuels
    (libre placeholder côté gauche : ce param est inert dans la nouvelle
    version 2-lignes, conservé pour compat de signature).
    """
    display = name.replace("_", " ")
    _ = nationality  # kept for caller compat, unused (was team_code 3-letter)

    # ── Mesures texte ──
    bbox_name = _FONT.getbbox(display)
    name_w = bbox_name[2] - bbox_name[0]
    name_h = bbox_name[3] - bbox_name[1]

    bib_prefix = "• " if (bib is not None and bib_confirmed) else ""
    bib_str = f"{bib_prefix}{bib}" if bib is not None else ""
    bib_w = (_FONT_SMALL.getbbox(bib_str)[2]
             - _FONT_SMALL.getbbox(bib_str)[0]) if bib_str else 0

    team_str = (team_name or "").strip()
    team_w = (_FONT_SMALL.getbbox(team_str)[2]
              - _FONT_SMALL.getbbox(team_str)[0]) if team_str else 0

    rank_str, gap_str = _placeholder_rank_text(name)
    rank_text = f"{rank_str} {gap_str}"
    rank_w = (_FONT_SMALL.getbbox(rank_text)[2]
              - _FONT_SMALL.getbbox(rank_text)[0])

    # ── Drapeau (taille fixe par le loader) ──
    flag_img = _load_flag(country_iso3)
    flag_w = flag_img.width if flag_img is not None else 0

    # ── Constantes layout ──
    pad_x          = 16
    pad_y_top      = 10
    pad_y_bottom   = 10
    pad_inter      = 5  # gap entre ligne 1 et ligne 2
    radius         = 14
    arrow_w        = 12
    arrow_h        = 10
    section_gap    = 10
    sep_w          = 1

    # ── Hauteurs de lignes ──
    line1_h = max(name_h, flag_img.height if flag_img else 0,
                  _FONT_SMALL_SIZE)
    line2_h = _FONT_SMALL_SIZE + 2
    content_h = line1_h + pad_inter + line2_h

    # ── Portrait à hauteur du bloc texte (carré). ──
    portrait = _load_portrait(uciid, target_h=content_h)
    portrait_w = portrait.width if portrait is not None else 0

    # ── Largeurs des 2 lignes pour caler le bloc texte ──
    line1_inner_w = name_w
    if bib_w:
        line1_inner_w += bib_w + section_gap
    if flag_w:
        line1_inner_w += section_gap + flag_w

    # Ligne 2 : team gauche, rank droite, gap minimal entre les 2 si
    # les 2 sont présents.
    if team_w and rank_w:
        line2_inner_w = team_w + section_gap * 2 + rank_w
    else:
        line2_inner_w = team_w + rank_w

    text_block_w = max(line1_inner_w, line2_inner_w)

    # ── Largeur totale de la carte ──
    sections_w = text_block_w
    if portrait_w:
        sections_w += portrait_w + section_gap + sep_w + section_gap

    w = sections_w + 2 * pad_x
    h_label = pad_y_top + content_h + pad_y_bottom
    h = h_label + arrow_h

    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Fond principal (ombre + carte) ──
    draw.rounded_rectangle((1, 2, w - 1, h_label),
                           radius=radius, fill=(0, 0, 0, 60))
    draw.rounded_rectangle((0, 0, w - 2, h_label - 1),
                           radius=radius, fill=(15, 18, 26, 140))

    cursor_x = pad_x
    sep_top = pad_y_top + 2
    sep_bot = h_label - pad_y_bottom - 2

    # 1. Portrait full-height à gauche + séparateur. Paste avec masque
    # alpha (portrait est RGBA) → le fond transparent des PNG ASO laisse
    # voir le fond dark du label, pas de carré greenish/blanc.
    if portrait is not None:
        portrait_y = pad_y_top + (content_h - portrait.height) // 2
        img.paste(portrait, (cursor_x, portrait_y), portrait)
        cursor_x += portrait.width + section_gap
        draw.rectangle(
            (cursor_x, sep_top, cursor_x + sep_w, sep_bot),
            fill=(255, 255, 255, 90),
        )
        cursor_x += sep_w + section_gap

    text_block_x = cursor_x

    # ── Ligne 1 : bib | NAME | flag (centré verticalement sur line1) ──
    line1_y_center = pad_y_top + line1_h // 2

    cx = text_block_x
    if bib_w:
        bib_y = line1_y_center - _FONT_SMALL_SIZE // 2 - 1
        draw.text((cx, bib_y), bib_str, font=_FONT_SMALL,
                  fill=(235, 235, 240, 255))
        cx += bib_w + section_gap

    name_y = line1_y_center - name_h // 2 - 2
    # Drop shadow + NAME blanc.
    draw.text((cx + 1, name_y + 1), display, font=_FONT,
              fill=(0, 0, 0, 200))
    draw.text((cx, name_y), display, font=_FONT,
              fill=(255, 255, 255, 255))
    cx += name_w

    # Drapeau aligné à droite de la ligne 1 si possible, sinon directement
    # après le nom.
    if flag_img is not None:
        right_aligned_x = text_block_x + text_block_w - flag_img.width
        flag_x = max(cx + section_gap, right_aligned_x)
        flag_y = line1_y_center - flag_img.height // 2
        img.paste(flag_img, (flag_x, flag_y), flag_img)

    # ── Ligne 2 : team gauche + rank droite ──
    line2_y_center = pad_y_top + line1_h + pad_inter + line2_h // 2
    line2_text_y = line2_y_center - _FONT_SMALL_SIZE // 2 - 1
    line2_right_x = text_block_x + text_block_w

    if team_w:
        draw.text((text_block_x, line2_text_y), team_str,
                  font=_FONT_SMALL, fill=(200, 205, 220, 240))

    if rank_w:
        draw.text((line2_right_x - rank_w, line2_text_y), rank_text,
                  font=_FONT_SMALL, fill=(255, 220, 100, 255))

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
    bgra = arr[:, :, [2, 1, 0, 3]].copy()
    # Pré-multiplication alpha : RGB *= A/255, alpha inchangé. Indispensable
    # pour que cv2.warpPerspective (INTER_LINEAR) ne crée pas de frange
    # sombre aux bords arrondis — sans premul, les samples border (R=G=B=0,
    # A=0) tirent les pixels d'arête vers le noir tandis que l'alpha
    # diminue indépendamment, ce qui produit un halo qui "respire" en sync
    # avec l'oscillation yaw → flicker visible sur le contour de la box.
    # Avec premul, la même interpolation produit un dégradé propre.
    a = bgra[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    bgra[:, :, :3] = (bgra[:, :, :3].astype(np.float32) * a).astype(np.uint8)
    return bgra


def composite_bgra(frame_bgr: np.ndarray, label_bgra: np.ndarray,
                    x: int, y: int,
                    mask: "np.ndarray | None" = None) -> None:
    """Alpha-blend in-place du label_bgra sur frame_bgr à position (x, y).

    Suppose label_bgra à alpha PRÉ-MULTIPLIÉ (RGB déjà × A/255). Cf
    render_lower_third. Le compositing devient :
        dst = src_rgb + (1 - src_alpha) * dst
    (= "source over" classique en premul). Clip aux limites du frame.

    Si `mask` (uint8 0..255 même H,W que frame) est fourni, l'alpha du
    label est modulé par (1 - mask/255) : là où mask=255 (rider plein),
    le label devient invisible ; là où mask=0 (background), le label
    est compositée normalement. Single-pass sur la ROI du label
    seulement, donc ~20× moins de pixels que blender full-frame.
    """
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
    label_alpha_norm = label[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    if mask is not None:
        # Modulate label alpha by (1 - mask/255) → label "derrière" rider.
        mask_roi = mask[y0:y1, x0:x1].astype(np.float32) * (1.0 / 255.0)
        label_alpha_norm = label_alpha_norm * (1.0 - mask_roi[..., None])
    # `label[:, :, :3]` est pré-multiplié par alpha 0..1, donc on doit
    # le ré-échelle si on module l'alpha :
    #   contribution = label_rgb_premult * (label_alpha_eff / label_alpha_orig)
    # Plus simple : reconstruire le rgb non-premultiplié × nouvel alpha.
    if mask is not None:
        label_alpha_orig = label[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
        # Évite div par 0 quand label transparent
        safe_orig = np.maximum(label_alpha_orig, 1e-3)
        label_rgb_unpremult = label[:, :, :3].astype(np.float32) / safe_orig
        label_contrib = label_rgb_unpremult * label_alpha_norm
    else:
        label_contrib = label[:, :, :3].astype(np.float32)
    inv_a = 1.0 - label_alpha_norm
    roi[:] = (label_contrib + inv_a * roi).astype(np.uint8)


# ─────────────── Tableau "compo équipe" (mode podium) ──────────────────
# Cache des cartes pré-rendues par (team_code, highlight_bib). Invalide
# implicitement quand on touche aux env vars (= restart service).
_TEAM_ROSTER_CACHE: dict[tuple, np.ndarray] = {}


def _font_roster_body() -> ImageFont.FreeTypeFont:
    """Police corps du tableau (taille TEAM_ROSTER_FONT_SIZE)."""
    return ImageFont.truetype(_FONT_PATH, TEAM_ROSTER_FONT_SIZE)


def _font_roster_header() -> ImageFont.FreeTypeFont:
    """Police header (nom équipe), légèrement plus grosse."""
    return ImageFont.truetype(_FONT_PATH, TEAM_ROSTER_HEADER_FONT_SIZE)


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    if len(s) != 6:
        return (163, 113, 247)  # accent par défaut
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (163, 113, 247)


TEAM_ROSTER_PORTRAIT_H = int(env("TEAM_ROSTER_PORTRAIT_H", "78"))
TEAM_ROSTER_LOGO_H = int(env("TEAM_ROSTER_LOGO_H", "64"))


def _make_rider_cartouche_pil(r: dict, accent_outline: tuple | None,
                               highlight: bool) -> "Image.Image":
    """Mini lower-third individuel pour un rider : fond arrondi sombre,
    photo carrée à gauche, bib + Firstname LASTNAME complet à droite.
    Style strictement aligné sur render_lower_third (radius 14, drop
    shadow 1 px, fond (15,18,26,140)).
    """
    portrait_h = TEAM_ROSTER_PORTRAIT_H
    font_bib = _FONT_SMALL
    font_name = _FONT
    radius = 14
    pad = 10
    portrait_gap = 10

    ln = (r.get("lastname") or "").upper()
    fn = (r.get("firstname") or "").strip()
    name_str = f"{fn} {ln}" if (fn and ln) else (fn or ln)
    bib_val = r.get("bib")
    bib_str = f"{bib_val}" if isinstance(bib_val, int) else "·"

    dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dummy)
    bbox_bib = dd.textbbox((0, 0), bib_str, font=font_bib)
    bbox_name = dd.textbbox((0, 0), name_str, font=font_name)
    bib_w = bbox_bib[2] - bbox_bib[0]
    bib_h = bbox_bib[3] - bbox_bib[1]
    name_w = bbox_name[2] - bbox_name[0]
    name_h = bbox_name[3] - bbox_name[1]

    text_w = max(bib_w, name_w)
    inner_w = portrait_h + portrait_gap + text_w
    inner_h = portrait_h
    w = inner_w + pad * 2
    h_label = inner_h + pad * 2
    h = h_label + 2  # marge pour drop shadow

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((1, 2, w - 1, h_label),
                           radius=radius, fill=(0, 0, 0, 60))
    draw.rounded_rectangle((0, 0, w - 2, h_label - 1),
                           radius=radius, fill=(15, 18, 26, 140))

    # Portrait carré à gauche.
    px = pad
    py = pad
    portrait = _load_portrait(r.get("uciid") or None,
                               target_h=portrait_h)
    if portrait is not None:
        ph = portrait.height
        pw = portrait.width
        if ph != portrait_h or pw != portrait_h:
            portrait = portrait.resize((portrait_h, portrait_h),
                                        Image.Resampling.LANCZOS)
        img.paste(portrait, (px, py), portrait)
    else:
        draw.rectangle((px, py, px + portrait_h, py + portrait_h),
                       fill=(30, 34, 44, 220))
        initial = (fn or ln or "?")[0].upper()
        bbi = dd.textbbox((0, 0), initial, font=font_name)
        ix = px + (portrait_h - (bbi[2] - bbi[0])) // 2 - bbi[0]
        iy = py + (portrait_h - (bbi[3] - bbi[1])) // 2 - bbi[1]
        draw.text((ix, iy), initial, font=font_name,
                  fill=(180, 184, 196, 240))
    if accent_outline is not None:
        draw.rectangle(
            (px - 1, py - 1, px + portrait_h, py + portrait_h),
            outline=(accent_outline[0], accent_outline[1],
                      accent_outline[2], 255),
            width=2,
        )

    # Bib + nom à droite.
    tx = px + portrait_h + portrait_gap
    mid_y = py + portrait_h // 2
    bib_y = mid_y - bib_h - 4 - bbox_bib[1]
    name_y = mid_y + 2 - bbox_name[1]
    draw.text((tx, bib_y), bib_str, font=font_bib,
              fill=(235, 235, 240, 230))
    if highlight:
        draw.text((tx + 1, name_y + 1), name_str, font=font_name,
                  fill=(0, 0, 0, 200))
        draw.text((tx, name_y), name_str, font=font_name,
                  fill=(255, 255, 255, 255))
    else:
        draw.text((tx, name_y), name_str, font=font_name,
                  fill=(225, 227, 235, 240))
    return img


def _make_team_header_cartouche_pil(team_code: str,
                                     team_name: str) -> "Image.Image":
    """Cartouche header : logo équipe (PNG team-logos) + nom équipe.

    Même style que les cartouches rider (fond arrondi sombre, drop
    shadow). Si le logo est absent on rend juste le nom.
    """
    logo_h = TEAM_ROSTER_LOGO_H
    radius = 14
    pad = 12
    gap = 16
    font_hdr = _FONT

    # Charge le logo en alpha complet (sans la modulation 0.5 de
    # get_team_logo_sized qui est faite pour le rendu "derrière rider").
    logo_pil: "Image.Image | None" = None
    raw = _load_team_logo_raw(team_code)
    if raw is not None:
        bh, bw = raw.shape[:2]
        new_w = max(1, int(round(bw * logo_h / bh)))
        sized = cv2.resize(raw, (new_w, logo_h), interpolation=cv2.INTER_AREA)
        # BGRA → RGBA pour PIL.
        rgba = sized[:, :, [2, 1, 0, 3]].copy()
        logo_pil = Image.fromarray(rgba, mode="RGBA")

    name_str = (team_name or team_code).upper()
    dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dummy)
    bbox = dd.textbbox((0, 0), name_str, font=font_hdr)
    name_w = bbox[2] - bbox[0]
    name_h = bbox[3] - bbox[1]

    logo_w = logo_pil.width if logo_pil is not None else 0
    inner_w = logo_w + (gap if logo_pil is not None else 0) + name_w
    inner_h = max(logo_h, name_h + 4)
    w = inner_w + pad * 2
    h_label = inner_h + pad * 2
    h = h_label + 2

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((1, 2, w - 1, h_label),
                           radius=radius, fill=(0, 0, 0, 60))
    draw.rounded_rectangle((0, 0, w - 2, h_label - 1),
                           radius=radius, fill=(15, 18, 26, 160))

    cur_x = pad
    if logo_pil is not None:
        ly = pad + (inner_h - logo_h) // 2
        img.paste(logo_pil, (cur_x, ly), logo_pil)
        cur_x += logo_w + gap
    ny = pad + (inner_h - name_h) // 2 - bbox[1]
    draw.text((cur_x + 1, ny + 1), name_str, font=font_hdr,
              fill=(0, 0, 0, 200))
    draw.text((cur_x, ny), name_str, font=font_hdr,
              fill=(255, 255, 255, 255))
    return img


def render_team_roster_card(team_code: str,
                             highlight_bib: int | None) -> "np.ndarray | None":
    """Compose le tableau compo équipe : cartouche header (logo + nom)
    + 8 cartouches rider indépendants en grille 3-3-2 (3 colonnes).

    Chaque cartouche est un mini lower-third autonome (fond arrondi,
    drop shadow) pour un effet broadcast. Le rider courant
    (bib == highlight_bib) a un liseré accent sur sa photo + nom blanc.

    Renvoie BGRA pré-multiplié (compat composite_bgra). None si
    team_code inconnu ou roster vide.
    """
    team = _TEAM_ROSTERS.get(team_code)
    if not team:
        return None
    riders = team.get("riders") or []
    if not riders:
        return None

    cache_key = (team_code, highlight_bib)
    cached = _TEAM_ROSTER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    accent = _hex_to_rgb(TEAM_ROSTER_ACCENT_HEX)
    cols: list[list[dict]] = [riders[0:3], riders[3:6], riders[6:8]]

    # 1) Pré-rendu des cartouches rider individuels (par colonne).
    rider_cards: list[list["Image.Image"]] = []
    max_card_w = 0
    max_card_h = 0
    for col in cols:
        col_imgs = []
        for r in col:
            is_hl = (highlight_bib is not None
                     and isinstance(r.get("bib"), int)
                     and r["bib"] == highlight_bib)
            card = _make_rider_cartouche_pil(
                r, accent_outline=accent if is_hl else None,
                highlight=is_hl,
            )
            col_imgs.append(card)
            if card.width > max_card_w:
                max_card_w = card.width
            if card.height > max_card_h:
                max_card_h = card.height
        rider_cards.append(col_imgs)

    # 2) Pré-rendu du cartouche header (logo + nom équipe).
    team_name = team.get("name") or team_code
    header_card = _make_team_header_cartouche_pil(team_code, team_name)

    # 3) Layout global.
    col_gap = 16
    row_gap = 12
    header_gap = 16  # entre le header et la grille de riders
    rows = 3  # colonnes 1 et 2 ont 3 riders, colonne 3 en a 2
    grid_w = max_card_w * len(cols) + col_gap * (len(cols) - 1)
    grid_h = max_card_h * rows + row_gap * (rows - 1)
    canvas_w = max(grid_w, header_card.width)
    canvas_h = header_card.height + header_gap + grid_h

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    # Header centré.
    hx = (canvas_w - header_card.width) // 2
    canvas.alpha_composite(header_card, (hx, 0))

    # Grille rider, centrée si le header est plus large que la grille.
    grid_x0 = (canvas_w - grid_w) // 2
    grid_y0 = header_card.height + header_gap
    for ci, col_imgs in enumerate(rider_cards):
        col_x = grid_x0 + ci * (max_card_w + col_gap)
        for ri, card in enumerate(col_imgs):
            cy = grid_y0 + ri * (max_card_h + row_gap)
            canvas.alpha_composite(card, (col_x, cy))

    rgba = np.array(canvas, dtype=np.uint8)
    bgra = rgba[:, :, [2, 1, 0, 3]].copy()
    a = bgra[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    bgra[:, :, :3] = (bgra[:, :, :3].astype(np.float32) * a).astype(np.uint8)

    _TEAM_ROSTER_CACHE[cache_key] = bgra
    return bgra


# ──────────────────────── Team logos (rendu derrière riders) ─────────────
_team_logo_cache: dict = {}


def _load_team_logo_raw(team_code: str) -> "np.ndarray | None":
    """Charge le PNG logo équipe (BGRA). None si absent."""
    if not team_code:
        return None
    path = Path(TEAM_LOGOS_DIR) / f"{team_code}.png"
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        # PNG sans alpha → alpha 255 partout
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    return img


def get_team_logo_sized(team_code: str, target_h: int) -> "np.ndarray | None":
    """Renvoie le logo BGRA pre-multiplié alpha, redimensionné à target_h
    (largeur proportionnelle), avec TEAM_LOGO_ALPHA appliqué.

    Bucketise target_h par tranche de 32 pour limiter les entrées cache.
    Cache None pour les codes absents (évite re-lookup disque).
    """
    bucket_h = (target_h // 32) * 32 or 32
    key = (team_code, bucket_h)
    if key in _team_logo_cache:
        return _team_logo_cache[key]
    raw = _load_team_logo_raw(team_code)
    if raw is None:
        _team_logo_cache[key] = None
        return None
    h, w = raw.shape[:2]
    new_w = max(1, int(round(w * bucket_h / h)))
    sized = cv2.resize(raw, (new_w, bucket_h), interpolation=cv2.INTER_AREA)
    # Module l'alpha par TEAM_LOGO_ALPHA puis pré-multiplie RGB par alpha
    # (composite_bgra attend de la pré-mul).
    a = sized[:, :, 3].astype(np.float32) * (TEAM_LOGO_ALPHA / 255.0)  # 0..TEAM_LOGO_ALPHA
    sized[:, :, 3] = (a * 255.0).clip(0, 255).astype(np.uint8)
    a3 = a[..., None]
    sized[:, :, :3] = (sized[:, :, :3].astype(np.float32) * a3).clip(0, 255).astype(np.uint8)
    _team_logo_cache[key] = sized
    return sized


_scene_logo_cache: dict = {}


def _load_scene_logo_sized(path: str, target_h: int, alpha: float) -> "np.ndarray | None":
    """Charge + dimensionne + pré-multiplie alpha le logo scene. Cache par
    (path, mtime, target_h, alpha) pour éviter de re-charger 60×/s."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (path, mtime, target_h, alpha)
    if key in _scene_logo_cache:
        return _scene_logo_cache[key]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        _scene_logo_cache[key] = None
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        a = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, a])
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * target_h / h)))
    # INTER_AREA optimal pour downscale, LANCZOS4 pour upscale (préserve
    # les bords nets quand on agrandit un petit logo). Branchement
    # automatique selon target.
    interp = cv2.INTER_AREA if target_h <= h else cv2.INTER_LANCZOS4
    sized = cv2.resize(img, (new_w, target_h), interpolation=interp)
    # Apply global alpha + pre-mul RGB
    a = sized[:, :, 3].astype(np.float32) * (alpha / 255.0)
    sized[:, :, 3] = (a * 255.0).clip(0, 255).astype(np.uint8)
    a3 = a[..., None]
    sized[:, :, :3] = (sized[:, :, :3].astype(np.float32) * a3).clip(0, 255).astype(np.uint8)
    _scene_logo_cache[key] = sized
    return sized


def _compute_chroma_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Calcule un mask uint8 (H, W) compatible composite_bgra(mask=...) :
    0 où le pixel est proche de la couleur clé (= fond, logo OK), 255
    sinon (= foreground, logo caché). Ramp linéaire sur la zone de
    transition pour des bords lisses, pas de stairstep.
    """
    try:
        b, g, r = (int(v) for v in SCENE_LOGO_KEY_BGR.split(","))
    except (ValueError, AttributeError):
        b, g, r = 60, 60, 60
    target = np.array([b, g, r], dtype=np.float32)
    diff = frame_bgr.astype(np.float32) - target
    dist = np.sqrt((diff * diff).sum(axis=2))
    inner = SCENE_LOGO_KEY_TOLERANCE
    outer = inner + max(1.0, SCENE_LOGO_KEY_SOFTNESS)
    mask = ((dist - inner) / (outer - inner) * 255.0).clip(0, 255).astype(np.uint8)
    return mask


def draw_scene_logo_behind(frame: np.ndarray,
                            mask: "np.ndarray | None") -> None:
    """Composite un logo générique à position fixe (X/Y_FRACTION du
    frame).

    Sélection du mask de découpe :
      - SCENE_LOGO_CHROMA_KEY=1 → mask calculé par proximité de couleur
        au fond (key BGR). Robuste pour les plateaux/scènes où RVM ne
        détecte pas tout (présentateurs en costume, etc.) car il suffit
        que le fond ait une couleur uniforme.
      - sinon → utilise le mask RVM passé en paramètre.
    """
    if not SCENE_LOGO_BEHIND:
        return
    logo = _load_scene_logo_sized(SCENE_LOGO_PATH, SCENE_LOGO_HEIGHT, SCENE_LOGO_ALPHA)
    if logo is None:
        return
    # Si chroma key actif ET mask RVM dispo : UNION des deux (= cache
    # le logo dès que l'un OU l'autre détecte du foreground). Ça
    # rattrape les vêtements proches du gris (RVM les voit) et les
    # objets non-humains (chroma les voit).
    if SCENE_LOGO_CHROMA_KEY:
        chroma = _compute_chroma_mask(frame)
        if mask is not None and mask.shape == chroma.shape:
            effective_mask = np.maximum(chroma, mask)
        else:
            effective_mask = chroma
    else:
        effective_mask = mask
    H, W = frame.shape[:2]
    lh, lw = logo.shape[:2]
    cx = int(W * SCENE_LOGO_X_FRACTION)
    cy = int(H * SCENE_LOGO_Y_FRACTION)
    composite_bgra(frame, logo, cx - lw // 2, cy - lh // 2, mask=effective_mask)


def draw_team_logos_behind(frame: np.ndarray, tracks,
                            mask: "np.ndarray | None") -> None:
    """Composite UN logo par ÉQUIPE (pas par rider), à position FIXE
    sur l'écran (= ancré au fond, pas au rider). Si plusieurs équipes
    identifiées, les logos sont alignés en row centré horizontalement,
    avec TEAM_LOGO_SPACING_PX entre eux. composite_bgra(mask=...) cache
    le logo là où le mask dit "rider" → l'effet "logo sur fond gris,
    coureur devant" est obtenu sans suivre les bboxes.
    """
    if not TEAM_LOGO_BEHIND:
        return
    # Liste ordonnée des team_codes uniques détectés (skip les
    # non-identifiés et ceux sans label rendu). Ordre déterministe =
    # tri alphabétique pour stabilité visuelle d'un frame à l'autre.
    seen: set[str] = set()
    team_codes: list[str] = []
    for t in tracks:
        team_code = getattr(t, "team_code", None)
        if not team_code or t.label_img is None:
            continue
        if team_code in seen:
            continue
        seen.add(team_code)
        team_codes.append(team_code)
    if not team_codes:
        return
    team_codes.sort()

    # Récupère les logos dimensionnés (skip ceux sans fichier dispo).
    logos = []
    for code in team_codes:
        lg = get_team_logo_sized(code, TEAM_LOGO_HEIGHT)
        if lg is not None:
            logos.append(lg)
    if not logos:
        return

    H, W = frame.shape[:2]
    total_w = sum(l.shape[1] for l in logos) + TEAM_LOGO_SPACING_PX * (len(logos) - 1)
    cur_x = (W - total_w) // 2
    cy = int(H * TEAM_LOGO_Y_FRACTION)
    for lg in logos:
        lh, lw = lg.shape[:2]
        composite_bgra(frame, lg, cur_x, cy - lh // 2, mask=mask)
        cur_x += lw + TEAM_LOGO_SPACING_PX


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
        # NOTE historique : ce champ a contenu pendant un temps le team_code
        # (étiquette de droite du label). Depuis l'ajout du drapeau on a
        # un champ dédié `team_code` ci-dessous, et `nationality` retrouve
        # son sens d'origine (code pays IOC, sert au lookup drapeau).
        self.nationality: str | None = None
        self.team_code: str | None = None
        self.team_name: str | None = None
        # Résolu à la première (re-)vote en fallback manifest puis
        # partants. Sert au lookup photo portrait (render_lower_third)
        # et à la vignette skeleton view.
        self.uciid: str | None = None
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
        # Cache du warp pseudo-3D par bucket d'angle yaw. Invalidé par
        # identité quand label_img est remplacé (re-render après re-vote
        # ou bib OCR confirm). Le yaw oscille lentement (~0.35°/frame),
        # cache plein après une orbite (~6s) puis 100% hit ensuite.
        self._orbit_cache_for: np.ndarray | None = None
        self._orbit_cache: dict[int, np.ndarray] = {}
        # Niveau d'empilement vertical (0 = au-dessus du visage à la
        # distance naturelle, +1 = 1 cran plus haut, etc.). Latché avec
        # hystérésis : on monte vite quand une collision apparaît, on
        # descend après _LABEL_DETACH_FRAMES frames consécutives sans
        # besoin. X du label reste TOUJOURS centré sur le visage propre,
        # seul Y bouge — chaque label reste lisiblement "sur son visage".
        self.stack_level: int = 0
        self.detach_countdown: int = 0
        # Compteur consécutif "perdant en dédup IoU" : incrémenté chaque
        # frame où ce track est éjecté du rendu par overlap fort avec un
        # track au meilleur score. Latché à _LABEL_IOU_SUPPRESS_FRAMES
        # avant suppression réelle, pour éviter le flicker quand 2 tracks
        # voisins en peloton dense échangent fréquemment la palme du score.
        self.iou_losing_streak: int = 0
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
_LABEL_DETACH_FRAMES = int(env("LABEL_DETACH_FRAMES", "30"))
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
# Seuil IoU pour considérer 2 tracks comme le même visage physique
# (= dédup spatial). 0.7 = overlap franc requis ; sous ce seuil les
# 2 labels coexistent (le stack vertical s'occupe du visuel).
# Historique : 0.5 flicker en peloton dense, 2 scores ArcFace voisins
# swappaient l'élection à chaque re-vote asynchrone.
_LABEL_IOU_DEDUP = float(env("LABEL_IOU_DEDUP", "0.7"))
# Latch : un track sort vraiment du rendu après ce nombre de frames
# consécutives perdantes en dédup IoU. En-deçà, il reste affiché —
# au prix d'un stack temporaire que les autres mécanismes amortissent.
# À 60 fps = 0.33s : assez pour qu'une vraie duplication d'identité
# soit confirmée, tout en absorbant les swaps ponctuels de score.
_LABEL_IOU_SUPPRESS_FRAMES = int(env("LABEL_IOU_SUPPRESS_FRAMES", "20"))

# ── Pseudo-3D billboard (titrage 3D qui orbite, effet broadcast pro) ──
# Toggle env var : 0 = label 2D classique.
PSEUDO_3D = bool(int(env("PSEUDO_3D", "1")))
# Période d'un cycle complet d'orbite (sec). Chaque label a sa phase
# aléatoire pour ne pas tous bouger en sync.
_ORBIT_PERIOD_S = float(env("ORBIT_PERIOD_S", "6.0"))
# Amplitude max du yaw (degrés). 15-25 = effet visible, > 40 = caricature.
_ORBIT_YAW_AMP_DEG = float(env("ORBIT_YAW_AMP_DEG", "20.0"))
# Pas de quantification du yaw pour le cache de warp. À 1° on a ~41
# warps uniques par track sur tout l'orbit ; visuellement le pas est
# indiscernable (< 0.5 px de différence sur la largeur du label). Cache
# évite de relancer cv2.warpPerspective chaque frame × chaque track,
# coût dominant quand le peloton dépasse 10 visages.
_ORBIT_YAW_BUCKET_DEG = float(env("ORBIT_YAW_BUCKET_DEG", "1.0"))


def _rect_collide(a, b) -> bool:
    """Intersection rectangulaire stricte. a, b = (x0, y0, x1, y1)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _orbital_warp_cached(t, yaw_deg: float) -> np.ndarray:
    """Wrapper de _orbital_warp avec cache par track + bucket d'angle.

    Le yaw oscille continûment (sin) sur ORBIT_PERIOD_S ; à 60 fps c'est
    ~0.35°/frame en peak rate. Avec un bucket de 1°, ~3 frames consécutives
    partagent le même warp → ~70% hit ratio dès la 1ère orbite, 100%
    ensuite. Coût warpPerspective ÷ ~3-30 selon la densité de peloton.
    Invalidé par identité de t.label_img : tout re-render du label
    (re-vote, bib confirm, index reload) reset le cache.
    """
    if t._orbit_cache_for is not t.label_img:
        t._orbit_cache_for = t.label_img
        t._orbit_cache.clear()
    bucket = int(round(yaw_deg / _ORBIT_YAW_BUCKET_DEG))
    cached = t._orbit_cache.get(bucket)
    if cached is None:
        cached = _orbital_warp(t.label_img,
                               bucket * _ORBIT_YAW_BUCKET_DEG)
        t._orbit_cache[bucket] = cached
    return cached


# State persistant du dédup par nom latché : name → (track_id keeper,
# score latché). Vidé pour les noms qui sortent de l'écran (évite
# croissance infinie). Marge nécessaire au challenger pour prendre la
# place du keeper actuel.
_NAME_KEEPER: dict[str, tuple[int, float]] = {}
_NAME_KEEPER_MARGIN = float(env("NAME_KEEPER_MARGIN", "0.05"))


def _dedup_by_name_latched(placeable):
    """Garde 1 seul track par nom (le keeper latché). Le keeper ne change
    que si un challenger même-nom dépasse son score de +MARGIN, sinon
    on conserve le keeper même quand un concurrent grimpe transitoirement.
    Si le keeper sort de placeable (track mort/perdu), on élit le meilleur
    score parmi les candidats restants."""
    by_name: dict[str, list] = {}
    for tr in placeable:
        by_name.setdefault(tr.name, []).append(tr)

    kept = []
    seen_names = set()
    for name, candidates in by_name.items():
        seen_names.add(name)
        if len(candidates) == 1:
            keeper = candidates[0]
            _NAME_KEEPER[name] = (keeper.id, keeper.score)
            kept.append(keeper)
            continue
        # Cherche le keeper actuel parmi les candidats présents.
        current = _NAME_KEEPER.get(name)
        keeper = None
        if current is not None:
            keeper_id, _ = current
            for tr in candidates:
                if tr.id == keeper_id:
                    keeper = tr
                    break
        if keeper is None:
            # Pas de keeper (mort ou jamais élu) → élit le meilleur score.
            keeper = max(candidates, key=lambda t: t.score)
        else:
            # Keeper toujours là : vérifie si un challenger domine de
            # +MARGIN. Sinon on garde.
            challenger = max(
                (t for t in candidates if t.id != keeper.id),
                key=lambda t: t.score, default=None,
            )
            if (challenger is not None
                    and challenger.score >= keeper.score
                                              + _NAME_KEEPER_MARGIN):
                keeper = challenger
        _NAME_KEEPER[name] = (keeper.id, keeper.score)
        kept.append(keeper)

    # GC du state pour les noms qui ne sont plus à l'écran (sinon le dict
    # croît à vie). On garde simple : tout nom absent de cette frame est
    # purgé, même s'il revient plus tard (perte du latch). Compromis OK
    # vu que le re-vote suivant ré-initialisera proprement.
    for stale in list(_NAME_KEEPER.keys()):
        if stale not in seen_names:
            del _NAME_KEEPER[stale]
    return kept


# Scratch buffer (H, W) uint8 pour le mask chroma du tableau podium :
# alloué une seule fois à la résolution frame, réutilisé à chaque
# frame. On n'écrit que la ROI du tableau, le reste n'est jamais lu par
# composite_bgra (il indexe mask[y0:y1, x0:x1] uniquement). Évite
# l'alloc 2 MB/frame qui massacrait les fps.
_PODIUM_CHROMA_SCRATCH: "np.ndarray | None" = None


# Sticky de l'équipe affichée dans le tableau compo. L'équipe
# affichée = celle la plus représentée parmi les riders reconnus à
# l'écran. On ne change qu'après TEAM_ROSTER_STICKY_S consécutives où
# une autre équipe est majoritaire, sinon on garde la dernière. Si
# plus aucun rider reconnu depuis TEAM_ROSTER_IDLE_HIDE_S, la carte
# disparaît.
#
# Position d'affichage lissée (EMA + deadzone + snap) sur le même
# modèle que les labels rider : `display_x` / `display_y` flottent
# vers la target avec _LABEL_SMOOTH_ALPHA ; on ignore les variations
# < _LABEL_TARGET_DEADZONE_PX (anti-jitter) ; on snap au-delà de
# _LABEL_SNAP_PX (changement de plan / re-id). Tue le "le tableau qui
# bouge à chaque frame quand un rider lève la tête".
_PODIUM_TEAM: dict = {
    "team_code": None,
    "highlight_bib": None,
    "last_seen_s": 0.0,
    "challenger": None,
    "challenger_since": 0.0,
    "display_x": None,
    "display_y": None,
}


def _pick_team_majority(placeable, now_s):
    """Met à jour _PODIUM_TEAM et retourne (team_code, highlight_bib).

    Compte les apparitions par team_code parmi les tracks reconnus dans
    _RIDERS_META. Le sticky empêche le ping-pong : on ne switche que si
    le challenger reste majoritaire ≥ TEAM_ROSTER_STICKY_S secondes.
    highlight_bib = bib d'un rider visible de l'équipe affichée (le
    premier rencontré → suffit pour signaler visuellement qui est en
    cadre).
    """
    counts: dict[str, int] = {}
    bib_by_team: dict[str, int] = {}
    for t in placeable:
        name = getattr(t, "name", None)
        if not name:
            continue
        meta = _RIDERS_META.get(name)
        if not meta:
            continue
        tc = meta.get("team_code") or ""
        if not tc:
            continue
        counts[tc] = counts.get(tc, 0) + 1
        if tc not in bib_by_team and isinstance(meta.get("bib"), int):
            bib_by_team[tc] = meta["bib"]
    if counts:
        _PODIUM_TEAM["last_seen_s"] = now_s
        top_team, _ = max(counts.items(), key=lambda kv: kv[1])
        cur = _PODIUM_TEAM["team_code"]
        if cur is None or cur not in counts:
            # Pas de keeper valide → adopter le top direct.
            _PODIUM_TEAM["team_code"] = top_team
            _PODIUM_TEAM["challenger"] = None
        elif top_team != cur:
            # Challenger : doit persister STICKY_S avant de prendre la main.
            if _PODIUM_TEAM["challenger"] != top_team:
                _PODIUM_TEAM["challenger"] = top_team
                _PODIUM_TEAM["challenger_since"] = now_s
            elif now_s - _PODIUM_TEAM["challenger_since"] >= TEAM_ROSTER_STICKY_S:
                _PODIUM_TEAM["team_code"] = top_team
                _PODIUM_TEAM["challenger"] = None
        else:
            _PODIUM_TEAM["challenger"] = None
        cur = _PODIUM_TEAM["team_code"]
        _PODIUM_TEAM["highlight_bib"] = bib_by_team.get(cur)
    return _PODIUM_TEAM["team_code"], _PODIUM_TEAM["highlight_bib"]


def _overlay_team_roster(frame: np.ndarray, placeable,
                         mask: "np.ndarray | None") -> None:
    """Compose le tableau compo équipe au-dessus de la tête des riders.

    Position : X centré horizontalement dans le frame, Y = sommet des
    têtes reconnues (= min y1 parmi les bbox riders connus) − hauteur
    carte − gap. Si aucun rider n'est positionnable, fallback Y =
    TEAM_ROSTER_TOP_Y_FRACTION × hauteur frame.

    L'équipe affichée est la plus représentée à l'écran (sticky 2s).
    Disparaît si plus aucun rider reconnu depuis TEAM_ROSTER_IDLE_HIDE_S.
    """
    if not podium_state.is_enabled():
        _PODIUM_TEAM["display_x"] = None
        _PODIUM_TEAM["display_y"] = None
        return
    if not _TEAM_ROSTERS:
        return
    now_s = time.monotonic()
    team_code, highlight_bib = _pick_team_majority(placeable, now_s)
    if not team_code:
        _PODIUM_TEAM["display_x"] = None
        _PODIUM_TEAM["display_y"] = None
        return
    if TEAM_ROSTER_IDLE_HIDE_S > 0:
        idle = now_s - _PODIUM_TEAM["last_seen_s"]
        if idle > TEAM_ROSTER_IDLE_HIDE_S:
            _PODIUM_TEAM["display_x"] = None
            _PODIUM_TEAM["display_y"] = None
            return
    card = render_team_roster_card(team_code, highlight_bib)
    if card is None:
        return
    ch, cw = card.shape[:2]
    fh, fw = frame.shape[:2]
    target_x = (fw - cw) // 2
    # Y cible = au-dessus du sommet le plus haut parmi les riders reconnus.
    top_y = None
    for t in placeable:
        name = getattr(t, "name", None)
        if not name or name not in _RIDERS_META:
            continue
        x1, y1, x2, y2 = t.bbox_xyxy()
        y1 = int(y1)
        if top_y is None or y1 < top_y:
            top_y = y1
    if top_y is None:
        target_y = int(fh * TEAM_ROSTER_TOP_Y_FRACTION)
    else:
        target_y = top_y - TEAM_ROSTER_GAP_PX - ch
    # Clamping aux bords du frame avant lissage (= sinon EMA "tire"
    # vers une position interdite et on voit le tableau migrer).
    target_y = max(TEAM_ROSTER_MARGIN_PX, target_y)
    target_x = max(TEAM_ROSTER_MARGIN_PX, target_x)
    target_x = min(fw - TEAM_ROSTER_MARGIN_PX - cw, target_x)

    # Smoothing : même pattern que les labels rider — snap si saut
    # énorme (changement de plan), deadzone pour absorber les jitters
    # 1-2 px, sinon EMA vers la cible.
    dx_f = _PODIUM_TEAM["display_x"]
    dy_f = _PODIUM_TEAM["display_y"]
    if (dx_f is None or dy_f is None
            or abs(target_x - dx_f) > _LABEL_SNAP_PX
            or abs(target_y - dy_f) > _LABEL_SNAP_PX):
        dx_f = float(target_x)
        dy_f = float(target_y)
    else:
        # Deadzone : on garde la position courante si l'écart est sous
        # le seuil, sur chaque axe indépendamment.
        if abs(target_x - dx_f) >= _LABEL_TARGET_DEADZONE_PX:
            dx_f += (target_x - dx_f) * _LABEL_SMOOTH_ALPHA
        if abs(target_y - dy_f) >= _LABEL_TARGET_DEADZONE_PX:
            dy_f += (target_y - dy_f) * _LABEL_SMOOTH_ALPHA
    _PODIUM_TEAM["display_x"] = dx_f
    _PODIUM_TEAM["display_y"] = dy_f
    dx_i = int(dx_f)
    dy_i = int(dy_f)

    # Chroma key fond plateau : le tableau s'efface là où le pixel
    # n'est PAS proche du gris du fond. Compute UNIQUEMENT sur la ROI
    # destination du tableau (pas la frame entière) → ~5× moins de
    # boulot numpy à 1080p, fps préservés.
    effective_mask = mask
    if TEAM_ROSTER_CHROMA_KEY:
        x0 = max(0, dx_i)
        y0 = max(0, dy_i)
        x1 = min(fw, dx_i + cw)
        y1 = min(fh, dy_i + ch)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            chroma_roi = _compute_chroma_mask(roi)
            if (mask is not None
                    and mask.shape == frame.shape[:2]):
                rvm_roi = mask[y0:y1, x0:x1]
                chroma_roi = np.maximum(chroma_roi, rvm_roi)
            global _PODIUM_CHROMA_SCRATCH
            if (_PODIUM_CHROMA_SCRATCH is None
                    or _PODIUM_CHROMA_SCRATCH.shape != (fh, fw)):
                _PODIUM_CHROMA_SCRATCH = np.zeros((fh, fw), dtype=np.uint8)
            _PODIUM_CHROMA_SCRATCH[y0:y1, x0:x1] = chroma_roi
            effective_mask = _PODIUM_CHROMA_SCRATCH
    composite_bgra(frame, card, dx_i, dy_i, mask=effective_mask)


def draw_tracks(frame: np.ndarray, tracks,
                bodies: list[dict] | None = None,
                mask: "np.ndarray | None" = None) -> None:
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

    # ── Dédup par nom LATCHÉ : un seul track affiché par nom à la fois.
    # Le keeper change UNIQUEMENT si un challenger dépasse son score de
    # +NAME_KEEPER_MARGIN — sinon on garde le keeper actuel, même si un
    # autre a temporairement un score plus haut. Tue le flicker quand
    # plusieurs faces (sosies / mismatches embedding) votent le même
    # nom : on choisit une fois pour toutes, on s'y tient.
    placeable = _dedup_by_name_latched(placeable)

    # ── Mode RAW : court-circuit total du placement pour isoler la
    # source du stutter. Placement direct au centre du visage, aucun
    # EMA, deadzone, stack, flip, dédup IoU, ni warp. Toggle via env
    # LABEL_RAW_PLACEMENT=1 (2026-05-29 debug session).
    draw_labels = layout_state.mode() != "none"
    if int(env("LABEL_RAW_PLACEMENT", "1")):
        if draw_labels:
            for t in placeable:
                x1, y1, x2, y2 = (int(v) for v in t.bbox_xyxy())
                lh, lw = t.label_img.shape[:2]
                dx = (x1 + x2) // 2 - lw // 2
                dy = y1 - lh - 18  # au-dessus du visage, gap fixe
                if dy < 0:
                    dy = y2 + 18  # flip dessous si hors cadre
                composite_bgra(frame, t.label_img, dx, dy, mask=mask)
        _overlay_team_roster(frame, placeable, mask)
        return

    # Dédup par nom + centrage SUPPRIMÉ (2026-05-29) : la distance-au-
    # centre se recalculait chaque frame, ce qui faisait osciller le
    # keeper entre 2 tracks de même nom → flicker synchrones perçus
    # comme "tous les labels d'un coup". On accepte le risque de voir un
    # nom afficher 2 fois (sosies, ghost, ré-id partielle) ; le dédup
    # spatial IoU≥0.7 latché ci-dessous suffit dans la plupart des cas.

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
        suppressed: list = []
        for tr in by_score:
            tr_bbox = tr.bbox_xyxy()
            if any(_iou(tr_bbox, k.bbox_xyxy()) >= _LABEL_IOU_DEDUP
                   for k in kept):
                suppressed.append(tr)
            else:
                kept.append(tr)
                tr.iou_losing_streak = 0
        # Latch : un track suppressé reste rendu tant qu'il n'a pas perdu
        # _LABEL_IOU_SUPPRESS_FRAMES frames consécutives. Au-delà on l'éjecte.
        # Le stack_level vertical absorbe l'overlap temporaire pendant la
        # période de grâce, mieux qu'un on/off frame-à-frame.
        for tr in suppressed:
            tr.iou_losing_streak += 1
            if tr.iou_losing_streak < _LABEL_IOU_SUPPRESS_FRAMES:
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
        # Warp caché par bucket d'angle (cf _orbital_warp_cached) : sans
        # cache, 10+ tracks × 60 fps × cv2.warpPerspective sature le CPU
        # et contribue aux stutters de label perçus en peloton dense.
        if PSEUDO_3D:
            yaw_deg = _orbital_yaw_deg(now_s, t.orbit_phase_s)
            label_to_draw = _orbital_warp_cached(t, yaw_deg)
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

        # 2. Premier arrivé premier servi : le stack_level est figé à la
        #    première placement du track. Les nouveaux tracks qui
        #    arrivent ensuite calculent leur needed_level contre les
        #    tracks déjà placés (boucle déterministe sortée par id) et
        #    se positionnent au-dessus si besoin. Les tracks existants
        #    NE bougent plus, même si le voisinage change.
        if not getattr(t, "_stack_locked", False):
            t.stack_level = needed_level
            t.detach_countdown = 0
            t._stack_locked = True  # type: ignore[attr-defined]
        # else: t.stack_level reste tel quel, on respecte la place
        # historique du track.

        # 3. Position cible finale + dead-zone.
        target_x = nat_x
        target_y = _ty_at(t.stack_level)

        # Anti-hors-cadre : si la position cible part au-dessus (ou en
        # dessous) du frame, force-flip vers l'autre côté ET reset latches
        # (court-circuite l'hystérésis qui sinon laisse le label invisible
        # pendant _LABEL_FLIP_HYST_FRAMES). On ne fait ça que si le côté
        # opposé tient effectivement dans le cadre — sinon clamp final.
        H_frame = frame.shape[0]
        if (target_y < 0) or (target_y + lh > H_frame):
            # Tente l'autre côté.
            alt_flipped = not flipped
            alt_nat_y = (y2 + face_gap) if alt_flipped else (y1 - lh - face_gap)
            alt_step_sign = +1 if alt_flipped else -1
            def _alt_ty_at(level: int) -> int:
                return alt_nat_y + alt_step_sign * level * step
            def _alt_collides_at(level: int) -> bool:
                ty = _alt_ty_at(level)
                r = (nat_x, ty, nat_x + lw, ty + lh)
                return any(_rect_collide(r, pr) for pr in placed_by_id.values())
            alt_level = 0
            while alt_level <= _LABEL_STACK_MAX and _alt_collides_at(alt_level):
                alt_level += 1
            if alt_level > _LABEL_STACK_MAX:
                alt_level = _LABEL_STACK_MAX
            alt_target_y = _alt_ty_at(alt_level)
            # Bascule SI l'autre côté donne une position effectivement
            # in-frame (sinon les deux côtés sont saturés, garde le moins
            # mauvais via le clamp ci-dessous).
            if 0 <= alt_target_y and alt_target_y + lh <= H_frame:
                flipped = alt_flipped
                step_sign = alt_step_sign
                t.label_flipped = flipped
                t.flip_change_countdown = 0
                t.stack_level = alt_level
                t.detach_countdown = 0
                target_y = alt_target_y

        # Clamp final de sécurité : si même l'autre côté ne tient pas,
        # on évite quand même l'invisible total.
        if target_y < 0:
            target_y = 0
        elif target_y + lh > H_frame:
            target_y = H_frame - lh

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
        if draw_labels:
            composite_bgra(frame, label_to_draw, dx, dy, mask=mask)

    _overlay_team_roster(frame, placeable, mask)


_stats_last_det_count = 0  # snapshot précédent du cumul détections worker

# Encoder NVJPEG global, init lazy à la 1ère frame (taille fixe après).
# Fallback automatique sur cv2.imencode si l'init échoue ou si encode
# raise une exception (logué une seule fois pour éviter le spam).
_nvjpeg_encoder = None
_nvjpeg_init_attempted = False
_nvjpeg_error_logged = False


def _gpu_encode_or_cv2(frame: np.ndarray) -> bytes | None:
    """Encode JPEG la frame BGR. Essaye NVJPEG GPU si NVJPEG_ENABLED,
    fallback cv2.imencode CPU sur toute erreur. Retourne bytes ou None."""
    global _nvjpeg_encoder, _nvjpeg_init_attempted, _nvjpeg_error_logged
    h, w = frame.shape[:2]
    if NVJPEG_ENABLED:
        # Init lazy à la 1ère frame.
        if not _nvjpeg_init_attempted:
            _nvjpeg_init_attempted = True
            try:
                from nvjpeg_encoder import NvJpegEncoder
                _nvjpeg_encoder = NvJpegEncoder(w, h, quality=JPEG_QUALITY,
                                                  sampling="420")
                log(f"NVJPEG encoder init OK ({w}x{h} Q={JPEG_QUALITY})")
            except Exception as e:
                log(f"NVJPEG init échec ({e}) → fallback cv2.imencode")
                _nvjpeg_encoder = None
        # Re-init si la taille de frame a changé (rare, ex: bullet-time
        # warp produit une taille différente).
        if (_nvjpeg_encoder is not None
                and (w != _nvjpeg_encoder.width
                     or h != _nvjpeg_encoder.height)):
            try:
                _nvjpeg_encoder.close()
                from nvjpeg_encoder import NvJpegEncoder
                _nvjpeg_encoder = NvJpegEncoder(w, h, quality=JPEG_QUALITY,
                                                  sampling="420")
            except Exception as e:
                if not _nvjpeg_error_logged:
                    log(f"NVJPEG re-init échec ({e}) → fallback cv2")
                    _nvjpeg_error_logged = True
                _nvjpeg_encoder = None
        if _nvjpeg_encoder is not None:
            try:
                return _nvjpeg_encoder.encode(frame)
            except Exception as e:
                if not _nvjpeg_error_logged:
                    log(f"NVJPEG encode err ({e}) → fallback cv2 (one-shot)")
                    _nvjpeg_error_logged = True
                # Continue vers le fallback cv2 ci-dessous.
    ok, buf = cv2.imencode(".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return bytes(buf)


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
    # Expose au HTTP handler pour les endpoints /pause /resume /toggle.
    MJPEGHandler.cap_worker = cap_worker

    # Tracker custom : 1 Kalman par face (filterpy) + matching IoU + ré-id
    # par embedding ArcFace. Quand un visage est baissé (RetinaFace perd la
    # détection), le track est "lost" mais survit max_missed frames. Si le
    # visage remonte ET que l'embedding match (cosine ≥ reid_threshold),
    # l'identité est restaurée — même track_id, même nom.
    # display_buffer auto = nb publish frames entre 2 détections + 1 buffer.
    # À 30/6 → 6 ; à 60/6 → 11 ; à 30/30 → 2. Évite que la bbox clignote
    # entre 2 détections.
    # Override via env DISPLAY_BUFFER : utile quand DETECT >= TARGET mais
    # la variance de latence GPU produit des is_new=False par bursts → tracks
    # masqués cycliquement. Floor = 6 frames (~200ms@30fps) pour absorber.
    _auto = max(6, int(round(TARGET_FPS / max(1, DETECT_FPS))) + 1)
    _display_buf = int(env("DISPLAY_BUFFER", str(_auto)))
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

    # Mask reader pour le compositing avec mask alpha (publié par
    # avtowan-mask-recog.service). Utilisé par MASK_BEHIND_LABELS et/ou
    # TEAM_LOGO_BEHIND. Lazy attach : si mask service down, latest()
    # renvoie None → fallback no-mask (labels au-dessus, logos sans
    # découpe). On l'instancie seulement si au moins une des features
    # qui en dépend est active.
    mask_reader = SHMMaskReader(MASK_SHM_NAME, MASK_STALE_MAX_AGE_S) \
        if (MASK_BEHIND_LABELS or TEAM_LOGO_BEHIND or SCENE_LOGO_BEHIND) else None

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

        _t0 = time.monotonic()
        _timings = {}

        # Pull la dernière frame du capture worker (drop-oldest).
        frame = cap_worker.get_latest()
        if frame is None:
            continue  # pas encore de frame, on retentera au tick suivant
        _timings["cap_get"] = (time.monotonic() - _t0) * 1000

        _t_a = time.monotonic()
        # Submit la frame courante au worker (copie nécessaire pour qu'il
        # ait sa propre vue stable pendant qu'on enchaîne sur la suivante).
        det_worker.submit(frame.copy())
        _timings["det_submit"] = (time.monotonic() - _t_a) * 1000
        _t_post_submit = time.monotonic()
        _t_section_revote = _t_post_submit  # marker pour la section re-vote

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

        _MAX_VOTES_PER_FRAME = int(env("MAX_VOTES_PER_FRAME", "2"))
        votes_this_frame = 0
        for t in active_tracks:
            if votes_this_frame >= _MAX_VOTES_PER_FRAME:
                break
            need_vote = False
            if (not t.name_resolved
                    and len(t.embedding_samples) >= _EMB_SAMPLES_MIN):
                need_vote = True
            elif (t.name_resolved
                    and t.samples_since_last_vote >= _REVOTE_INTERVAL):
                need_vote = True
            if not need_vote:
                continue
            votes_this_frame += 1
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
            # Stickiness : si déjà résolu et que le concurrent ne domine pas
            # franchement, on garde l'ancien nom (et on lisse le score). Évite
            # les flips d'identité sur les riders pile au seuil.
            if (name_changed and t.name_resolved
                    and new_score < t.score + _NAME_SWITCH_MARGIN):
                new_name = t.name
                name_changed = False
            t.name = new_name
            # EMA sur le score si pas de changement d'identité (sinon snap).
            if t.name_resolved and not name_changed:
                t.score = (1.0 - _SCORE_SMOOTH_ALPHA) * t.score \
                          + _SCORE_SMOOTH_ALPHA * new_score
            else:
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
                    t.team_code = meta.get("team_code")
                    t.team_name = meta.get("team_name")
                elif name_changed:
                    # Nouveau nom hors partants : on jette les meta.
                    t.bib = None
                    t.nationality = None
                    t.team_code = None
                    t.team_name = None
                    t.bib_confirmed = False

            # Charge la vignette photo 1× depuis face-db/<uciid>/ pour le
            # rendu skeleton-only. Cache négatif = False si pas trouvée
            # (ne re-tente pas à chaque re-vote). Si le nom change, on
            # invalide pour re-tenter avec le nouveau uciid.
            # UCI ID lookup : priorité manifest rider-recognition (couvre
            # tous les sportifs du dataset), fallback partants TDF.
            if name_changed:
                t.photo_thumb = None
                t.uciid = None
            if t.uciid is None:
                uciid = _NAME_TO_UCIID.get(t.name, "")
                if not uciid:
                    meta = _RIDERS_META.get(t.name)
                    uciid = meta.get("uciid") if meta else ""
                t.uciid = uciid or None
            # Photo thumb pour skeleton view : charge UNIQUEMENT si le
            # skeleton stream a au moins un subscriber. Sinon on perd
            # 5-20 ms par track sur un load disque dont personne ne se
            # sert. draw_skeleton_view est gated par subscribers > 0
            # plus loin, donc cohérent.
            if t.photo_thumb is None and skel_buf.subscribers > 0:
                photo = load_rider_photo(t.uciid or "", SKELETON_PHOTO_SIZE)
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
                sig = (t.name, t.bib, t.team_code, t.team_name,
                       t.nationality, t.bib_confirmed, t.uciid)
                was_none = t.label_img is None
                if was_none or sig != getattr(t, "_label_sig", None):
                    # 1) Lookup cache pré-rendu si bib_confirmed=False
                    # (cas standard, OCR off). Sinon render à la volée
                    # (rare). Évite ~15 ms PIL render dans la critical path.
                    cached = (_LABEL_PRERENDER_CACHE.get(t.name)
                              if not t.bib_confirmed else None)
                    if cached is not None:
                        t.label_img = cached
                    else:
                        t.label_img = render_lower_third(
                            t.name, t.score,
                            bib=t.bib, nationality=t.team_code,
                            bib_confirmed=t.bib_confirmed,
                            country_iso3=t.nationality,
                            uciid=t.uciid,
                            team_name=t.team_name,
                        )
                    t._label_sig = sig  # type: ignore[attr-defined]
                    if was_none:
                        log(f"LBL+ track={t.id} '{t.name}' "
                            f"score={t.score:.3f}")
            else:
                if t.label_img is not None:
                    log(f"LBL- track={t.id} '{t.name}' "
                        f"score={t.score:.3f} (show={t.score >= show_thresh} "
                        f"keep={t.score >= keep_thresh})")
                t.label_img = None
                t._label_sig = None  # type: ignore[attr-defined]

        # Fusion bib OCR : applique le cross-check après résolution face.
        # Si bibs OCR ont changé l'état d'un track (nouveau bib OU passage
        # confirmé), on re-render les labels affectés. La fonction met
        # aussi à jour t.body_track_id en passant (info utile association).
        _timings["revote_loop"] = (time.monotonic()
                                    - _t_section_revote) * 1000
        _t_section_post_revote = time.monotonic()
        bibs_now = bibs_state.get()
        if bibs_now:
            bibs_changed = associate_bibs_to_tracks(active_tracks, bibs_now)
            if bibs_changed:
                for t in active_tracks:
                    if t.label_img is None or t.score < THRESHOLD * 0.7:
                        continue
                    # Idem branche re-vote : signature pour éviter le
                    # re-render PIL si rien n'a changé visuellement.
                    sig = (t.name, t.bib, t.team_code, t.team_name,
                           t.nationality, t.bib_confirmed, t.uciid)
                    if sig == getattr(t, "_label_sig", None):
                        continue
                    t.label_img = render_lower_third(
                        t.name, t.score,
                        bib=t.bib, nationality=t.team_code,
                        bib_confirmed=t.bib_confirmed,
                        country_iso3=t.nationality,
                        uciid=t.uciid,
                        team_name=t.team_name,
                    )
                    t._label_sig = sig  # type: ignore[attr-defined]

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
        _timings["post_revote_to_draw"] = (time.monotonic()
                                            - _t_section_post_revote) * 1000
        _t_section_draw = time.monotonic()

        # Mask alpha (RVM) lu une seule fois par tick. Utilisé par :
        #   - draw_scene_logo_behind si SCENE_LOGO_BEHIND=1
        #   - draw_team_logos_behind si TEAM_LOGO_BEHIND=1
        #   - draw_tracks (compositing label-behind) si MASK_BEHIND_LABELS=1
        # Pas besoin de active_tracks pour scene-logo (indépendant des tracks).
        mask_for_blend = None
        if mask_reader is not None:
            m = mask_reader.latest()
            if m is not None and m.shape == frame.shape[:2]:
                mask_for_blend = m

        # Étape 0 : scene-logo arrière-plan (= couche la plus en fond).
        if SCENE_LOGO_BEHIND:
            draw_scene_logo_behind(frame, mask_for_blend)

        # Étape 1 : logos équipe DERRIÈRE riders (couche intermédiaire).
        if active_tracks and TEAM_LOGO_BEHIND:
            draw_team_logos_behind(frame, active_tracks, mask_for_blend)

        # Étape 2 : labels lower-third (devant les logos ; optionnellement
        # derrière les riders via le même mask si MASK_BEHIND_LABELS=1).
        if active_tracks:
            mask_for_labels = mask_for_blend if MASK_BEHIND_LABELS else None
            draw_tracks(frame, active_tracks, bodies_now, mask=mask_for_labels)

        # Overlay chrono en bas-droite (plaque 3D pseudo-warp + bevel).
        # Toggle via CLOCK_ENABLED env. Cache interne par seconde.
        overlay_clock(frame)

        _timings["draw_tracks"] = (time.monotonic()
                                    - _t_section_draw) * 1000
        _t_b = time.monotonic()
        # Met à jour la dernière frame native dispo (utile si un POST
        # /bullet-time arrive : on freeze ce contenu). Avant warp.
        bullet_state.update_live(frame)
        _timings["bullet_update"] = (time.monotonic() - _t_b) * 1000

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

        _t_r = time.monotonic()
        # Downscale optionnel avant encode JPEG (allège le décodage browser à
        # haut fps). Garde le ratio source. INTER_LINEAR pour la vitesse —
        # INTER_AREA prenait 17-20ms à 1080p→720p sur cette box (mesuré
        # 2026-05-29) alors qu'INTER_LINEAR fait le même downscale en ~3ms,
        # qualité visuelle négligeablement différente après JPEG.
        if PUBLISH_HEIGHT and frame.shape[0] > PUBLISH_HEIGHT:
            ratio = PUBLISH_HEIGHT / frame.shape[0]
            new_w = int(round(frame.shape[1] * ratio))
            frame = cv2.resize(frame, (new_w, PUBLISH_HEIGHT),
                               interpolation=cv2.INTER_LINEAR)
        _timings["resize"] = (time.monotonic() - _t_r) * 1000

        _t_pre_enc = time.monotonic()
        # Encode + push MJPEG /stream.mjpeg UNIQUEMENT si au moins un
        # client browser regarde le preview. body_recog + bib_recog ne
        # consomment plus le MJPEG (ils lisent en SHM), donc 0 subscribers
        # = personne ne regarde = pas d'encode = ~30% CPU libérée.
        if fbuf.subscribers > 0:
            buf_bytes = _gpu_encode_or_cv2(frame)
            if buf_bytes is not None:
                fbuf.push(buf_bytes)
        _t_end = time.monotonic()
        _total_ms = (_t_end - _t0) * 1000
        if _total_ms > 25:
            _enc_ms = (_t_end - _t_pre_enc) * 1000
            parts = " ".join(f"{k}={v:.1f}"
                              for k, v in _timings.items() if v >= 1.0)
            log(f"STALL {_total_ms:.0f}ms encode={_enc_ms:.0f}ms "
                f"[{parts}] tracks={len(active_tracks)}")

        # Si quelqu'un consomme le flux skeleton, on rend la vue
        # séparée et on push. Skip render quand 0 client (économie CPU).
        if skel_buf.subscribers > 0:
            skel = np.zeros_like(frame)
            # `bodies_now` a été calculé avec target=frame.shape AVANT
            # le resize ligne 3800. `skel` est zeros_like(frame post-resize)
            # donc 720p si PUBLISH_HEIGHT actif → re-query avec les dims
            # actuelles du skel canvas, sinon offsets visibles.
            sk_h, sk_w = skel.shape[:2]
            sk_bodies = bodies_state.get_persons(target_w=sk_w, target_h=sk_h)
            draw_skeleton_view(skel, sk_bodies, active_tracks)
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
    cap_worker: "CaptureWorker | None" = None

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
        elif self.path == "/pause":
            # GET /pause → renvoie l'état courant {paused: bool}
            cw = MJPEGHandler.cap_worker
            paused = bool(cw and cw.is_paused())
            body = f'{{"paused": {str(paused).lower()}}}\n'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/podium-mode":
            body = (f'{{"enabled":{str(podium_state.is_enabled()).lower()}}}\n'
                    ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/layout":
            body = f'{{"mode":"{layout_state.mode()}"}}\n'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_fullscreen(self, stream_path: str, title: str) -> None:
        # Page strictement alignée sur celle du studio webui Go
        # (cmd/avtowan-webui → /face-recog) : même HTML, même CSS, même
        # <header>+<main>+<img>, mêmes interactions (skeleton toggle,
        # fullscreen, bullet-time). Seules les URLs sont locales :
        # /stream.mjpeg au lieu de /face-recog/stream.mjpeg, etc.
        # Garantit l'identité visuelle (sizing image, scaling browser)
        # entre l'arbox et le studio — sinon CSS divergente = upscaling
        # bilinéaire qui rend visible la quantification JPEG (flicker).
        del title  # arbox utilise le titre studio
        del stream_path  # idem, sources hardcodées comme côté studio
        html = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>AVtoWan — Reconnaissance faciale</title>
<style>
  body { margin: 0; background: #000; color: #c9d1d9; font-family: -apple-system,BlinkMacSystemFont,sans-serif; }
  header { padding: 8px 16px; background: #161b22; border-bottom: 1px solid #30363d;
           display: flex; align-items: center; gap: 16px; }
  header h1 { margin: 0; font-size: 14px; font-weight: 600; color: #a371f7; }
  header .meta { font-size: 11px; color: #8b949e; }
  header button { margin-left: auto; background: #21262d; color: #c9d1d9;
                  border: 1px solid #30363d; border-radius: 4px;
                  padding: 4px 10px; font-size: 12px; cursor: pointer; }
  header button:hover { background: #30363d; }
  main { display: flex; justify-content: center; align-items: center;
         min-height: calc(100vh - 38px); padding: 8px; }
  img { max-width: 100%; max-height: calc(100vh - 60px); object-fit: contain;
        border-radius: 4px; background: #0d1117; cursor: zoom-in; }
  /* Mode fullscreen sur l'image seule : remplir l'écran, fond noir, curseur de sortie */
  img:fullscreen { max-height: 100vh; max-width: 100vw; border-radius: 0;
                   width: 100vw; height: 100vh; cursor: zoom-out; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
<header>
  <h1>Reconnaissance faciale</h1>
  <span class="meta">Source : NDI / UDP HEVC depuis studio</span>
  <span class="meta">·</span>
  <span class="meta">Modèle : InsightFace buffalo_l (RetinaFace + ArcFace) sur RTX 3080</span>
  <button id="podium-btn" onclick="togglePodium()" title="Affiche le tableau compo équipe au-dessus du flux (équipe majoritaire à l'écran)">⛶ Compo équipe</button>
  <button id="skel-btn" onclick="toggleSkeleton(this)" title="Bascule entre flux vidéo annoté et fond noir + squelettes + photos">⛶ Skeleton</button>
  <button onclick="fs()" title="Plein écran (Esc pour sortir)">⛶ Plein écran</button>
</header>
<main>
  <img id="stream" src="/stream.mjpeg" alt="flux annoté" onclick="fs()">
</main>
<script>
  function fs() {
    var el = document.getElementById('stream');
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else if (el.requestFullscreen) {
      el.requestFullscreen();
    }
  }
  var SKEL_SRC = '/stream-skeleton.mjpeg';
  var VIDEO_SRC = '/stream.mjpeg';
  function refreshSkeletonBtn(enabled) {
    var btn = document.getElementById('skel-btn');
    if (!btn) return;
    btn.textContent = enabled ? '◼ Skeleton ON' : '⛶ Skeleton';
    btn.style.background = enabled ? '#3d2a4a' : '#21262d';
    btn.style.borderColor = enabled ? '#a371f7' : '#30363d';
  }
  function toggleSkeleton(btn) {
    var img = document.getElementById('stream');
    var on = img.src.indexOf('stream-skeleton') === -1;
    var sep = '?t=' + Date.now();
    img.src = (on ? SKEL_SRC : VIDEO_SRC) + sep;
    refreshSkeletonBtn(on);
  }
  function refreshPodiumBtn(enabled) {
    var btn = document.getElementById('podium-btn');
    if (!btn) return;
    btn.textContent = enabled ? '◼ Compo équipe ON' : '⛶ Compo équipe';
    btn.style.background = enabled ? '#3d2a4a' : '#21262d';
    btn.style.borderColor = enabled ? '#a371f7' : '#30363d';
  }
  function togglePodium() {
    fetch('/podium-mode/toggle', { method: 'POST' })
      .then(function (r) { return r.json(); })
      .then(function (j) { refreshPodiumBtn(!!j.enabled); })
      .catch(function () { /* silent */ });
  }
  // Sync initial du bouton avec l'état serveur au chargement.
  fetch('/podium-mode')
    .then(function (r) { return r.json(); })
    .then(function (j) { refreshPodiumBtn(!!j.enabled); })
    .catch(function () { /* silent */ });
</script>
</body>
</html>""".encode("utf-8")
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
        if self.path in ("/pause", "/resume", "/pause/toggle"):
            # POST /pause   → set pause
            # POST /resume  → clear pause
            # POST /pause/toggle → toggle
            cw = MJPEGHandler.cap_worker
            if cw is None:
                body = b'{"ok":false,"reason":"capture worker not ready"}'
                code = 503
            else:
                if self.path == "/pause":
                    cw.set_paused(True)
                elif self.path == "/resume":
                    cw.set_paused(False)
                else:
                    cw.set_paused(not cw.is_paused())
                body = f'{{"ok":true,"paused":{str(cw.is_paused()).lower()}}}'.encode()
                code = 200
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path in ("/layout", "/layout/toggle"):
            if self.path == "/layout/toggle":
                new_mode = layout_state.toggle()
            else:
                # Mode via query (?mode=solo|none) ou body JSON {"mode": ...}.
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                mode_val = None
                if "mode" in qs and qs["mode"]:
                    mode_val = qs["mode"][0]
                if mode_val is None:
                    length = 0
                    try:
                        length = int(self.headers.get("Content-Length") or 0)
                    except ValueError:
                        length = 0
                    raw = self.rfile.read(length) if length > 0 else b""
                    try:
                        if raw:
                            data = json.loads(raw)
                            if isinstance(data, dict) and "mode" in data:
                                mode_val = data["mode"]
                    except (json.JSONDecodeError, ValueError):
                        mode_val = None
                if mode_val is None:
                    mode_val = "solo"
                new_mode = layout_state.set_mode(str(mode_val))
            body = f'{{"ok":true,"mode":"{new_mode}"}}'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path in ("/podium-mode", "/podium-mode/toggle"):
            if self.path == "/podium-mode/toggle":
                new_val = podium_state.toggle()
            else:
                length = 0
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                raw = self.rfile.read(length) if length > 0 else b""
                on_val = True
                try:
                    if raw:
                        data = json.loads(raw)
                        if isinstance(data, dict) and "on" in data:
                            on_val = bool(data["on"])
                except (json.JSONDecodeError, ValueError):
                    on_val = True
                new_val = podium_state.set_enabled(on_val)
            body = f'{{"ok":true,"enabled":{str(new_val).lower()}}}'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
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
    # Provider priority : Tensorrt > CUDA > CPU. Tensorrt = engines précompilés
    # 2-4× plus rapide que CUDA EP pour SCRFD + ArcFace. Le 1er run par modèle
    # compile l'engine (~30 s) puis cache dans TRT_ENGINE_CACHE_PATH (default
    # /tmp/onnxruntime_trt_cache). Toggle off via USE_TRT=0 en cas de pb.
    USE_TRT = bool(int(env("USE_TRT", "1")))
    if GPU_ID >= 0:
        if USE_TRT:
            trt_cache = env("TRT_ENGINE_CACHE_PATH",
                             "/var/cache/face-recog/trt-engines")
            os.makedirs(trt_cache, exist_ok=True)
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache,
                    "trt_fp16_enable": True,
                    "device_id": GPU_ID,
                }),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=providers)
    app.prepare(ctx_id=GPU_ID, det_size=(DET_SIZE, DET_SIZE))
    log(f"détecteur ready en {time.monotonic() - t0:.1f}s")

    index = FaceIndex(INDEX)
    if len(index.names) == 0:
        log(f"WARN: index vide ou introuvable ({INDEX}) — bbox seront affichés "
            f"sans nom (run index_faces.py d'abord)")

    # Méta riders (bib + nationalité) depuis le JSON partants ASO.
    global _RIDERS_META, _NAME_TO_UCIID, _TEAM_ROSTERS
    _RIDERS_META = load_riders_meta(PARTANTS_JSON)
    if _RIDERS_META:
        log(f"partants chargés : {len(_RIDERS_META)} riders depuis "
            f"{PARTANTS_JSON}")
    else:
        log(f"partants JSON absent/vide ({PARTANTS_JSON}) — labels sans "
            f"bib ni nationalité")

    # Rosters par équipe pour le tableau compo (mode podium).
    _TEAM_ROSTERS = load_team_rosters(PARTANTS_JSON)
    if _TEAM_ROSTERS:
        log(f"team rosters chargés : {len(_TEAM_ROSTERS)} équipes")
    else:
        log("team rosters vides — tableau compo équipe désactivé")

    # Map name → uciid (pour le lookup photo, plus large que partants).
    _NAME_TO_UCIID = load_name_to_uciid(RIDER_MANIFEST_JSON)
    if _NAME_TO_UCIID:
        log(f"manifest rider chargé : {len(_NAME_TO_UCIID)} name→uciid "
            f"depuis {RIDER_MANIFEST_JSON}")
    else:
        log(f"manifest rider absent/vide ({RIDER_MANIFEST_JSON}) — photos "
            f"limitées aux UCI IDs des partants")

    # Pre-warm portrait cache : charge tous les portraits des partants au
    # boot pour éviter les ~20 ms spike disk-read au 1er render de chaque
    # label. À 200 riders × ~20 ms = ~4 s de boot supplémentaire, mais
    # tue les pics de revote_loop quand un nouveau rider apparaît.
    if PORTRAIT_PREWARM:
        # Compute portrait target_h identique à render_lower_third (2 lignes,
        # tag size = ~70 px). Si la résolution change plus tard, le cache
        # est invalidé naturellement par (uciid, h) key — coût juste un
        # 2e load.
        _approx_content_h = 70
        n_loaded = 0
        for name, meta in _RIDERS_META.items():
            uciid = meta.get("uciid") or _NAME_TO_UCIID.get(name)
            if uciid and _load_portrait(uciid,
                                          target_h=_approx_content_h) is not None:
                n_loaded += 1
        log(f"portrait cache prewarm : {n_loaded}/{len(_RIDERS_META)} riders")

    # Pre-render TOUS les labels au boot. Chaque label PIL = ~10-15 ms à
    # rendre ; si on les fait au runtime dans la revote loop, ça spike la
    # publish loop (cluster de votes en lockstep = stutter visible).
    # En pré-rendant, la revote loop devient : match() + dict lookup, ~1 ms.
    # Stockage : (name, bib, team_code, team_name, nationality, uciid,
    # bib_confirmed=False) → BGRA premul. bib_confirmed=True n'est pas
    # pré-rendu (rare, recharge à la première occurrence — OCR off de toute
    # façon).
    global _LABEL_PRERENDER_CACHE
    _LABEL_PRERENDER_CACHE = {}
    if LABEL_PRERENDER:
        n_pre = 0
        t_pre_start = time.monotonic()
        for name, meta in _RIDERS_META.items():
            uciid = meta.get("uciid") or _NAME_TO_UCIID.get(name)
            bib = meta.get("bib")
            try:
                img = render_lower_third(
                    name, score=0.0,
                    bib=bib, nationality=meta.get("team_code"),
                    bib_confirmed=False,
                    country_iso3=meta.get("nationality"),
                    uciid=uciid,
                    team_name=meta.get("team_name"),
                )
                _LABEL_PRERENDER_CACHE[name] = img
                n_pre += 1
            except Exception as e:
                log(f"prerender err name={name}: {e}")
        dt = (time.monotonic() - t_pre_start) * 1000
        log(f"label prerender : {n_pre}/{len(_RIDERS_META)} riders "
            f"({dt:.0f} ms)")

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
