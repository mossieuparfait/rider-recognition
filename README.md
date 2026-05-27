# rider-recognition

**Reconnaissance de visage** uniquement. Dataset fourni de l'extérieur
(import = autre projet). Pas de live timing, pas de multi-courses, pas
d'ingest.

## Pipeline

1. Charger le dataset (un dossier `<PERSON>/photos.png`).
2. Calculer les embeddings ArcFace (InsightFace buffalo_l).
3. Produire un index `.npz` consommable par le service face-recog AVtoWan
   (`videoWan/cmd/avtowan-face-recog/face_recog_service.py`) qui le
   reload automatiquement sur mtime change.

## Source de données

Fournie de l'extérieur. En dev actuel : `signatureNG` (autre projet de
l'utilisateur) maintient `signature/public/data/rider_photos/<UCIID>/*.png`
+ un `_manifest.json` qui mappe UCIID → nom humain.

## Scripts

- `scripts/build_index.py` — embeddings ArcFace 512D (1 par photo)
- `scripts/to_avtowan_format.py` — mean par personne + format AVtoWan
  (`embeddings, names`)
- `scripts/scan_dataset.py` — rapport stats sur le dataset

Note : `index_faces.py` côté `avtowan-face-recog` fait directement le
job équivalent en une commande sur un dossier `<PERSON>/photos.png` ; ce
repo n'apporte de la valeur que sur les cas où on veut garder
plusieurs embeddings par personne, ou un mapping UCIID → nom comme dans
le manifest signatureNG.
