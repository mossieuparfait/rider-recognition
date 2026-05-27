# CLAUDE.md

Behavioral guidelines + project guide pour **rider-recognition**.

## Comportement attendu

1. **Think before coding** : énoncer les hypothèses, ne pas en planquer. Si
   plusieurs interprétations existent → demander, pas choisir en silence.
2. **Simplicity first** : le minimum qui résout le problème. Pas de
   "flexibilité" non demandée, pas d'abstraction prématurée.
3. **Surgical changes** : ne toucher que ce qu'on doit. Pas de "j'en profite
   pour refactor adjacent".
4. **Goal-driven execution** : critère de succès vérifiable avant d'écrire la
   première ligne.

Code commenté et messages CLI en **français**.

## Projet

Reconnaissance visuelle de coureurs cyclistes pour overlay broadcast.
Tourne sur la **box face-recog** (i3-12100 + RTX 3080 + 2.5 GbE), séparée
d'AVtoWan.

Cible : à partir d'une frame vidéo entrante, identifier les coureurs visibles
(nom + dossard + équipe) et fournir leur info live (écart, vitesse, position
dans le peloton) à un outil broadcast (overlay, OSD, régie).

Course-agnostique : letour, paris-nice, dauphiné, vuelta, etc. — toutes les
courses ASO exposant `racecenter.<course>.fr/api/*`.

## Principe architectural (à respecter)

- **Box dédiée** : aucune cohabitation hot-path avec AVtoWan. La box
  rider-recognition reçoit la vidéo (NDI ou autre) et publie des métadonnées
  séparées.
- **Aucun couplage repo** avec videoWan. Les deux projets vivent en parallèle.
- **Ingest hors scope** : la BDD de référence (coureurs, photos, live timing)
  est produite par **signatureNG** (`/home/ben/AIlocal/signatureNG/`). Pas
  de re-scraping ici, pas d'appel direct aux APIs ASO.
- **Course-agnostique dès la V1** : pas de constante "letour" hardcodée.

## Accès à la data signatureNG

**Dev (actuel)** : signatureNG est sur la **même machine**. Tout est dans
`signature/public/data/rider_photos/` :

```
rider_photos/
├── _manifest.json          ← source unique : 857 UCIIDs → {name, photos: [{race,type,url}]}
├── 10048858880/
│   ├── 01_podium_VUE2024.png
│   ├── 02_podium_VUE2025.png
│   ├── 03_portrait_VUE2024.png
│   └── 04_portrait_VUE2025.png
├── 10006895064/
│   └── ...
└── ...                     ← 782 dossiers physiques, 2320 photos, 226 MB
```

- `_manifest.json` est la **source de vérité** : `{uciid: {name, photos: [...]}}`
- Le nom de fichier physique suit `<NN>_<type>_<RACE><YEAR>.png`
  (type ∈ {`podium` 660×1000, `portrait` 400×400}, RACE ∈ {VUE, TDF, PRX, LBL, PN...})
- Diff manifest (857) vs disque (782) : 75 UCIIDs indexés sans fichiers locaux
  → à ignorer (skipper en chargement)

Pas besoin de MongoDB, pas besoin de `test.json` (qui n'est qu'une course
PN26 isolée — le manifest les couvre toutes).

**Prod (à venir)** : signatureNG migrera sur autre machine du LAN studio.
Mode d'accès final (API HTTP, NFS, rsync) à trancher à la migration. **Pas
d'abstraction prématurée tant que dev local** : path en config, on
basculera quand le déploiement le demandera.

## Layout (Python)

```
rider_recognition/      package Python (code partagé)
  __init__.py
  dataset.py            charge _manifest.json + scanne rider_photos/
  ...                   (embeddings, recog, etc. à venir)

scripts/                CLI / scripts utilitaires
  scan_dataset.py       rapport sur le dataset disponible
  ...

deploy/                 systemd units pour la box face-recog (plus tard)
docs/                   architecture, formats
```

Stack : **Python** (cohérent avec `avtowan-face-recog` côté videoWan,
InsightFace/PyTorch/ONNX/TensorRT). Pas de C++ tant que pas besoin.

## Conventions

- Pas de TODO non daté + assigné. Pas de commentaire `// legacy/deprecated`.
- Chaque sous-`cmd/` a un README.md qui dit ce que fait le binaire.
- Aucun secret committé (cf `.gitignore`).
- Tests à côté du code (pas de `tests/` séparé).

## Hardware cible

- **Box face-recog** : i3-12100 + RTX 3080 + 2.5 GbE (cf [[project_face_recog_box]])
- GPU compute : CUDA + (TensorRT si on optimise). Pas de partage avec d'autres
  workloads tant que la box est dédiée broadcast.

## Critères "propre" (à maintenir)

- Aucun TODO sans owner + date.
- Chaque ajout passe le critère "every changed line traces directly to the
  user request".
- Pipeline ingest reproductible (re-run idempotent).
