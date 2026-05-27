# rider-recognition

Reconnaissance visuelle de coureurs cyclistes pour overlay broadcast
(nom + dossard + équipe + écart live).

Tourne sur la **box face-recog** (i3-12100 + RTX 3080 + 2.5 GbE),
indépendante d'AVtoWan (cf [[project_face_recog_box]]).

## Pipeline (cible)

1. **Recognition visuelle** : face + dossard + maillot → bib/nom/équipe par
   frame entrante. GPU RTX 3080.
2. **Live timing overlay** : consume position / écart / vitesse pour overlay
   broadcast.

L'**ingestion** des données coureurs / étapes / live timing n'est **pas**
dans ce projet — elle est faite par une app séparée du user qui exposera
ses accès. Ce repo est consommateur.

Course-agnostique dès le départ (letour, paris-nice, dauphiné, vuelta,
liège-bastogne-liège…).

## Source de données

Fournie par **signatureNG** (`/home/ben/AIlocal/signatureNG/`) — app
signature ASO podium qui maintient déjà la BDD coureurs + photos.

**Dev (local) :**
- Photos : `signature/public/data/rider_photos/<UCIID>/<NN>_portrait_<TAG>.png`
  → 782 coureurs, 2320 photos, 226 MB
- Métadonnées : MongoDB via models `signature/models/{rider,team,race}.js`

**Prod (LAN) :** signatureNG sur autre machine du LAN studio. Mode d'accès
(API HTTP / NFS / rsync) à trancher à la migration.
