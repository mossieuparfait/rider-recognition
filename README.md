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

## Source de données (en attente)

L'ingest est externe. Le user fournira :
- URL ou path d'accès à la base coureurs (nom, dossard, équipe, photos référence)
- Format / schéma des données
- Endpoint live timing à consommer

À documenter ici une fois reçu.
