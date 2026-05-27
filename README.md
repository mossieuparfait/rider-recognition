# rider-recognition

Reconnaissance visuelle de coureurs cyclistes pour overlay broadcast
(nom + dossard + équipe + écart live).

Tourne sur la **box face-recog** (i3-12100 + RTX 3080 + 2.5 GbE),
indépendante d'AVtoWan (cf [[project_face_recog_box]]).

## Pipeline (cible)

1. **Ingestion ASO** : `racecenter.<course>.fr/api/*` → DB locale riders + photos
   référence (cache `img.aso.fr`).
2. **Recognition visuelle** : face + dossard + maillot → bib/nom/équipe par
   frame entrante.
3. **Live timing overlay** : SSE `/live-stream` → position / écart / vitesse
   exposés aux outils broadcast.

Course-agnostique dès le départ (letour, paris-nice, dauphiné, vuelta,
liège-bastogne-liège…).

## Source données ASO

API publique racecenter (sans auth) :

| Endpoint | Contenu |
|---|---|
| `/api/allCompetitors-<year>` | Coureurs (UCICode, idUCI, nom, dossard, photos `img.aso.fr`) |
| `/api/stage-<year>` | Étapes (parcours, départ, arrivée) |
| `/api/team-<year>` | Équipes |
| `/live-stream` | SSE temps réel (positions, écarts, vitesses) |
| `/profils/<year>/profile-NN-<hash>.csv` | Profil altimétrique étape NN |

Creds B2B `directioncyclisme` (cf `[secrets]`) probablement pour `api.aso.fr`
(IP-whitelisté) — données enrichies non documentées, à creuser plus tard.
