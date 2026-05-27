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
- **Course-agnostique dès la V1** : pas de constante "letour" hardcodée — un
  paramètre `--race <slug>` qui choisit l'hôte racecenter.
- **DB locale autonome** : la box doit pouvoir tourner offline une fois la DB
  ingérée (race day = pas de dépendance au RTT internet).

## Source données ASO

API publique racecenter (sans auth, découverte 2026-05-27) :

| Endpoint | Contenu |
|---|---|
| `/api/allCompetitors-<year>` | Coureurs (UCICode, idUCI, nom, dossard, 4 URLs photos) |
| `/api/stage-<year>` | Étapes (parcours, départ, arrivée) |
| `/api/team-<year>` | Équipes |
| `/live-stream` | SSE temps réel (positions, écarts, vitesses) |
| `/profils/<year>/profile-NN-<hash>.csv` | Profil altimétrique étape NN |

CDN photos `img.aso.fr` public, paramétrable côté URL pour la taille/crop.

Creds OAuth2 `directioncyclisme` (fichier hors-repo) probablement pour
`api.aso.fr` (IP-whitelisté). Non utilisé en V1 — à creuser quand on aura le
mail ASO d'origine.

## Layout (cible)

```
cmd/
  rider-ingest/    Pipeline ASO → DB locale
  rider-recog/     Reco visuelle (face + dossard + maillot) — GPU
  rider-live/      Consumer SSE /live-stream

pkg/                Code partagé (clients ASO, schemas, etc.)

data/               DB locale + cache photos (gitignore)
deploy/             systemd units pour la box face-recog
docs/               architecture, formats, etc.
```

(Layout sera réajusté quand on aura tranché la stack — Python vs C++ vs mix.)

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
