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

**Dev (actuel)** : signatureNG est sur la **même machine**. Accès direct
filesystem :

```
signature/public/data/rider_photos/<UCIID>/<NN>_portrait_<TAG>.png
  → 782 dossiers, 2320 photos, 226 MB (constaté 2026-05-27)
  → UCIID = clé identité officielle UCI (11 chiffres)
  → plusieurs photos par coureur, taggées par contexte (course+année)
```

Métadonnées riders (nom, équipe, etc.) : modèles Mongoose dans
`signatureNG/signature/models/rider.js` + `team.js` + jointures.

**Prod (à venir)** : signatureNG migrera sur une autre machine du même LAN
que le studio. Mode d'accès final non tranché — abstraire la source de data
derrière une seule interface (un chargeur) pour pouvoir basculer
local→HTTP/NFS sans toucher le code de reco. **Pas d'abstraction
prématurée tant que dev local** : on commence avec un path en config, on
basculera quand le déploiement le demandera.

## Layout (cible)

```
cmd/
  rider-recog/     Reco visuelle (face + dossard + maillot) — GPU
  rider-live/      Consumer live timing (format fourni par l'app d'ingest)

pkg/                Code partagé (schemas data consommée, utils)
deploy/             systemd units pour la box face-recog
docs/               architecture, formats, etc.
```

(Layout sera réajusté quand on aura tranché la stack et le format d'accès
aux data.)

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
