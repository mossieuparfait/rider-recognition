# CLAUDE.md

## Comportement

1. **Think before coding**, hypothèses énoncées.
2. **Simplicity first** — minimum qui résout, pas d'abstraction prématurée.
3. **Surgical changes**.
4. **Critère de succès vérifiable** avant la première ligne.

Code et CLI en **français**.

## Scope (strict)

**Reconnaissance de visage uniquement.** Rien d'autre :

- ❌ Pas d'ingest de data (= autre projet, fournit déjà le dataset).
- ❌ Pas de live timing.
- ❌ Pas de multi-courses, pas de logique broadcast, pas d'overlay.
- ✅ Charger un dataset déjà formaté `<PERSON>/photos.png` ou équivalent.
- ✅ Calculer les embeddings ArcFace via InsightFace.
- ✅ Produire un index `.npz` au format AVtoWan face-recog.

## Outil existant à privilégier

Avant d'écrire un nouveau script, vérifier
`videoWan/cmd/avtowan-face-recog/index_faces.py` — il fait du
`--db <dossier> --out <face-index.npz>` en une commande sur un dossier
`<PERSON>/photos.*`. Pour les cas standard, c'est l'outil à utiliser
directement, pas réinventer.

Ce repo n'apporte de la valeur que sur :
- Datasets indexés par UCIID + lookup nom (signatureNG `_manifest.json`)
- Multi-embedding par personne (matching plus robuste vs un mean seul)

Si la tâche c'est juste "indexer un dossier `<NOM>/photos`" → utiliser
`index_faces.py` directement, sans toucher à ce repo.

## Layout

```
rider_recognition/
  dataset.py            charge signatureNG (_manifest.json + photos)
scripts/
  scan_dataset.py       rapport stats
  build_index.py        embeddings ArcFace (1 par photo)
  to_avtowan_format.py  conversion → format AVtoWan (mean par personne)
```

## Stack

Python + InsightFace + onnxruntime-gpu (déjà installés dans
`/opt/avtowan-face-recog/venv` sur le studio). Réutiliser ce venv, ne
pas en créer un autre.

**Bootstrap CUDA obligatoire** dans tout script qui import onnxruntime :
preload RTLD_GLOBAL des `.so` nvidia depuis `site-packages/nvidia/*/lib/`
AVANT `import onnxruntime`, sinon fallback CPU silencieux (cf
[[feedback_replicate_critical_patterns]]).
