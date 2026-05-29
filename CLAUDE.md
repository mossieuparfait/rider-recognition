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
- ✅ Produire un index `.npz` (embeddings + names) consommable par tout
  reconnaisseur ArcFace.

## Quand utiliser ce repo

Apporte de la valeur seulement sur :
- Datasets indexés par UCIID + lookup nom (signatureNG `_manifest.json`)
- Multi-embedding par personne (matching plus robuste vs un mean seul)

Pour un cas standard "indexer un dossier `<NOM>/photos`", un outil
généraliste type `insightface` CLI suffit — ce repo serait du surplus.

## Layout

```
rider_recognition/
  dataset.py            charge signatureNG (_manifest.json + photos)
scripts/
  scan_dataset.py            rapport stats
  build_index.py             embeddings ArcFace (1 par photo)
  export_mean_index_npz.py   mean par personne → .npz (embeddings, names)
```

## Stack

Python + InsightFace + onnxruntime-gpu. Venv dédié `venv/` à la racine
du repo (`python -m venv venv && venv/bin/pip install -r requirements.txt`).

**Bootstrap CUDA obligatoire** dans tout script qui import onnxruntime :
preload RTLD_GLOBAL des `.so` nvidia depuis `site-packages/nvidia/*/lib/`
AVANT `import onnxruntime`, sinon fallback CPU silencieux (cf
[[feedback_replicate_critical_patterns]]).
