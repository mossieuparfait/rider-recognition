# face_recog_ds — DeepStream-based face-recog rewrite

**Objectif** : pipeline 100% GPU/CUDA pour reconnaissance faciale broadcast
sur arbox (RTX 3080). Remplace progressivement le pipeline Python
orchestré (`scripts/{body,bib}_recog_service.py` + un consommateur
externe d'index ArcFace), qui plafonne CPU.

Décision actée le 2026-05-28.

## Stack engagée

- **DeepStream 7.1** (container `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`)
- **TensorRT 10.3** (bundlé dans le container)
- **GStreamer 1.24** (Ubuntu 24.04 + container DS)
- **CUDA 12.x** (driver 595 sur arbox supporte CUDA 13.2, container utilise 12.x)

## Structure

- `configs/`     : .yaml et .txt pour nvinfer, nvtracker, nvdsosd
- `engines/`     : moteurs TensorRT générés (binaires, gitignore)
- `pipelines/`  : scripts gst-launch ou Python gst pour les pipelines
- `plugins/`    : sources des plugins GStreamer custom (NDI src/sink, custom probes)
- `docs/`       : notes d'archi, choix de design

## Phases (cf tasks)

1. **Phase 1** : install SDK + TRT engines RetinaFace/ArcFace + plugin NDI src + POC pipeline
2. **Phase 2** : recognition + tracking nvtracker, comparer perf vs Python
3. **Phase 3** : porter multi-frame voting + label placement + bib fusion
4. **Phase 4** : body + bib pipelines parallèles + NDI out + déploiement prod

## Non-buts (pour ne pas dériver)

- ❌ Pas de réécriture en Python orchestré, on reste en pipeline GStreamer
- ❌ Pas d'appsrc Python en hot path pour NDI in — plugin C custom
- ❌ Pas de cv2.cuda pour des opérations isolées — tout passe par TensorRT/CUDA kernels intégrés au pipeline
