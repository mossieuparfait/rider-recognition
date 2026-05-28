#!/bin/bash
# POC pipeline DeepStream : NDI → NVMM → nvinfer (RetinaFace TRT) → fakesink
# Mesure : GPU utilization pendant la détection vs le service Python actuel.
#
# Lancer en parallèle dans un autre shell :
#   watch -n 0.5 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

set -e

# Paths montés depuis l'hôte (engines + configs).
WORK=/work

# Plugins paths DeepStream + ndi.
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/x86_64-linux-gnu/gstreamer-1.0
export NDI_RUNTIME_DIR_V6=/opt/ndi-runtime

NDI_SOURCE="${NDI_SOURCE:-STUDIO (AVtoWan-FaceRecog)}"

echo "[poc] NDI source: ${NDI_SOURCE}"
echo "[poc] launching pipeline ndisrc → NVMM → nvinfer(retinaface) → fakesink"
echo "[poc] press Ctrl+C to stop"

gst-launch-1.0 \
    ndisrc ndi-name="${NDI_SOURCE}" \
  ! ndisrcdemux name=ndid \
  ndid.video \
  ! queue \
  ! videoconvert \
  ! video/x-raw,format=NV12 \
  ! nvvideoconvert \
  ! "video/x-raw(memory:NVMM),format=NV12" \
  ! mux.sink_0 \
    nvstreammux name=mux batch-size=1 width=1920 height=1080 \
                batched-push-timeout=33000 live-source=1 \
  ! nvinfer config-file-path="${WORK}/configs/retinaface_minimal.txt" \
            unique-id=1 \
  ! nvvideoconvert \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
  ndid.audio \
  ! queue \
  ! fakesink sync=false silent=true
