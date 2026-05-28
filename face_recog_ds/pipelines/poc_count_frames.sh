#!/bin/bash
# Compte les frames qui passent à différents points du pipeline pour
# isoler où ça bloque. identity name=tap silent=false logge chaque buffer.

WORK=/work
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/x86_64-linux-gnu/gstreamer-1.0
export NDI_RUNTIME_DIR_V6=/opt/ndi-runtime

NDI_SOURCE="${NDI_SOURCE:-STUDIO (AVtoWan-FaceRecog)}"

gst-launch-1.0 \
    ndisrc ndi-name="${NDI_SOURCE}" \
  ! ndisrcdemux name=d \
  d.video \
  ! queue \
  ! identity name=tap_raw silent=false \
  ! videoconvert \
  ! video/x-raw,format=NV12 \
  ! identity name=tap_nv12 silent=false \
  ! nvvideoconvert \
  ! "video/x-raw(memory:NVMM),format=NV12" \
  ! identity name=tap_nvmm silent=false \
  ! mux.sink_0 \
    nvstreammux name=mux batch-size=1 width=1920 height=1080 \
                batched-push-timeout=33000 live-source=1 \
  ! identity name=tap_mux silent=false \
  ! nvinfer config-file-path="${WORK}/configs/retinaface_minimal.txt" unique-id=1 \
  ! identity name=tap_infer silent=false \
  ! fakesink sync=false \
  d.audio ! queue ! fakesink sync=false
