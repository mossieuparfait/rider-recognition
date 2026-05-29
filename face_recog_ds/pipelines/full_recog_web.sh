#!/bin/bash
# Pipeline DeepStream complet avec sortie MJPEG HTTP pour page web.
# Tee + HDMI seront ajoutés dans une version suivante quand la session
# graphique sera setupée sur arbox.

export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/x86_64-linux-gnu/gstreamer-1.0

PORT="${PORT:-5000}"
PREVIEW_PORT="${PREVIEW_PORT:-8810}"
WORK=/work
DS=/opt/nvidia/deepstream/deepstream

gst-launch-1.0 \
    udpsrc port="${PORT}" buffer-size=8388608 \
  ! tsdemux \
  ! h265parse \
  ! nvv4l2decoder \
  ! "video/x-raw(memory:NVMM),format=P010_10LE" \
  ! nvvideoconvert \
  ! "video/x-raw(memory:NVMM),format=NV12" \
  ! mux.sink_0 \
    nvstreammux name=mux batch-size=1 width=1920 height=1080 \
                batched-push-timeout=33000 live-source=1 \
  ! nvinfer config-file-path="${WORK}/configs/yolov8l_face.txt" \
            unique-id=1 \
  ! nvtracker \
        tracker-width=640 tracker-height=384 \
        ll-lib-file="${DS}/lib/libnvds_nvmultiobjecttracker.so" \
        ll-config-file="${DS}/samples/configs/deepstream-app/config_tracker_NvSORT.yml" \
        gpu-id=0 \
  ! nvinfer config-file-path="${WORK}/configs/arcface_secondary.txt" \
            unique-id=2 \
  ! nvvideoconvert \
  ! nvdsosd \
  ! nvvideoconvert \
  ! "video/x-raw,format=I420" \
  ! jpegenc quality=80 \
  ! image/jpeg \
  ! multipartmux boundary=rider-mjpeg \
  ! tcpserversink host=0.0.0.0 port="${PREVIEW_PORT}" \
                  sync=false async=false
