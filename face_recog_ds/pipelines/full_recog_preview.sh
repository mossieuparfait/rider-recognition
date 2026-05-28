#!/bin/bash
# Full face-recog pipeline avec output MJPEG sur HTTP pour preview.
#
# Architecture identique à full_recog.sh mais le fakesink final est
# remplacé par nvjpegenc (encode JPEG sur GPU = NVJPEG) puis
# multipartmux + tcpserversink pour exposer le MJPEG stream.
#
# Preview : curl http://192.168.1.175:8810 | mpv -
# ou ouvrir http://192.168.1.175:8810 dans un browser MJPEG-compat.
#
# Tout reste GPU jusqu'à l'encode JPEG inclus. Seuls les bytes JPEG
# (~50-150 KB par frame) descendent en CPU pour la sortie TCP.

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
  ! "video/x-raw(memory:NVMM),format=NV12" \
  ! nvvideoconvert \
  ! "video/x-raw,format=UYVY" \
  ! ndisink ndi-name="${NDI_OUT_NAME:-arbox-FaceRecog}"
