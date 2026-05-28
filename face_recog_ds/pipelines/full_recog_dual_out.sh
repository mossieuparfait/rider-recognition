#!/bin/bash
# Pipeline DeepStream complet avec double sortie :
#   - HDMI direct via nveglglessink (zero CPU, GPU framebuffer)
#   - MJPEG HTTP sur port 8810 pour la page web dashboard
#
# Le tee fork le flux annoté APRÈS nvdsosd. L'écran HDMI voit en
# permanence ce que voit aussi le browser distant.

export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/x86_64-linux-gnu/gstreamer-1.0

PORT="${PORT:-5000}"
PREVIEW_PORT="${PREVIEW_PORT:-8810}"
WORK=/work
DS=/opt/nvidia/deepstream/deepstream

# Permet à GStreamer d'utiliser le serveur X de l'hôte pour le rendu GL.
# Le container est démarré avec -e DISPLAY=:0 + le socket X11 monté.
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
  ! tee name=t \
  t. ! queue ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" \
       ! nveglglessink sync=false async=false \
  t. ! queue ! nvvideoconvert \
       ! "video/x-raw,format=I420" \
       ! jpegenc quality=80 \
       ! image/jpeg \
       ! multipartmux boundary=avtowan-mjpeg \
       ! tcpserversink host=0.0.0.0 port="${PREVIEW_PORT}" \
                       sync=false async=false
