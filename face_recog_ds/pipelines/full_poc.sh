#!/bin/bash
# Full POC : HEVC receive → NVDEC → nvinfer YOLOv8L-Face → nvdsosd → fakesink
# Mesure GPU usage pendant la détection face.

export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/x86_64-linux-gnu/gstreamer-1.0

PORT="${PORT:-5000}"
WORK=/work

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
  ! nvvideoconvert \
  ! nvdsosd \
  ! nvvideoconvert \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false
