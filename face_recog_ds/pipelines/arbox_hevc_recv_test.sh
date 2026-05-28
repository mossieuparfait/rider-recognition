#!/bin/bash
# Test reception HEVC sur arbox depuis sender studio.
# Pipeline : udpsrc → tsdemux → h265parse → nvv4l2decoder → fakesink.
# Compte les frames + valide que NVDEC décode bien le flux.

export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/x86_64-linux-gnu/gstreamer-1.0

PORT="${PORT:-5000}"

gst-launch-1.0 -v \
    udpsrc port="${PORT}" buffer-size=8388608 \
  ! tsdemux \
  ! h265parse \
  ! nvv4l2decoder \
  ! "video/x-raw(memory:NVMM)" \
  ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false
