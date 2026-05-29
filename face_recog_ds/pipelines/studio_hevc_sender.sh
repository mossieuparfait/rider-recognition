#!/bin/bash
# Studio HEVC sender — Magewell SDI /dev/video0 → NVENC HEVC Main10 50 Mbps
# → MPEG-TS sur UDP vers arbox.
#
# Lance via systemd unit dédiée (ne pas exécuter en parallèle d'un
# autre consommateur de /dev/video0 — V4L2 = single-open).
#
# Params NVENC tuned pour reconnaissance face broadcast :
# - 50 Mbps CBR : niveau contribution broadcast, indiscernable du source
#   sur visages, latence basse, footprint LAN raisonnable (50% GbE)
# - Main10 4:2:0 : 10-bit dépth pour préserver le grain peau / dégradés
# - preset p5 + tune ll : équilibre qualité/latence, no B-frames
# - GOP 60 (1s @ 60fps) : low latency, recovery rapide si paquet perdu
# - delay 0 : pas de bufferisation encodeur
# - MPEG-TS sur UDP : conteneur self-contained (SPS/PPS inline), simple
#   à decoder côté arbox via tsdemux + nvv4l2decoder DeepStream

DST_HOST="${DST_HOST:-192.168.1.175}"
DST_PORT="${DST_PORT:-5000}"
BITRATE="${BITRATE:-50M}"
DEVICE="${DEVICE:-/dev/video0}"

exec ffmpeg -hide_banner -loglevel info \
    -f v4l2 -framerate 60 -video_size 1920x1080 -i "${DEVICE}" \
    -c:v hevc_nvenc \
    -preset p5 -tune ll -profile:v main10 -pix_fmt p010le \
    -b:v "${BITRATE}" -maxrate "${BITRATE}" -bufsize 2M -rc cbr \
    -bf 0 -g 60 -delay 0 -no-scenecut 1 \
    -f mpegts "udp://${DST_HOST}:${DST_PORT}?pkt_size=1316"
