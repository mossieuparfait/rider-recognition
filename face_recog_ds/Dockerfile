# Custom DeepStream image avec NDI source plugin baked in.
# Évite les bind-mounts + LD_LIBRARY_PATH global qui cassent les autres
# utilitaires (sed, etc.) du container.

FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

# 0. Support mDNS dans le resolver libc.
#    Sans ça, le container ne résout pas les hostnames .local que NDI
#    utilise pour annoncer ses sources via avahi → la connexion au flux
#    échoue malgré que la discovery aille bien.
#    Installation de libnss-mdns + retire les repos qui causent des
#    erreurs au build (librealsense, jammy-security a parfois des
#    refresh issues).
RUN sed -i 's/jammy-security/jammy/g' /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/librealsense* 2>/dev/null || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends libnss-mdns avahi-utils && \
    rm -rf /var/lib/apt/lists/*

# 1. Install libndi via ldconfig dans un répertoire dédié
#    (pas /usr/local/lib qui mélangerait avec d'autres libs).
COPY ndi-libs/ /opt/ndi-runtime/
RUN echo "/opt/ndi-runtime" > /etc/ld.so.conf.d/ndi.conf && ldconfig

# 2. Install le plugin GStreamer ndi (Rust-compiled host-side).
COPY libgstndi.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/

# 3. Force le scan des plugins GStreamer (peuple le registry).
RUN GST_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gstreamer-1.0 \
    NDI_RUNTIME_DIR_V6=/opt/ndi-runtime \
    gst-inspect-1.0 ndisrc 2>&1 | head -5

# Env vars permanents pour le runtime (pas de LD_LIBRARY_PATH global,
# libndi est trouvée via ldconfig direct).
ENV NDI_RUNTIME_DIR_V6=/opt/ndi-runtime

# Pour utiliser cette image :
#   sudo docker run --rm --gpus all \
#     --network=host \
#     --security-opt apparmor=unconfined \
#     -v /var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket \
#     -v /var/run/avahi-daemon:/var/run/avahi-daemon \
#     -v /opt/face-recog-ds:/work \
#     face-recog-ds:phase1 <command>
#
# Les 2 mounts socket + apparmor=unconfined sont nécessaires pour que
# libndi puisse résoudre studio.local via avahi-daemon du host.
