# Custom DeepStream image avec NDI source plugin baked in.
# Évite les bind-mounts + LD_LIBRARY_PATH global qui cassent les autres
# utilitaires (sed, etc.) du container.

FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

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
