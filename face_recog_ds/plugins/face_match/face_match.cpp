// face_match.cpp — DeepStream custom classifier parser pour ArcFace.
//
// Logique :
//   1. Au 1er appel, charge embeddings.bin + names.txt dans une matrice
//      [N×512] résidente en VRAM. Pré-normalisé L2 par convert_index.py.
//   2. À chaque face crop processée par nvinfer secondary, le tensor de
//      sortie 512-D arrive en CPU memory (nvinfer fait le D2H pour les
//      classifiers — c'est petit donc négligeable).
//   3. Le query 512-D est normalisé L2 (CPU, 512 ops), uploadé GPU.
//   4. cublasSgemv calcule scores = embeddings × query (N produits
//      scalaires), équivalent cosine puisque tout est L2-normé.
//   5. cublasIsamax trouve le best index. Score copié D2H.
//   6. Si score >= threshold, on écrit l'attribut (name, score) dans
//      attrList → nvdsosd downstream l'affiche.
//
// Build : make -f Makefile (CUDA 12.6, DS 7.1 headers, cublas)
// Config nvinfer : parse-classifier-func-name=NvDsInferParseFaceMatch
//                  + custom-lib-path=/work/.../libface_match.so

#include "nvdsinfer_custom_impl.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int EMBED_DIM = 512;

struct FaceIndex {
    float* d_embeddings = nullptr;    // [N × DIM] row-major en VRAM
    float* d_query = nullptr;         // [DIM] scratch pour query par face
    float* d_scores = nullptr;        // [N] scratch pour scores
    std::vector<std::string> names;
    int n = 0;
    cublasHandle_t cublas = nullptr;
    bool loaded = false;
    std::mutex init_lock;
};

static FaceIndex g_index;

bool load_index_locked(const std::string& emb_path,
                       const std::string& names_path) {
    // Names d'abord (pour connaître N).
    std::ifstream nf(names_path);
    if (!nf.is_open()) {
        std::cerr << "[face_match] cannot open " << names_path << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(nf, line)) {
        if (!line.empty()) {
            g_index.names.push_back(line);
        }
    }
    g_index.n = static_cast<int>(g_index.names.size());
    if (g_index.n == 0) {
        std::cerr << "[face_match] empty names file" << std::endl;
        return false;
    }

    // Embeddings (raw float32).
    std::ifstream ef(emb_path, std::ios::binary | std::ios::ate);
    if (!ef.is_open()) {
        std::cerr << "[face_match] cannot open " << emb_path << std::endl;
        return false;
    }
    const size_t expected = static_cast<size_t>(g_index.n) *
                            EMBED_DIM * sizeof(float);
    const size_t size = static_cast<size_t>(ef.tellg());
    if (size != expected) {
        std::cerr << "[face_match] embeddings size mismatch : got " << size
                  << " expected " << expected << std::endl;
        return false;
    }
    ef.seekg(0);
    std::vector<float> h_embeddings(g_index.n * EMBED_DIM);
    ef.read(reinterpret_cast<char*>(h_embeddings.data()), expected);

    // Allocations GPU.
    if (cudaMalloc(&g_index.d_embeddings, expected) != cudaSuccess) {
        std::cerr << "[face_match] cudaMalloc embeddings failed" << std::endl;
        return false;
    }
    cudaMemcpy(g_index.d_embeddings, h_embeddings.data(), expected,
               cudaMemcpyHostToDevice);
    cudaMalloc(&g_index.d_query, EMBED_DIM * sizeof(float));
    cudaMalloc(&g_index.d_scores, g_index.n * sizeof(float));

    cublasCreate(&g_index.cublas);

    g_index.loaded = true;
    std::cerr << "[face_match] loaded " << g_index.n
              << " embeddings × " << EMBED_DIM
              << " into VRAM (" << (expected / (1024 * 1024)) << " MB)"
              << std::endl;
    return true;
}

bool ensure_loaded() {
    if (g_index.loaded) return true;
    std::lock_guard<std::mutex> lock(g_index.init_lock);
    if (g_index.loaded) return true;
    const char* emb_env = std::getenv("FACE_INDEX_EMBEDDINGS");
    const char* names_env = std::getenv("FACE_INDEX_NAMES");
    const std::string emb_path = emb_env
        ? emb_env : "/work/index/embeddings.bin";
    const std::string names_path = names_env
        ? names_env : "/work/index/names.txt";
    return load_index_locked(emb_path, names_path);
}

}  // namespace

extern "C"
bool NvDsInferParseFaceMatch(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
    (void)networkInfo;
    if (outputLayersInfo.empty()) return false;
    if (!ensure_loaded()) return false;

    const auto& layer = outputLayersInfo[0];
    if (layer.buffer == nullptr) return false;
    if (layer.inferDims.numElements != EMBED_DIM) {
        std::cerr << "[face_match] unexpected output dim "
                  << layer.inferDims.numElements
                  << " (expected " << EMBED_DIM << ")" << std::endl;
        return false;
    }

    const float* h_query = static_cast<const float*>(layer.buffer);

    // L2 normalize sur CPU (cheap, 1024 flops). On évite ainsi de coder
    // un kernel CUDA pour ça.
    float norm = 0.f;
    for (int i = 0; i < EMBED_DIM; ++i) norm += h_query[i] * h_query[i];
    norm = std::sqrt(norm);
    if (norm < 1e-6f) return false;
    float query_norm[EMBED_DIM];
    const float inv = 1.0f / norm;
    for (int i = 0; i < EMBED_DIM; ++i) query_norm[i] = h_query[i] * inv;

    // Upload query en VRAM (2 KB, négligeable).
    cudaMemcpy(g_index.d_query, query_norm, EMBED_DIM * sizeof(float),
               cudaMemcpyHostToDevice);

    // scores = embeddings × query.
    // cuBLAS = col-major. Nos embeddings sont row-major [N×DIM] = vue
    // col-major [DIM×N]. Pour calculer scores[N] = A_row[N×DIM] × q[DIM]
    // on fait op_T sur la vue col-major [DIM×N] → A_T × q en col-major
    // donne le scores[N] en col-major (= [N×1]).
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemv(g_index.cublas, CUBLAS_OP_T,
                EMBED_DIM, g_index.n,
                &alpha, g_index.d_embeddings, EMBED_DIM,
                g_index.d_query, 1,
                &beta, g_index.d_scores, 1);

    // argmax.
    int best_1based = 0;
    cublasIsamax(g_index.cublas, g_index.n, g_index.d_scores, 1,
                 &best_1based);
    const int best = best_1based - 1;  // cuBLAS Isamax = 1-indexé
    if (best < 0 || best >= g_index.n) return false;

    // Lire le score back.
    float best_score = 0.f;
    cudaMemcpy(&best_score, g_index.d_scores + best, sizeof(float),
               cudaMemcpyDeviceToHost);

    if (best_score < classifierThreshold) {
        // Sous le seuil = pas de match confiant. On retourne true mais
        // sans attribut → nvdsosd n'affichera pas de nom.
        return true;
    }

    NvDsInferAttribute attr;
    attr.attributeIndex = 0;
    attr.attributeValue = static_cast<unsigned int>(best);
    attr.attributeConfidence = best_score;
    attr.attributeLabel = strdup(g_index.names[best].c_str());
    attrList.push_back(attr);
    descString = g_index.names[best];

    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseFaceMatch);
