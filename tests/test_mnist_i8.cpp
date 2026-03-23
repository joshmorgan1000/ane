/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_mnist_i8.cpp
 * @brief Tests INT8 quantized forward pass with FP32 backward pass (mixed-precision MNIST training).
 *
 * Forward pass: raw uint8 pixels via dense_u8s8 (USMOPA) for first dense, then
 *   quantize_fp32_i8 + dense_i8 (SMOPA) for second dense. Model parameters W1/W2 are
 *   quantized and packed every REPACK_INTERVAL batches (not per-batch). Input pixels
 *   skip the float normalization + re-quantization pipeline entirely.
 * Backward pass: dense_fp32, transpose_fp32, relu_backward_fp32 in FP32.
 * Prediction counting: count_matches bytecode dispatch (no scalar loops).
 *
 * @author Josh Morgan (@joshmorgan1000 on GitHub) with help from Claude and Gemini
 * Released under the MIT License
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <cfloat>
#include <thread>
#include <atomic>
#include <simd/simd.h>
#include <ane/ane.hpp>

/** --------------------------------------------------------------------------------------------------------- Constants
 * @brief MNIST geometry and dimension constants.
 *   input (784) -> dense+relu (128) -> dense+softmax (10)
 *   INP_K: input dimension padded to 64 for INT8 dense K alignment (832)
 *   HID_K: hidden dimension padded to 64 (128, already aligned)
 *   OUT_PAD: output dimension padded to 16 for SVLs alignment
 */
static constexpr int INPUT_DIM = 784;
static constexpr int HIDDEN_DIM = 128;
static constexpr int OUTPUT_DIM = 10;
static constexpr int TRAIN_IMAGES = 60000;
static constexpr int TEST_IMAGES = 10000;
static constexpr int BATCH_SIZE = 50;
static constexpr int INP_K = ((INPUT_DIM + 63) / 64) * 64;   ///< 832 -- K pad for dense input
static constexpr int HID_K = ((HIDDEN_DIM + 63) / 64) * 64;  ///< 128 -- K pad for dense hidden
static constexpr int OUT_PAD = ((OUTPUT_DIM + 15) / 16) * 16; ///< 16  -- output padded to SVLs
static constexpr float U8_SCALE = 1.0f / 255.0f;              ///< Dequant scale for raw uint8 [0,255] -> [0,1]
alignas(64) static const float zero_bias[INP_K] = {};         ///< Zero-filled bias for dense ops
alignas(64) static const float logit_bias[OUT_PAD] = {        ///< Bias that masks padding logit columns
    0,0,0,0,0,0,0,0,0,0, -1000.0f,-1000.0f,-1000.0f,-1000.0f,-1000.0f,-1000.0f
};
/** --------------------------------------------------------------------------------------------------------- Padding Helpers
 * @brief Round up to the next multiple of 16 (SVLs) or 32 (tile group).
 */
static int pad16(int n) { return ((n + 15) >> 4) << 4; }
static int pad32(int n) { return ((n + 31) >> 5) << 5; }
/** --------------------------------------------------------------------------------------------------------- Aligned Allocation
 * @brief Allocates 64-byte aligned memory for the given size.
 */
static void* alloc64(size_t size) {
    if (size & 63) size = ((size + 63) >> 6) << 6;
    return aligned_alloc(64, size);
}
/** --------------------------------------------------------------------------------------------------------- Free Deleter
 * @struct FreeDeleter
 * @brief Custom deleter for aligned memory allocated with alloc64, used with std::unique_ptr.
 */
struct FreeDeleter { void operator()(void* p) const { free(p); } };
/** --------------------------------------------------------------------------------------------------------- Normalize Images
 * @brief Converts raw uint8 MNIST images to float [0,1] using SIMD, writing into a pre-zeroed buffer.
 *  Used only for the backward pass which requires FP32 input for gradient computation.
 * @param src  Raw uint8 image data (count x 784)
 * @param dst  Pre-zeroed float output buffer (count x stride)
 * @param count Number of images to normalize
 * @param stride Padded row stride in floats (INP_K = 832 for i8 alignment)
 */
static void normalize_images(const uint8_t* src, float* dst, int count, int stride) {
    simd_float16 inv255 = 1.0f / 255.0f;
    for (int s = 0; s < count; s++) {
        const simd_uchar64* in = (const simd_uchar64*)&src[s * INPUT_DIM];
        simd_float16* out = (simd_float16*)&dst[s * stride];
        for (int i = 0; i < INPUT_DIM / 64; i++) {
            simd_uchar16* q = (simd_uchar16*)&in[i];
            out[i * 4]     = simd_float(q[0]) * inv255;
            out[i * 4 + 1] = simd_float(q[1]) * inv255;
            out[i * 4 + 2] = simd_float(q[2]) * inv255;
            out[i * 4 + 3] = simd_float(q[3]) * inv255;
        }
        simd_uchar16 tail = *(simd_uchar16*)&src[s * INPUT_DIM + 768];
        out[48] = simd_float(tail) * inv255;
    }
}
/** --------------------------------------------------------------------------------------------------------- Pad U8 Images
 * @brief Pads raw uint8 MNIST images to INP_K stride for direct use by dense_u8s8.
 *  Zero-pads each row from INPUT_DIM (784) to INP_K (832) so the K dimension is 64-aligned.
 * @param src  Raw uint8 image data (count x 784, contiguous)
 * @param dst  Pre-zeroed uint8 output buffer (count x INP_K)
 * @param count Number of images to pad
 */
static void pad_u8_images(const uint8_t* src, uint8_t* dst, int count) {
    simd_uchar64 zero64 = 0;
    for (int s = 0; s < count; s++) {
        const simd_uchar64* in = (const simd_uchar64*)&src[s * INPUT_DIM];
        simd_uchar64* out = (simd_uchar64*)&dst[s * INP_K];
        out[0]  = in[0];
        out[1]  = in[1];
        out[2]  = in[2];
        out[3]  = in[3];
        out[4]  = in[4];
        out[5]  = in[5];
        out[6]  = in[6];
        out[7]  = in[7];
        out[8]  = in[8];
        out[9]  = in[9];
        out[10] = in[10];
        out[11] = in[11];
        // Bytes 768..783 (16 bytes) + zero-pad rest of row 12
        simd_uchar64 tail = zero64;
        __builtin_memcpy(&tail, &src[s * INPUT_DIM + 768], 16);
        out[12] = tail;
    }
}
/** --------------------------------------------------------------------------------------------------------- Load Images
 * @brief Loads MNIST images from the given file path.
 * @param path File path to MNIST images file
 * @param out Vector to store the loaded images
 * @param n Expected number of images
 * @return True if loaded successfully
 */
static bool load_images(const char* path, std::vector<uint8_t>& out, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t h[4];
    fread(h, 4, 4, f);
    for (int i = 0; i < 4; i++) h[i] = __builtin_bswap32(h[i]);
    if (h[0] != 2051 || (int)h[1] != n) { fclose(f); return false; }
    out.resize(n * 784);
    fread(out.data(), 1, n * 784, f);
    fclose(f);
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Load Labels
 * @brief Loads MNIST labels from the given file path.
 * @param path File path to MNIST labels file
 * @param out Buffer to store the loaded labels
 * @param n Expected number of labels
 * @return True if loaded successfully
 */
static bool load_labels(const char* path, uint8_t* out, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) [[unlikely]] { return false; }
    uint32_t h[2];
    fread(h, 4, 2, f);
    for (int i = 0; i < 2; i++) h[i] = __builtin_bswap32(h[i]);
    if (h[0] != 2049 || (int)h[1] != n) { fclose(f); return false; }
    fread(out, 1, n, f);
    fclose(f);
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Packed B Size
 * @brief Computes the byte size of a packed B matrix for dense_i8 / dense_u8s8.
 * @param K Number of rows in B (will be padded to multiple of 4)
 * @param N Number of columns in B (will be padded to multiple of 16)
 * @return Size in bytes of the packed representation
 */
static int packed_b_size(int K, int N) {
    int k_groups = ((K + 3) / 4 * 4) / 4;
    int panels = ((N + 15) / 16 * 16) / 16;
    return panels * k_groups * 64;
}
/** --------------------------------------------------------------------------------------------------------- Shared Packed Params
 * @struct SharedPackedParams
 * @brief Pre-quantized and pre-packed model parameter buffers shared across all workers.
 *  Quantized+packed every REPACK_INTERVAL batches instead of per-batch (75x reduction).
 */
struct SharedPackedParams {
    int8_t* W1_i8;         ///< INP_K x HIDDEN_DIM quantized first-dense model params
    int8_t* W1_packed;     ///< Packed W1 in SMOPA panel format
    int8_t* W2_i8;         ///< HID_K x OUT_PAD quantized second-dense model params
    int8_t* W2_packed;     ///< Packed W2 in SMOPA panel format
    float scale_W1;        ///< W1 quantization scale
    float scale_W2;        ///< W2 quantization scale
    static SharedPackedParams create() {
        SharedPackedParams p{};
        p.W1_i8     = (int8_t*)alloc64(INP_K * HIDDEN_DIM);
        p.W1_packed = (int8_t*)alloc64(packed_b_size(INP_K, HIDDEN_DIM));
        p.W2_i8     = (int8_t*)alloc64(HID_K * OUT_PAD);
        p.W2_packed = (int8_t*)alloc64(packed_b_size(HID_K, OUT_PAD));
        p.scale_W1  = 1.0f;
        p.scale_W2  = 1.0f;
        return p;
    }
    /** ------------------------------------------------------------------------------------------------- Pack Params
     * @brief Quantizes and packs W1 and W2 from FP32 into INT8 SMOPA panel format.
     *  Called every REPACK_INTERVAL batches (not per-batch).
     * @param W1 FP32 first-dense model params (INP_K x HIDDEN_DIM)
     * @param W2 FP32 second-dense model params (HIDDEN_DIM x OUT_PAD)
     */
    void pack(const float* W1, const float* W2) {
        ane::dispatch(ane::Op::quantize_fp32_i8,
            (uint32_t)(INP_K * HIDDEN_DIM), W1, W1_i8, &scale_W1);
        ane::dispatch(ane::Op::pack_b_i8,
            (uint32_t)INP_K, (uint32_t)HIDDEN_DIM, W1_i8, W1_packed);
        ane::dispatch(ane::Op::quantize_fp32_i8,
            (uint32_t)(HID_K * OUT_PAD), W2, W2_i8, &scale_W2);
        ane::dispatch(ane::Op::pack_b_i8,
            (uint32_t)HID_K, (uint32_t)OUT_PAD, W2_i8, W2_packed);
    }
    void destroy() {
        free(W1_i8); free(W1_packed);
        free(W2_i8); free(W2_packed);
    }
};
/** --------------------------------------------------------------------------------------------------------- Worker Context
 * @struct WorkerCtx
 * @brief Per-thread scratch buffers for mixed-precision forward/backward pass.
 * Forward: raw uint8 input via dense_u8s8 (USMOPA), quantized hidden via dense_i8 (SMOPA).
 * Backward: FP32 matmuls via dense_fp32. SGD updates applied Hogwild-style.
 */
struct WorkerCtx {
    int8_t* hidden_i8;     ///< B x HID_K quantized hidden forward output
    float* hidden;         ///< B x HID_K forward hidden output (fp32 from dense_u8s8)
    float* logits;         ///< B x OUT_PAD forward logits (fp32 from dense_i8)
    float* probs;          ///< B x OUT_PAD softmax output
    float* g_out;          ///< B x OUT_PAD output gradient (from softmax backward)
    float* g_hid;          ///< B x HID_K hidden gradient
    float* g_hid_r;        ///< B x HID_K ReLU-masked hidden gradient
    float* W2_T;           ///< Transposed W2 scratch (OUT_PAD x HIDDEN_DIM)
    float* x_T;            ///< INP_K x B transposed input batch
    float* h_T;            ///< HID_K x B transposed hidden
    float* g_W1_sample;    ///< Per-batch gradient scratch for W1 (INP_K x HIDDEN_DIM)
    float* g_W2_sample;    ///< Per-batch gradient scratch for W2 (HIDDEN_DIM x OUT_PAD)
    int32_t* pred;         ///< B prediction indices from argmax
    float scale_h;         ///< Per-batch hidden quantization scale
    static WorkerCtx create(int batch) {
        int bp = pad32(batch);
        int bp16 = pad16(batch);
        int op32 = pad32(OUT_PAD);
        WorkerCtx ctx{};
        ctx.hidden_i8   = (int8_t*)alloc64(bp * HID_K);
        ctx.hidden      = (float*)alloc64(bp * HID_K * sizeof(float));
        ctx.logits      = (float*)alloc64(bp * op32 * sizeof(float));
        ctx.probs       = (float*)alloc64(bp * op32 * sizeof(float));
        ctx.g_out       = (float*)alloc64(bp * op32 * sizeof(float));
        ctx.g_hid       = (float*)alloc64(bp * HID_K * sizeof(float));
        ctx.g_hid_r     = (float*)alloc64(bp * HID_K * sizeof(float));
        ctx.W2_T        = (float*)alloc64(op32 * HIDDEN_DIM * sizeof(float));
        ctx.x_T         = (float*)alloc64(INP_K * bp16 * sizeof(float));
        ctx.h_T         = (float*)alloc64(HID_K * bp16 * sizeof(float));
        ctx.g_W1_sample = (float*)alloc64(INP_K * HIDDEN_DIM * sizeof(float));
        ctx.g_W2_sample = (float*)alloc64(HIDDEN_DIM * op32 * sizeof(float));
        ctx.pred        = (int32_t*)alloc64(bp * sizeof(int32_t));
        memset(ctx.hidden_i8, 0, bp * HID_K);
        return ctx;
    }
    void destroy() {
        free(hidden_i8);
        free(hidden); free(logits); free(probs); free(g_out);
        free(g_hid); free(g_hid_r); free(W2_T); free(x_T); free(h_T);
        free(g_W1_sample); free(g_W2_sample); free(pred);
    }
};
/** --------------------------------------------------------------------------------------------------------- Worker Function
 * @brief Worker function for mixed-precision mini-batch SGD with Hogwild updates.
 * Forward pass: raw uint8 input via dense_u8s8 (first dense), quantized hidden via dense_i8
 * (second dense). Model params W1/W2 are pre-packed in SharedPackedParams.
 * Backward pass dispatches FP32 matmuls.
 * @param tid Thread ID
 * @param next_batch_index Atomic counter for grabbing next batch
 * @param seg_end Sample index upper bound for this segment
 * @param train_u8 Padded uint8 training images (TRAIN_IMAGES x INP_K)
 * @param train_fp Normalized FP32 training images for backward pass (TRAIN_IMAGES x INP_K)
 * @param train_lbl Training labels
 * @param W1 Shared first-dense model params (INP_K x HIDDEN_DIM fp32)
 * @param W2 Shared second-dense model params (HIDDEN_DIM x OUT_PAD fp32)
 * @param packed Pre-quantized+packed model params (shared, read-only within segment)
 * @param workers Per-thread scratch contexts
 * @param total_correct Atomic counter for correct predictions
 * @param batch_size Number of samples per batch
 * @param lr Learning rate
 */
static void worker(
    int tid,
    std::atomic<uint64_t>* next_batch_index,
    uint64_t seg_end,
    const uint8_t* train_u8,
    const float* train_fp,
    const uint8_t* train_lbl,
    float* W1,
    float* W2,
    const SharedPackedParams* packed,
    WorkerCtx* workers,
    std::atomic<int>* total_correct,
    int batch_size,
    float lr
) {
    auto& ctx = workers[tid];
    while (true) {
        size_t batch_start = next_batch_index->fetch_add(batch_size, std::memory_order_acquire);
        if (batch_start + batch_size > seg_end) [[unlikely]] { return; }
        const uint8_t* xi_u8 = &train_u8[batch_start * INP_K];
        const float* xi_fp = &train_fp[batch_start * INP_K];
        const uint8_t* batch_labels = &train_lbl[batch_start];
        // -- Forward: raw uint8 input x pre-packed W1 via dense_u8s8 (USMOPA, relu) --
        ane::dispatch(ane::Op::dense_u8s8,
            batch_size, HIDDEN_DIM, INP_K,
            U8_SCALE, packed->scale_W1, true,
            xi_u8, packed->W1_packed, zero_bias, ctx.hidden);
        // -- Forward: quantize hidden -> i8, then hidden_i8 x pre-packed W2 via dense_i8 --
        ane::dispatch(ane::Op::quantize_fp32_i8,
            (uint32_t)(batch_size * HID_K), ctx.hidden, ctx.hidden_i8, &ctx.scale_h);
        ane::dispatch(ane::Op::dense_i8,
            batch_size, OUT_PAD, HID_K,
            ctx.scale_h, packed->scale_W2, false,
            ctx.hidden_i8, packed->W2_packed, logit_bias, ctx.logits);
        // -- Fused softmax + cross-entropy backward + argmax --
        ane::dispatch(ane::Op::softmax_argmax_fp32,
            batch_size, OUT_PAD, ctx.logits, ctx.probs, batch_labels, ctx.g_out, ctx.pred);
        // -- Count correct predictions (bytecode dispatch, no scalar loop) --
        int32_t correct = 0;
        ane::dispatch(ane::Op::count_matches,
            (uint32_t)batch_size, ctx.pred, batch_labels, &correct);
        total_correct->fetch_add(correct, std::memory_order_relaxed);
        // -- Backward: transpose hidden(BxHID) -> h_T(HIDxB), h_T @ g_out -> g_W2 --
        ane::dispatch(ane::Op::transpose_fp32, batch_size, HIDDEN_DIM, ctx.hidden, ctx.h_T);
        ane::dispatch(ane::Op::dense_fp32,
            HIDDEN_DIM, OUT_PAD, batch_size,
            1.0f, false, ctx.h_T, ctx.g_out, zero_bias, ctx.g_W2_sample);
        // -- Backward: transpose W2 -> W2_T(OUTxHID), g_out @ W2_T -> g_hid --
        ane::dispatch(ane::Op::transpose_fp32, HIDDEN_DIM, OUT_PAD, W2, ctx.W2_T);
        ane::dispatch(ane::Op::dense_fp32,
            batch_size, HIDDEN_DIM, OUT_PAD,
            1.0f, false, ctx.g_out, ctx.W2_T, zero_bias, ctx.g_hid);
        // -- ReLU backward: g_hid_r = mask(hidden > 0) * g_hid --
        ane::dispatch(ane::Op::relu_backward_fp32,
            (uint32_t)(batch_size * HID_K), ctx.hidden, ctx.g_hid, ctx.g_hid_r);
        // -- Backward: transpose xi(BxINP_K) -> x_T(INP_KxB), x_T @ g_hid_r -> g_W1 --
        ane::dispatch(ane::Op::transpose_fp32, batch_size, INP_K, xi_fp, ctx.x_T);
        ane::dispatch(ane::Op::dense_fp32,
            INP_K, HIDDEN_DIM, batch_size,
            1.0f, false, ctx.x_T, ctx.g_hid_r, zero_bias, ctx.g_W1_sample);
        // -- SGD update: W -= (lr/B) * gradient (Hogwild, races OK) --
        float step = -lr / batch_size;
        ane::dispatch(ane::Op::elementwise_scaled_add_fp32,
            (uint32_t)(INP_K * HIDDEN_DIM), step, W1, ctx.g_W1_sample, W1);
        ane::dispatch(ane::Op::elementwise_scaled_add_fp32,
            (uint32_t)(HIDDEN_DIM * OUT_PAD), step, W2, ctx.g_W2_sample, W2);
    }
}
/** --------------------------------------------------------------------------------------------------------- Main
 */
int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 1;
    printf("MNIST INT8 Training Demo — Mixed-Precision Mini-batch SGD (%d threads, batch=%d)\n",
        num_threads, BATCH_SIZE);
    printf("Network: 784 -> 128 (ReLU) -> 10 (Softmax)\n");
    printf("Forward: raw uint8 via USMOPA (dense_u8s8) + INT8 via SMOPA (dense_i8)\n");
    printf("Backward: FP32 matmul via FMOPA (dense_fp32)\n\n");
    std::string data_dir = "data/mnist";
    if (argc > 1) data_dir = argv[1];
    printf("Loading MNIST from %s...\n", data_dir.c_str());
    std::vector<uint8_t> train_img, test_img;
    std::unique_ptr<uint8_t[]> train_lbl(new uint8_t[TRAIN_IMAGES]);
    std::unique_ptr<uint8_t[]> test_lbl(new uint8_t[TEST_IMAGES]);
    if (!load_images((data_dir + "/train-images-idx3-ubyte").c_str(), train_img, TRAIN_IMAGES) ||
        !load_labels((data_dir + "/train-labels-idx1-ubyte").c_str(), train_lbl.get(), TRAIN_IMAGES) ||
        !load_images((data_dir + "/t10k-images-idx3-ubyte").c_str(), test_img, TEST_IMAGES) ||
        !load_labels((data_dir + "/t10k-labels-idx1-ubyte").c_str(), test_lbl.get(), TEST_IMAGES)) {
        printf("Failed. Run: python3 scripts/download_mnist.py\n");
        return 1;
    }
    printf("  %d train, %d test\n\n", TRAIN_IMAGES, TEST_IMAGES);
    // -- Allocate padded uint8 buffers for forward pass (raw bytes, no float normalization) --
    int train_u8_alloc = (TRAIN_IMAGES + pad16(BATCH_SIZE)) * INP_K;
    int test_u8_alloc = (TEST_IMAGES + pad16(BATCH_SIZE)) * INP_K;
    std::unique_ptr<uint8_t[], FreeDeleter> train_u8((uint8_t*)alloc64(train_u8_alloc));
    std::unique_ptr<uint8_t[], FreeDeleter> test_u8((uint8_t*)alloc64(test_u8_alloc));
    memset(train_u8.get(), 0, train_u8_alloc);
    memset(test_u8.get(), 0, test_u8_alloc);
    pad_u8_images(train_img.data(), train_u8.get(), TRAIN_IMAGES);
    pad_u8_images(test_img.data(), test_u8.get(), TEST_IMAGES);
    printf("  Data padded to stride=%d for direct uint8 forward path\n", INP_K);
    // -- Allocate FP32 normalized buffers for backward pass only --
    int train_fp_alloc = (TRAIN_IMAGES + pad16(BATCH_SIZE)) * INP_K;
    std::unique_ptr<float[], FreeDeleter> train_fp((float*)alloc64(train_fp_alloc * sizeof(float)));
    memset(train_fp.get(), 0, train_fp_alloc * sizeof(float));
    normalize_images(train_img.data(), train_fp.get(), TRAIN_IMAGES, INP_K);
    printf("  FP32 normalized for backward pass (stride=%d)\n", INP_K);
    // -- Initialize model parameters --
    std::unique_ptr<float[]> W1(new float[INP_K * HIDDEN_DIM]);
    std::unique_ptr<float[]> W2(new float[HIDDEN_DIM * OUT_PAD]);
    memset(W1.get(), 0, INP_K * HIDDEN_DIM * sizeof(float));
    memset(W2.get(), 0, HIDDEN_DIM * OUT_PAD * sizeof(float));
    uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    {
        std::normal_distribution<float> d1(0, sqrtf(2.0f / INPUT_DIM));
        std::normal_distribution<float> d2(0, sqrtf(2.0f / HIDDEN_DIM));
        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) W1[i] = d1(rng);
        for (int r = 0; r < HIDDEN_DIM; r++) {
            for (int c = 0; c < OUTPUT_DIM; c++) W2[r * OUT_PAD + c] = d2(rng);
        }
    }
    // -- Create shared pre-packed model params and per-thread worker contexts --
    SharedPackedParams packed = SharedPackedParams::create();
    std::unique_ptr<WorkerCtx[]> workers(new WorkerCtx[num_threads]);
    for (int t = 0; t < num_threads; t++) workers[t] = WorkerCtx::create(BATCH_SIZE);
    auto& eval_ctx = workers[0];
    const float lr = 0.01f;
    const int epochs = 10;
    int num_batches = TRAIN_IMAGES / BATCH_SIZE;
    printf("Training: %d epochs, lr=%g, batch=%d, %d batches/epoch, %d threads\n\n",
        epochs, lr, BATCH_SIZE, num_batches, num_threads);
    static constexpr int REPACK_INTERVAL = 16;   ///< Repack model params every N batches (1 per thread)
    int segments_per_epoch = (num_batches + REPACK_INTERVAL - 1) / REPACK_INTERVAL;
    for (int ep = 0; ep < epochs; ep++) {
        // -- Shuffle training data in-place (Fisher-Yates on rows, both u8 and fp32) --
        for (int i = TRAIN_IMAGES - 1; i > 0; i--) {
            int j = std::uniform_int_distribution<int>(0, i)(rng);
            std::swap_ranges(
                &train_u8[i * INP_K], &train_u8[(i + 1) * INP_K], &train_u8[j * INP_K]);
            std::swap_ranges(
                &train_fp[i * INP_K], &train_fp[(i + 1) * INP_K], &train_fp[j * INP_K]);
            std::swap(train_lbl[i], train_lbl[j]);
        }
        std::atomic<int> total_correct(0);
        uint64_t start_time = std::chrono::steady_clock::now().time_since_epoch().count();
        // -- Process epoch in segments, repacking model params between segments --
        int batches_done = 0;
        for (int seg = 0; seg < segments_per_epoch; seg++) {
            int seg_batches = std::min(REPACK_INTERVAL, num_batches - batches_done);
            // Repack W1/W2 at the start of each segment
            packed.pack(W1.get(), W2.get());
            std::atomic<uint64_t> next_batch_index(batches_done * BATCH_SIZE);
            uint64_t seg_end = (uint64_t)(batches_done + seg_batches) * BATCH_SIZE;
            std::vector<std::thread> threads;
            for (int t = 1; t < num_threads; t++) threads.emplace_back(
                &worker, t, &next_batch_index, seg_end,
                train_u8.get(), train_fp.get(), train_lbl.get(),
                W1.get(), W2.get(), &packed, workers.get(), &total_correct,
                BATCH_SIZE, lr
            );
            worker(0, &next_batch_index, seg_end,
                train_u8.get(), train_fp.get(), train_lbl.get(),
                W1.get(), W2.get(), &packed, workers.get(), &total_correct,
                BATCH_SIZE, lr
            );
            for (auto& t : threads) t.join();
            batches_done += seg_batches;
        }
        uint64_t end_time = std::chrono::steady_clock::now().time_since_epoch().count();
        float samples_per_sec = (float)(num_batches * BATCH_SIZE) / ((end_time - start_time) / 1e9);
        printf("  Epoch %d done: acc=%.2f%%, throughput=%.2f samples/sec, time=%.2f sec\n",
            ep + 1,
            100.0f * total_correct.load(std::memory_order_acquire) / (num_batches * BATCH_SIZE),
            samples_per_sec, (end_time - start_time) / 1e9);
        // -- Eval: u8s8 forward pass (single-threaded) --
        // Re-pack model params with current post-SGD state for eval
        packed.pack(W1.get(), W2.get());
        int tc = 0;
        for (int s = 0; s < TEST_IMAGES; s++) {
            const uint8_t* xi_u8 = &test_u8.get()[s * INP_K];
            // dense_u8s8: raw uint8 input x W1_packed -> hidden (relu)
            ane::dispatch(ane::Op::dense_u8s8,
                1, HIDDEN_DIM, INP_K,
                U8_SCALE, packed.scale_W1, true,
                xi_u8, packed.W1_packed, zero_bias, eval_ctx.hidden);
            // Quantize hidden
            ane::dispatch(ane::Op::quantize_fp32_i8,
                (uint32_t)HID_K, eval_ctx.hidden, eval_ctx.hidden_i8, &eval_ctx.scale_h);
            // dense_i8: hidden_i8 x W2_packed -> logits (no relu, logit_bias masks padding)
            ane::dispatch(ane::Op::dense_i8,
                1, OUT_PAD, HID_K,
                eval_ctx.scale_h, packed.scale_W2, false,
                eval_ctx.hidden_i8, packed.W2_packed, logit_bias, eval_ctx.logits);
            // Softmax + argmax (single sample)
            int32_t pred = 0;
            float g_out_dummy[16] = {};
            ane::dispatch(ane::Op::softmax_argmax_fp32,
                1, OUT_PAD, eval_ctx.logits, eval_ctx.probs,
                &test_lbl[s], g_out_dummy, &pred);
            if (pred == (int32_t)test_lbl[s]) tc++;
        }
        printf("  Test: %.2f%% (%d/%d)\n\n", 100.0f * tc / TEST_IMAGES, tc, TEST_IMAGES);
    }
    printf("Training complete.\n");
    packed.destroy();
    for (int t = 0; t < num_threads; t++) workers[t].destroy();
    return 0;
}
