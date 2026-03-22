/** --------------------------------------------------------------------------------------------------------- File Info
 * @file test_mnist.cpp
 * @brief Tests the library against the MNIST dataset with multi-threaded mini-batch SGD.
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
 * @brief MNIST geometry and SME tile constants.
 *   input (784) → dense+relu (128) → dense+softmax (10)
 */
static constexpr int SVLs = 16;
static constexpr int INPUT_DIM = 784;
static constexpr int HIDDEN_DIM = 128;
static constexpr int OUTPUT_DIM = 10;
static constexpr int TRAIN_IMAGES = 60000;
static constexpr int TEST_IMAGES = 10000;
static constexpr int BATCH_SIZE = 50;
/** --------------------------------------------------------------------------------------------------------- Utility Functions
 * @brief Pads the given count up to the next multiple of 16 (SME SVL) for buffer allocation.
 * @param count The original count of elements.
 * @return The padded count, rounded up to the nearest multiple of 16.
 */
static int pad_count(int count) {
    return ((count + 15) >> 4) << 4;
}
/** --------------------------------------------------------------------------------------------------------- Aligned Allocation
 * @brief Allocates 64-byte aligned memory for the given size.
 */
static void* alloc64(size_t size) {
    if (size & 63) {
        size = ((size + 63) >> 6) << 6;
    }
    return aligned_alloc(64, size);
}
/** --------------------------------------------------------------------------------------------------------- Worker Context
 * @struct WorkerCtx
 * @brief Per-thread scratch buffers for forward/backward pass with batch-sized allocations.
 * SGD updates are applied per-batch (Hogwild style) directly to shared model parameters.
 */
struct WorkerCtx {
    float* hidden;       ///< B × hid_pad forward hidden activations
    float* logits;       ///< B × out_pad forward logits
    float* probs;        ///< B × out_pad softmax output
    float* g_out;        ///< B × out_pad output gradient
    float* g_hid;        ///< B × hid_pad hidden gradient
    float* g_hid_r;      ///< B × hid_pad ReLU-masked hidden gradient
    float* W2_T;         ///< Transposed W2 scratch (OUTPUT_DIM × HIDDEN_DIM)
    float* x_T;          ///< inp_pad × B transposed input batch
    float* h_T;          ///< hid_pad × B transposed hidden activations
    float* g_W1_sample;  ///< Per-batch gradient scratch for W1 (INPUT_DIM × HIDDEN_DIM)
    float* g_W2_sample;  ///< Per-batch gradient scratch for W2 (HIDDEN_DIM × OUTPUT_DIM)
    int32_t* pred;       ///< B prediction indices from argmax
    static WorkerCtx create(int batch, int inp_pad, int hid_pad, int out_pad) {
        return {
            (float*)alloc64(batch * hid_pad * sizeof(float)),
            (float*)alloc64(batch * out_pad * sizeof(float)),
            (float*)alloc64(batch * out_pad * sizeof(float)),
            (float*)alloc64(batch * out_pad * sizeof(float)),
            (float*)alloc64(batch * hid_pad * sizeof(float)),
            (float*)alloc64(batch * hid_pad * sizeof(float)),
            (float*)alloc64(out_pad * HIDDEN_DIM * sizeof(float)),
            (float*)alloc64(inp_pad * batch * sizeof(float)),
            (float*)alloc64(hid_pad * batch * sizeof(float)),
            (float*)alloc64(INPUT_DIM * HIDDEN_DIM * sizeof(float)),
            (float*)alloc64(HIDDEN_DIM * out_pad * sizeof(float)),
            (int32_t*)alloc64(batch * sizeof(int32_t)),
        };
    }
    void destroy() {
        free(hidden); free(logits); free(probs); free(g_out);
        free(g_hid); free(g_hid_r); free(W2_T); free(x_T); free(h_T);
        free(g_W1_sample); free(g_W2_sample); free(pred);
    }
};
/** --------------------------------------------------------------------------------------------------------- MNIST Data Loading
 * @brief Loads MNIST images from the given file path into the provided vector.
 * @param path The file path to the MNIST images file (e.g., "train-images-idx3-ubyte").
 * @param out The vector to store the loaded images.
 * @param n The expected number of images to load.
 * @return True if the images were loaded successfully and the file format is correct, false otherwise.
 */
static bool load_images(const char* path, std::vector<uint8_t>& out, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint32_t h[4];
    fread(h, 4, 4, f);
    for (int i = 0; i < 4; i++) {
        h[i] = __builtin_bswap32(h[i]);
    }
    if (h[0] != 2051 || (int)h[1] != n) {
        fclose(f);
        return false;
    }
    out.resize(n * 784);
    fread(out.data(), 1, n * 784, f);
    fclose(f);
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Load Labels
 * @brief Loads MNIST labels from the given file path into the provided vector.
 * @param path The file path to the MNIST labels file (e.g., "train-labels-idx1-ubyte").
 * @param out The vector to store the loaded labels. It will be resized to hold 'n' labels.
 * @param n The expected number of labels to load.
 * @return True if the labels were loaded successfully and the file format is correct, false otherwise.
 */
static bool load_labels(const char* path, uint8_t* out, int n) {
    FILE* f = fopen(path, "rb");
    if (!f) [[unlikely]] {
        return false;
    }
    uint32_t h[2];
    fread(h, 4, 2, f);
    for (int i = 0; i < 2; i++) {
        h[i] = __builtin_bswap32(h[i]);
    }
    if (h[0] != 2049 || (int)h[1] != n) {
        fclose(f);
        return false;
    }
    fread(out, 1, n, f);
    fclose(f);
    return true;
}
/** --------------------------------------------------------------------------------------------------------- Worker Function
 * @brief Worker function for processing batches of MNIST data with Hogwild SGD.
 * Each thread grabs batches atomically and applies SGD updates directly to shared model parameters.
 * @param tid Thread ID
 * @param next_batch_index Atomic counter for grabbing next batch
 * @param idx Shuffled sample indices for this epoch
 * @param train_fp Normalized training images (TRAIN_IMAGES × inp_pad)
 * @param train_lbl Training labels
 * @param W1 Shared model parameters for first dense (INPUT_DIM × HIDDEN_DIM)
 * @param W2 Shared model parameters for second dense (HIDDEN_DIM × OUTPUT_DIM)
 * @param workers Per-thread scratch contexts
 * @param total_correct Atomic counter for correct predictions
 * @param inp_pad Padded input dimension
 * @param hid_pad Padded hidden dimension
 * @param out_pad Padded output dimension
 * @param batch_size Number of samples per batch
 * @param lr Learning rate
 */
static void worker(
    int tid,
    std::atomic<uint64_t>* next_batch_index,
    const float* train_fp,
    const uint8_t* train_lbl,
    float* W1,
    float* W2,
    WorkerCtx* workers,
    std::atomic<int>* total_correct,
    int inp_pad,
    int hid_pad,
    int out_pad,
    int batch_size,
    float lr
) {
    auto& ctx = workers[tid];
    while (true) {
        size_t batch_start = next_batch_index->fetch_add(batch_size, std::memory_order_acquire);
        if (batch_start + batch_size > TRAIN_IMAGES) [[unlikely]] {
            return;
        }
        const float* xi = &train_fp[batch_start * inp_pad];
        // ── Forward: xi(B×784) @ W1 → hidden(B×128) + relu ──
        memset(ctx.hidden, 0, batch_size * hid_pad * sizeof(float));
        ane::dispatch(
            ane::Op::dense_fused_i8, batch_size, HIDDEN_DIM, INPUT_DIM,
            1.0f, true, xi, W1, ctx.hidden
        );
        // ── Forward: hidden(B×128) @ W2 → logits(B×10) ──
        memset(ctx.logits, 0, batch_size * out_pad * sizeof(float));
        ane::dispatch(
            ane::Op::dense_fused_i8, batch_size, out_pad, HIDDEN_DIM,
            1.0f, false, ctx.hidden, W2, ctx.logits
        );
        // ── Pad logits columns 10-15 with -1000.0f per row (SIMD bitselect) ──
        {
            static const simd_int16 pad_mask = {0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1};
            static const simd_float16 neg_thousand = simd_float16(-1000.0f);
            for (int b = 0; b < batch_size; b++) {
                simd_float16& row = *(simd_float16*)&ctx.logits[b * out_pad];
                row = simd_bitselect(row, neg_thousand, pad_mask);
            }
        }
        // ── Fused softmax + cross-entropy backward + argmax ──
        const uint8_t* batch_labels = &train_lbl[batch_start];
        ane::dispatch(
            ane::Op::softmax_argmax_fp32, batch_size, out_pad,
            ctx.logits, ctx.probs, batch_labels, ctx.g_out, ctx.pred
        );
        int correct = 0;
        for (int b = 0; b < batch_size; b++) {
            correct += (ctx.pred[b] == (int32_t)batch_labels[b]);
        }
        total_correct->fetch_add(correct, std::memory_order_relaxed);
        // ── Backward: transpose hidden(B×128) → h_T(128×B), h_T @ g_out → g_W2_sample(128×10) ──
        ane::dispatch(ane::Op::transpose_fp32, batch_size, HIDDEN_DIM, ctx.hidden, ctx.h_T);
        memset(ctx.g_W2_sample, 0, HIDDEN_DIM * out_pad * sizeof(float));
        ane::dispatch(
            ane::Op::dense_fused_i8, HIDDEN_DIM, out_pad, batch_size,
            1.0f, false, ctx.h_T, ctx.g_out, ctx.g_W2_sample
        );
        // ── Backward: transpose W2 → W2_T(out_pad×128), g_out @ W2_T → g_hid(B×128) ──
        ane::dispatch(ane::Op::transpose_fp32, HIDDEN_DIM, out_pad, W2, ctx.W2_T);
        memset(ctx.g_hid, 0, batch_size * hid_pad * sizeof(float));
        ane::dispatch(
            ane::Op::dense_fused_i8, batch_size, HIDDEN_DIM, out_pad,
            1.0f, false, ctx.g_out, ctx.W2_T, ctx.g_hid
        );
        // ── ReLU backward: g_hid_r = mask(hidden) * g_hid ──
        ane::dispatch(
            ane::Op::relu_backward_fp32, batch_size * hid_pad,
            ctx.hidden, ctx.g_hid, ctx.g_hid_r
        );
        // ── Backward: transpose xi(B×784) → x_T(784×B), x_T @ g_hid_r → g_W1_sample(784×128) ──
        ane::dispatch(ane::Op::transpose_fp32, batch_size, INPUT_DIM, xi, ctx.x_T);
        memset(ctx.g_W1_sample, 0, INPUT_DIM * HIDDEN_DIM * sizeof(float));
        ane::dispatch(
            ane::Op::dense_fused_i8, INPUT_DIM, HIDDEN_DIM, batch_size,
            1.0f, false, ctx.x_T, ctx.g_hid_r, ctx.g_W1_sample
        );
        // ── SGD update: W -= (lr/B) * gradient (Hogwild, races OK) ──
        float step = -lr / batch_size;
        ane::dispatch(
            ane::Op::elementwise_scaled_add_fp32, INPUT_DIM * HIDDEN_DIM, step,
            W1, ctx.g_W1_sample, W1
        );
        ane::dispatch(
            ane::Op::elementwise_scaled_add_fp32, HIDDEN_DIM * out_pad, step,
            W2, ctx.g_W2_sample, W2
        );
    }
}
/** --------------------------------------------------------------------------------------------------------- Main
 */
int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads < 1) {
        num_threads = 1;
    }
    printf("MNIST Training Demo — Mini-batch SGD (%d threads, batch=%d)\n", num_threads, BATCH_SIZE);
    printf("Network: 784 → 128 (ReLU) → 10 (Softmax)\n");
    printf("All compute dispatched through SME streaming bytecode interpreter.\n\n");
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
    int inp_pad = pad_count(INPUT_DIM);
    int hid_pad = pad_count(HIDDEN_DIM);
    int out_pad = pad_count(OUTPUT_DIM);
    auto* train_fp = (float*)alloc64(TRAIN_IMAGES * inp_pad * sizeof(float));
    auto* test_fp  = (float*)alloc64(TEST_IMAGES * inp_pad * sizeof(float));
    memset(train_fp, 0, TRAIN_IMAGES * inp_pad * sizeof(float));
    memset(test_fp, 0, TEST_IMAGES * inp_pad * sizeof(float));
    {
        simd_float16 inv255 = 1.0f / 255.0f;
        simd_float16 half = 0.5f;
        for (int s = 0; s < TRAIN_IMAGES; s++) {
            const simd_uchar64* in = (const simd_uchar64*)&train_img[s * INPUT_DIM];
            simd_float16* out = (simd_float16*)&train_fp[s * inp_pad];
            for (int i = 0; i < INPUT_DIM / 64; i++) {
                simd_uchar16* q = (simd_uchar16*)&in[i];
                out[i * 4]     = simd_float(q[0]) * inv255 - half;
                out[i * 4 + 1] = simd_float(q[1]) * inv255 - half;
                out[i * 4 + 2] = simd_float(q[2]) * inv255 - half;
                out[i * 4 + 3] = simd_float(q[3]) * inv255 - half;
            }
            simd_uchar16 tail = *(simd_uchar16*)&train_img[s * INPUT_DIM + (INPUT_DIM / 64) * 64];
            out[INPUT_DIM / 16] = simd_float(tail) * inv255 - half;
        }
        for (int s = 0; s < TEST_IMAGES; s++) {
            const simd_uchar64* in = (const simd_uchar64*)&test_img[s * INPUT_DIM];
            simd_float16* out = (simd_float16*)&test_fp[s * inp_pad];
            for (int i = 0; i < INPUT_DIM / 64; i++) {
                simd_uchar16* q = (simd_uchar16*)&in[i];
                out[i * 4]     = simd_float(q[0]) * inv255 - half;
                out[i * 4 + 1] = simd_float(q[1]) * inv255 - half;
                out[i * 4 + 2] = simd_float(q[2]) * inv255 - half;
                out[i * 4 + 3] = simd_float(q[3]) * inv255 - half;
            }
            simd_uchar16 tail = *(simd_uchar16*)&test_img[s * INPUT_DIM + (INPUT_DIM / 64) * 64];
            out[INPUT_DIM / 16] = simd_float(tail) * inv255 - half;
        }
    }
    printf("  Data normalized\n");
    std::unique_ptr<float[]> W1(new float[INPUT_DIM * HIDDEN_DIM]);
    std::unique_ptr<float[]> W2(new float[HIDDEN_DIM * out_pad]);
    uint64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    {
        std::normal_distribution<float> d1(0, sqrtf(2.0f / INPUT_DIM)), d2(0, sqrtf(2.0f / HIDDEN_DIM));
        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) {
            W1[i] = d1(rng);
        }
        memset(W2.get(), 0, HIDDEN_DIM * out_pad * sizeof(float));
        for (int r = 0; r < HIDDEN_DIM; r++) {
            for (int c = 0; c < OUTPUT_DIM; c++) {
                W2[r * out_pad + c] = d2(rng);
            }
        }
    }
    std::unique_ptr<WorkerCtx[]> workers(new WorkerCtx[num_threads]);
    for (int t = 0; t < num_threads; t++) {
        workers[t] = WorkerCtx::create(BATCH_SIZE, inp_pad, hid_pad, out_pad);
    }
    auto& eval_ctx = workers[0];
    const float lr = 0.01f;
    const int epochs = 3;
    int num_batches = TRAIN_IMAGES / BATCH_SIZE;
    printf(
        "Training: %d epochs, lr=%g, batch=%d, %d batches/epoch, %d threads\n\n",
        epochs, lr, BATCH_SIZE, num_batches, num_threads
    );
    for (int ep = 0; ep < epochs; ep++) {
        std::atomic<int> total_correct(0);
        std::atomic<uint64_t> next_batch_index(0);
        std::vector<std::thread> threads;
        uint64_t start_time = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int t = 1; t < num_threads; t++) threads.emplace_back(
            &worker,
            t, &next_batch_index, train_fp, train_lbl.get(),
            W1.get(), W2.get(), workers.get(), &total_correct,
            inp_pad, hid_pad, out_pad, BATCH_SIZE, lr
        );
        worker(
            0, &next_batch_index, train_fp, train_lbl.get(),
            W1.get(), W2.get(), workers.get(), &total_correct,
            inp_pad, hid_pad, out_pad, BATCH_SIZE, lr
        );
        for (auto& t : threads) {
            t.join();
        }
        uint64_t end_time = std::chrono::steady_clock::now().time_since_epoch().count();
        float samples_per_sec = (float)(num_batches * BATCH_SIZE) / ((end_time - start_time) / 1e9);
        printf(
            "  Epoch %d done: acc=%.2f%%, throughput=%.2f samples/sec, time=%.2f sec\n",
            ep + 1,
            100.0f * total_correct.load(std::memory_order_acquire) / (num_batches * BATCH_SIZE),
            samples_per_sec,
            (end_time - start_time) / 1e9
        );
        // ── Eval (single-threaded) ──
        int tc = 0;
        for (int s = 0; s < TEST_IMAGES; s++) {
            const float* xi = &test_fp[s * inp_pad];
            memset(eval_ctx.hidden, 0, hid_pad * sizeof(float));
            ane::dispatch(
                ane::Op::dense_fused_i8, 1, HIDDEN_DIM, INPUT_DIM,
                1.0f, true, xi, W1.get(), eval_ctx.hidden
            );
            memset(eval_ctx.logits, 0, out_pad * sizeof(float));
            ane::dispatch(
                ane::Op::dense_fused_i8, 1, out_pad, HIDDEN_DIM,
                1.0f, false, eval_ctx.hidden, W2.get(), eval_ctx.logits
            );
            for (int i = OUTPUT_DIM; i < out_pad; i++) eval_ctx.logits[i] = -1000.0f;  // mask padding
            int32_t pred = 0;
            float g_out_dummy[16] = {};
            ane::dispatch(
                ane::Op::softmax_argmax_fp32, 1, out_pad,
                eval_ctx.logits, eval_ctx.probs, &test_lbl[s], g_out_dummy, &pred
            );
            if (pred == (int32_t)test_lbl[s]) {
                tc++;
            }
        }
        printf("  Test: %.2f%% (%d/%d)\n\n", 100.0f * tc / TEST_IMAGES, tc, TEST_IMAGES);
    }
    printf("Training complete.\n");
    for (int t = 0; t < num_threads; t++) {
        workers[t].destroy();
    }
    free(train_fp); free(test_fp);
    return 0;
}
