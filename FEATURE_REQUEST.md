# Feature Request: ByteCode kernels needed

## Kernels Requested

### 1. L2 Squared Distance

`sum((a[i] - b[i])^2)` for two arrays of length `dim`. Returns one value. Need this for bfloat16, float, and double. It would be cool if you could specify the output precision, so this might actually be best if it was split into multiple instructions - subtraction, then square, then accumulate. Do we have a designated accumulator tile we can "reserve" that will persist across bytecode instructions? That would be nice. Although we realize that this may be something that there is a specialized instruction for? Use your best judgement.

### 2. Cosine Distance (float32)

`1 - dot(a,b) / (||a|| * ||b||)` also for bfloat16, float, and double. Same deal as the L2.

### 3. Quantized SoA Accumulators

We have 8-bit, 4-bit, 3-bit, 2-bit, and even single-bit quantized vectors, each dimension stored SoA, so the above two methods would be a stream of dimension 0 vs. element 1, then a stream of dimension 1 vs. element 2, etc... so essentially (for L2 at least) a vector (or matrix tile!) minus a scalar (which might just be a bit or 2), multiplied by a scale and then streamed out to a bfloat16 accumulator stream. This will need to handle multiple inputs. We're actually thinking a good approach to try also is using the LUTI2 - if we can load up a za.b with a handful of different pointer loads, and then do a full za tile LUTI2 or LUTI4, we think we can potentially widen the single-bit output to a pair of 16-bit outputs (even if encoded as a single 32-bit output) making the operation one LUTI2 and a few adds into an accumulator before the store command.

### 5. Threshold bitmap

If we provide a specific threshold (is number greater than T?) return a bitmap or vector of boolean values indexed by which elements exceed the threshold.

### 6. Normalize (float32)

In-place normalize a float32 array to unit length: `vec[i] /= ||vec||`. Length `dim`. Used during vector ingestion and query prep.

### 7. DCT-II Forward (float32)

H.264 integer butterfly DCT on groups of 4: convert to fixed-point int32, butterfly (s0+s1, d0+(d1>>1), s0-s1, (d0>>1)-d1), convert back. Length `dim` (always multiple of 4). Used during ingestion to compute DCT coefficients.

### 8. DCT-II Inverse (float32)

Inverse of the above. Same fixed-point butterfly, inverse pattern. Length `dim`.

### 9. Compute Stats (Welford)

Online mean/variance/maxabs computation across `n_vectors` of dimension `dim`. Uses Welford's algorithm: for each element, update running mean, M2 accumulator (for variance), and track max absolute value. Outputs per-dimension: mean (double), stddev (double), maxabs (double), scale (double). Used during ingestion to compute quantization parameters.

Input type is float32 but all accumulation must be in double precision (Welford is numerically sensitive).

### 10. Quantize and Pack 4-Bit (float32)

Given float32 source vectors (AoS, `n * dim`), per-dimension DimStats (mean, scale), quantize each value to a signed 4-bit integer [-7, +7] via `round((val - mean) * scale)`, clamp, then pack two nibbles per byte into SoA layout (`dim` columns of `n/2` bytes each). Processes both raw and DCT source arrays simultaneously (dual output).

### 11. 2-Bit Quantize + Accumulate

Same pattern as the 4-bit accumulate but for 2-bit packed values (4 values per byte). Ternary quantization {-1, 0, +1}. With per-coefficient bf16 weights.

### 12. 8-Bit Accumulate

Same pattern but for 8-bit values (1 value per byte). Full signed int8 range [-127, +127]. With per-coefficient bf16 weights.

### 13. 8-Bit Threshold

Same as the 6-bit threshold but for 8-bit counters across 8 bitplanes (b0-b7). Threshold range 0-255.

### 14. Bitmap Score Full Pipeline

The whole thing as one operation: given `n_streams` bitmap arrays, `is_high` flags per stream, `n_bytes` per bitmap, `n_vectors`, `score_min` threshold, and `max_candidates` limit — accumulate all streams via ripple-carry, threshold, extract candidate IDs. Returns candidate count + ID array. This is `bitmap_score_1bit` as a single kernel so the entire accumulate-threshold-extract pipeline stays in one streaming session.
