#pragma once
/** --------------------------------------------------------------------------------------------------------- z_stream
 * @fiule z_stream.hpp
 * @brief Defines the z_stream class, which provides a high-level interface for building and executing
 * sequences of operations on z-vectors and ZA tiles using the SME matrix unit.
 */
#include <ane/tiles/concepts.hpp>
#include <ane/tiles/z_vector.hpp>

namespace ane {
/** --------------------------------------------------------------------------------------------------------- z_stream 
 * @class z_stream
 * @brief A helper class to manage a stream of z[n] registers for use in assembly kernels.
 * This class abstracts away the details of allocating and managing the aligned memory needed 
 * for streaming data into the z[n] registers, and provides a convenient interface for working
 * with streams of vectors of various types and various operations.
 */
template<ValidZType T>
class z_stream {
private:
    /// @brief Type-erased pointer to allocate the aligned memory for the z_stream
    z_vector<T>* data_ = nullptr;
    /// @brief Number of zvecs (512-bit vectors) allocated in the stream
    size_t num_zvecs_ = 0;
    /** --------------------------------------------------------------------------------- Aligned Memory Allocation
     * @brief Allocates aligned memory for the z_stream based on the number of zvecs
     * needed. Each zvec is 64 bytes, so the total size allocated is num_zvecs * 64
     * bytes, rounded up to the nearest multiple of 64 for alignment.
     * @param size The total size in bytes needed for the z_stream (should be
     * num_zvecs * 64) 
     */
    void alignedAlloc(size_t size) {
        size <<= 6; // Multiply by 64 to get the total size in bytes
        // Align to the size of the full za.b tile (4096 bytes) to ensure optimal
        // access patterns for streaming
        data_ = static_cast<z_vector<T>*>(std::aligned_alloc(4096, size));
        if (!data_) [[unlikely]] {
            std::string error_msg = "Failed to allocate memory for z_stream:"
                    " requested size " + std::to_string(size) + " bytes";
            throw std::runtime_error(error_msg);
        }
    }
public:
    /** --------------------------------------------------------------------------------- Constructor
     * @brief Constructor for z_stream. Allocates memory for the specified number of
     * zvecs, each of which is 512 bits (64 bytes).
     * @param num_zvecs The number of 512-bit vectors (zvecs) to allocate in the stream
     */
    z_stream(size_t num_zvecs, T initial_val = T()) : num_zvecs_(num_zvecs) {
        alignedAlloc(num_zvecs_);
        std::fill(data_, data_ + num_zvecs_, z_vector<T>(initial_val));
    }
    /** --------------------------------------------------------------------------------- Destructor
     * @brief Destructor for z_stream. Frees the allocated memory for the stream.
     */
    ~z_stream() {
        if (data_) {
            std::free(data_);
        }
    }
    // Delete copy constructor and copy assignment operator to prevent accidental copying
    z_stream(const z_stream&) = delete;
    z_stream& operator=(const z_stream&) = delete;
    /** --------------------------------------------------------------------------------- Get Pointer
     * @brief Gets a pointer to the raw data of the z_stream. This pointer can be used
     * to load data into the zvecs for streaming into assembly kernels. The pointer is
     * typed as T* for convenience, but it actually points to the aligned memory
     * allocated for the z_stream, which is suitable for use with the z[n] registers.
     * @tparam T The SIMD vector type of the z_stream (e.g., simd_float16, simd_uchar64,
     * etc.).
     */
    template<SIMDVectorEquivalent U = T>
        requires std::same_as<U, T> || std::is_convertible_v<T, U>
    U* ptr() {
        return data_;
    }
    /** --------------------------------------------------------------------------------- Get Const Pointer
     * @brief Gets a const pointer to the raw data of the z_stream. This can be used
     * when the z_stream is only being read from (e.g., for streaming data into kernels
     * without modifying it).
     * @tparam T The SIMD vector type of the z_stream (e.g., simd_float16, simd_uchar64,
     * etc.).
     */
    template<SIMDVectorEquivalent U = T>
        requires std::same_as<U, T> || std::is_convertible_v<T, U>
    const U* ptr() const {
        return data_;
    }
    /** --------------------------------------------------------------------------------- Clone
     * @brief Creates a deep copy of the z_stream with the same number of zvecs and data.
     */
    z_stream clone() const {
        z_stream copy(num_zvecs_);
        std::memcpy(copy.data_, data_, num_zvecs_ << 6); // num_zvecs * 64 bytes
        return copy;
    }
};
/** --------------------------------------------------------------------------------------------------------- aligned_pointer
 * @struct aligned_pointer
 * @brief A simple wrapper for a pointer that ensures it is properly aligned for use with z_streams and
 * z_vectors. This can be used to safely pass pointers to assembly kernels that expect aligned memory.
 */
struct aligned_pointer {
    void* ptr;
    explicit aligned_pointer(void* p) : ptr(p) {
        if constexpr (std::alignment_of_v<std::max_align_t> > 64) {
            if (reinterpret_cast<uintptr_t>(ptr) % 64 != 0) {
                throw std::bad_alloc();
            }
        } else {
            if (reinterpret_cast<uintptr_t>(ptr) % std::alignment_of_v<std::max_align_t> != 0) {
                throw std::bad_alloc();
            }
        }
    }
};
}