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
    T* data_ = nullptr;
    /// @brief Size of the stream in bytes
    size_t size_bytes_ = 0;
    /** --------------------------------------------------------------------------------- Aligned Memory Allocation
     * @brief Allocates aligned memory for the z_stream, rounded up to the nearest
     * multiple of 64 bytes for z-vector alignment.
     * @param size_bytes The total size in bytes needed for the z_stream
     */
    void alignedAlloc(size_t size_bytes) {
        size_t aligned_size = (size_bytes + 63) & ~size_t(63);
        data_ = static_cast<T*>(std::aligned_alloc(4096, aligned_size));
        if (!data_) [[unlikely]] {
            std::string error_msg = "Failed to allocate memory for z_stream:"
                    " requested size " + std::to_string(aligned_size) + " bytes";
            throw std::runtime_error(error_msg);
        }
    }
public:
    /** --------------------------------------------------------------------------------- Constructor
     * @brief Constructor for z_stream. Allocates memory for the specified size in bytes.
     * @param size_bytes The total size in bytes to allocate for the stream
     */
    z_stream(size_t size_bytes, T initial_val = T()) : size_bytes_(size_bytes) {
        alignedAlloc(size_bytes_);
        std::fill(data_, data_ + (size_bytes_ / sizeof(T)), initial_val);
    }
    /** --------------------------------------------------------------------------------- Constructor from Existing Pointer
     * @brief Constructor from an existing pointer. Does not take ownership — the caller
     * is responsible for the lifetime of the pointed-to memory.
     * @param ptr A pointer to memory that is at least 64-byte aligned
     * @param size_bytes The size of the buffer in bytes
     */
    z_stream(T* ptr, size_t size_bytes) : data_(ptr), size_bytes_(size_bytes) {
        if ((reinterpret_cast<uintptr_t>(ptr) & 0x3F) != 0) {
            std::string error_msg = "Pointer provided to z_stream constructor is"
                    " not 64-byte aligned: "
                    + std::to_string(reinterpret_cast<uintptr_t>(ptr));
            throw std::invalid_argument(error_msg);
        }
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
        z_stream copy(size_bytes_);
        std::memcpy(copy.data_, data_, size_bytes_);
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
        if ((reinterpret_cast<uintptr_t>(ptr) & 0x3F) != 0) {
            throw std::bad_alloc();
        }
    }
};
}