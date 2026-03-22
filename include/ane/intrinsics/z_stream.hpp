#pragma once
/** --------------------------------------------------------------------------------------------------------- z-stream
 * @file z_stream.hpp
 * @brief Header file for the z_stream struct, a convienience wrapper to dynamically
 * build and define `za` tile streaming operations without mucking around in
 * the assembly yourself.
 */
#if defined(__aarch64__) && defined(__APPLE__)
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include "../extern/asm.hpp"
#include "common.hpp"
#include "lut.hpp"
#include <simd/simd.h>
#include <vector>

namespace ane {
/** ------------------------------------------------------------------------------------------ z_stream 
 * @class z_stream
 * @brief A helper class to manage a stream of z[n] registers for use in assembly kernels.
 * This class abstracts away the details of allocating and managing the aligned memory needed 
 * for streaming data into the z[n] registers, and provides a convenient interface for working
 * with streams of vectors of various types and various operations.
 */
template<SIMDVectorEquivalent T>
class z_stream {
private:
    /// @brief Type-erased pointer to allocate the aligned memory for the z_stream
    T* data_ = nullptr;
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
        data_ = static_cast<simd_uchar64*>(std::aligned_alloc(4096, size));
        if (!data_) [[unlikely]] {
            std::string error_msg = "Failed to allocate memory for z_stream:"
                    " requested size " + std::to_string(size) + " bytes";
            throw std::runtime_error(error_msg);
        }
    }
public:
    z_stream(size_t num_zvecs) : num_zvecs_(num_zvecs) {
        alignedAlloc(num_zvecs_);
    }
    ~z_stream() {
        if (data_) {
            std::free(data_);
        }
    }
    z_stream(const z_stream&) = delete;
    z_stream& operator=(const z_stream&) = delete;
    z_stream(z_stream&& other) noexcept : data_(other.data_), num_zvecs_(other.num_zvecs_) {
        other.data_ = nullptr;
        other.num_zvecs_ = 0;
    }
    z_stream& operator=(z_stream&& other) noexcept {
        if (this != &other) {
            if (data_) {
                std::free(data_);
            }
            data_ = other.data_;
            num_zvecs_ = other.num_zvecs_;
            other.data_ = nullptr;
            other.num_zvecs_ = 0;
        }
        return *this;
    }
};
} // namespace ane
#endif // __aarch64__ && __APPLE__