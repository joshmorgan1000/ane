#pragma once
/** --------------------------------------------------------------------------------------------------------- Apple Neural Engine
 * @file common.hpp
 * @brief Common utilities and concepts for the Apple Neural Engine intrinsics.
 */
#if defined(__aarch64__) && defined(__APPLE__)
#include <arm_neon.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include "../extern/asm.hpp"
#include <simd/simd.h>

namespace ane {
/** --------------------------------------------------------------------------------------------------------- always_false
 * @brief Template helper for static_assert in constexpr if branches
 */
template<typename T>
inline constexpr bool always_false = false;
/** --------------------------------------------------------------------------------------------------------- ValidZType Concept
 * @brief Concept to constrain valid types for z_stream operations. This includes all the standard
 * integer and floating-point types that the Apple Neural Engine supports for its vectorized
 */
template<typename T>
concept ValidZType = std::same_as<T, float>
    || std::same_as<T, int32_t>
    || std::same_as<T, uint32_t>
    || std::same_as<T, int16_t>
    || std::same_as<T, uint16_t>
    || std::same_as<T, int8_t>
    || std::same_as<T, uint8_t>
    || std::same_as<T, bfloat16_t>
    || std::same_as<T, double>
    || std::same_as<T, int64_t>
    || std::same_as<T, uint64_t>;
/** --------------------------------------------------------------------------------------------------------- SIMDVectorEquivalent Concept
 * @brief Concept to constrain types that have a corresponding SIMD vector type in simd.h.
 * Makes sure memory is aligned and adds a lot of stuff so we don't have to
 */
template<typename T>
concept SIMDVectorEquivalent = std::same_as<T, simd_float16>
    || std::same_as<T, simd_uchar64>
    || std::same_as<T, simd_char64>
    || std::same_as<T, simd_ushort32>
    || std::same_as<T, simd_short32>
    || std::same_as<T, simd_half32>
    || std::same_as<T, simd_int16>
    || std::same_as<T, simd_uint16>
    || std::same_as<T, simd_long8>
    || std::same_as<T, simd_ulong8>
    || std::same_as<T, simd_double8>;
/** --------------------------------------------------------------------------------------------------------- Find Convertable Type
 * @brief Helper function to find a common type for mixed-type z_vector operations.
 *  - If T and U are the same, returns that type
 *  - If T is convertible to U, returns U
 *  - If U is convertible to T, returns T
 *  - If one is pc8 and the other is float, returns float (promote pc8 to float for arithmetic)
 *  - Otherwise, triggers a static assertion failure
 * @tparam T First type
 * @tparam U Second type
 * @return A type that both T and U can be converted to for mixed operations
 */
template<ValidZType T, ValidZType U>
consteval U find_convertable_type() {
    if constexpr (std::same_as<T, U>) {
        return U{};
    } else if constexpr (std::is_convertible_v<T, U>) {
        return U{};
    } else if constexpr (std::is_convertible_v<U, T>) {
        return T{};
    } else {
        static_assert(always_false<T>, "No valid conversion between types");
    }
}
/** --------------------------------------------------------------------------------------------------------- Find Largest SIMD Type for Scalar Type
 * @brief Helper function to find the largest SIMD type that can be used for a given scalar type.
 *  - For float, returns simd_float16 (256-bit / 16 bits per element = 16 elements)
 *  - For int32_t and uint32_t, returns simd_int16 and simd_uint16
 *         (256-bit / 16 bits per element = 16 elements)
 *  - For int16_t and uint16_t, returns simd_short32 and simd_ushort32
 *         (256-bit / 8 bits per element = 32 elements)
 *  - For int8_t and uint8_t, returns simd_char64 and simd_uchar64
 *         (256-bit / 4 bits per element = 64 elements)
 *  - For bfloat16_t, returns bfloat16x8_t (256-bit / 16 bits per element = 8 elements)
 *  - For psyne::pc8, returns simd_uchar64 (256-bit / 4 bits per element = 64 elements)
 * @tparam T The SIMD type to return
 * @tparam U The scalar type to find a SIMD type for
 * @return The largest SIMD type that can be used for the given scalar type
 */
template<SIMDVectorEquivalent T, ValidZType U>
consteval T find_largest_simd_type_for(U value) {
    if constexpr (std::same_as<U, float>) {
        return simd_float16{};
    } else
    if constexpr (std::same_as<U, int32_t>) {
        return simd_int16{};
    } else
    if constexpr (std::same_as<U, uint32_t>) {
        return simd_uint16{};
    } else
    if constexpr (std::same_as<U, int16_t>) {
        return simd_short32{};
    } else
    if constexpr (std::same_as<U, uint16_t>) {
        return simd_ushort32{};
    } else
    if constexpr (std::same_as<U, int8_t>) {
        return simd_char64{};
    } else
    if constexpr (std::same_as<U, uint8_t>) {
        return simd_uchar64{};
    } else
    if constexpr (std::same_as<U, bfloat16_t>) {
        return bfloat16x8_t{};
    } else {
        static_assert(always_false<U>, "No valid SIMD type for the given scalar type");
    }
}
} // namespace ane
#endif