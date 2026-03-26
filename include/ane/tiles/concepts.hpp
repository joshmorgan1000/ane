#pragma once
/** --------------------------------------------------------------------------------------------------------- Concepts
 * @file concepts.hpp
 * @brief Defines C++20 concepts to constrain template parameters for the ane library.
 * 
 * @author Josh Morgan (@joshmorgan1000 on GitHub)
 * Released under the MIT License
 */
#include <concepts>
#include <type_traits>
#include <simd/simd.h>

namespace ane {
/** --------------------------------------------------------------------------------------------------------- ValidZType Concept
 * @brief Concept to constrain valid types for z_stream operations. This includes all the standard
 * integer and floating-point types that the Apple Neural Engine supports for its vectorized
 */
template<typename T>
concept ValidZType = std::same_as<T, float>
    || std::same_as<T, int32_t>
    || std::same_as<T, uint32_t>
    || std::same_as<T, int>
    || std::same_as<T, unsigned int>
    || std::same_as<T, int16_t>
    || std::same_as<T, uint16_t>
    || std::same_as<T, int8_t>
    || std::same_as<T, uint8_t>
    || std::same_as<T, char>
    || std::same_as<T, unsigned char>
    || std::same_as<T, bfloat16_t>
    || std::same_as<T, double>
    || std::same_as<T, int64_t>
    || std::same_as<T, uint64_t>
    || std::same_as<T, long>
    || std::same_as<T, unsigned long>
    || std::same_as<T, long long>
    || std::same_as<T, unsigned long long>;
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
/** --------------------------------------------------------------------------------------------------------- LUTCompatible Concept
 * @brief Concept to constrain types that are compatible with the LUTI2 or LUTI4 instructions.
 */
template<typename T>
concept LUTCompatible = std::same_as<T, uint8_t>
    || std::same_as<T, int8_t>
    || std::same_as<T, char>
    || std::same_as<T, unsigned char>
    || std::same_as<T, uint16_t>
    || std::same_as<T, int16_t>
    || std::same_as<T, bfloat16_t>
    || std::same_as<T, uint32_t>
    || std::same_as<T, int32_t>
    || std::same_as<T, int>
    || std::same_as<T, unsigned int>
    || std::same_as<T, float>;
} // namespace ane