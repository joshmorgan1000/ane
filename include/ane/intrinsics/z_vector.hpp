#pragma once
/** --------------------------------------------------------------------------------------------------------- z-stream
 * @file z_vector.hpp
 * @brief Header file for the z_vector class, a convienience wrapper to dynamically
 * build and define `za` tile vector operations without mucking around in
 * the assembly yourself.
 * 
 * We actually just use the 64-byte aligned simd::vector types for this since the performance indicates that
 * they use the z[n] registers under the hood, and it saves us from having to write a lot of boilerplate to
 * manage aligned memory and streaming manually.
 */
#if defined(__aarch64__) && defined(__APPLE__)
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <ane/extern/asm.hpp>
// We use simd.h because their 64-byte aligned vector types line up with the z[n] register sizes exactly
// (they likely even use the z[n] registers under the hood, but we'd like to expose more)
#include <simd/simd.h>
#include "common.hpp"

namespace ane {
/** ------------------------------------------------------------------------------------------ z_vector 
 * @class z_vector
 * @brief A helper class to manage a z[n] vector for use in assembly kernels. This class
 * abstracts away the details of allocating and managing the aligned memory needed for a z[n]
 * vector, and provides a convenient interface for working with vectors 
 */
class alignas(64) z_vector {
private:
    /** --------------------------------------------------------------------------------- Aligned Memory Allocation
     * @brief Type-erased storage for the z_vector data. We use a std::variant of the
     * simd vector types that correspond to the valid types for z_vector. This allows us
     * to store the data in a type-safe way while still abstracting away the details of
     * the underlying SIMD types.
     */
    std::variant<
        simd_float16,
        simd_uchar64,
        simd_char64,
        simd_ushort32,
        simd_short32,
        simd_half32,
        simd_int16,
        simd_uint16,
        simd_long8,
        simd_ulong8,
        simd_double8
    > data_;
public:
    /** ----------------------------------------------------------------------------------------- Constructor
     * @brief Constructs a z_vector with the specified initial value. The type of the z_vector is
     * determined by the template parameter T, which must be one of the valid types defined in
     * the ValidZType concept. The constructor initializes the underlying SIMD vector with the
     * provided initial value.
     * @tparam T The type of the elements in the z_vector (must satisfy ValidZType)
     * @param initial_value The initial value to set for all elements in the z_vector
     * (default is 0)
     */
    template<ValidZType T = uint8_t>
    z_vector(T initial_value = T()) {
        static_assert(SIMDVectorEquivalent<decltype(data_)>, "Invalid type for z_vector");
        if constexpr (std::same_as<T, float>) {
            data_ = simd_float16(initial_value);
        } else if constexpr (std::same_as<T, uint8_t>) {
            data_ = simd_uchar64(initial_value);
        } else if constexpr (std::same_as<T, int8_t>) {
            data_ = simd_char64(initial_value);
        } else if constexpr (std::same_as<T, uint16_t>) {
            data_ = simd_ushort32(initial_value);
        } else if constexpr (std::same_as<T, int16_t>) {
            data_ = simd_short32(initial_value);
        } else if constexpr (std::same_as<T, bfloat16_t>) {
            data_ = simd_half32(initial_value);
        } else if constexpr (std::same_as<T, int32_t>) {
            data_ = simd_int16(initial_value);
        } else if constexpr (std::same_as<T, uint32_t>) {
            data_ = simd_uint16(initial_value);
        } else if constexpr (std::same_as<T, int64_t>) {
            data_ = simd_long8(initial_value);
        } else if constexpr (std::same_as<T, uint64_t>) {
            data_ = simd_ulong8(initial_value);
        } else if constexpr (std::same_as<T, double>) {
            data_ = simd_double8(initial_value);
        } else {
            static_assert(always_false<T>, "Unsupported type for z_vector");
        }
    }
    auto& operator*() {
        return std::get<std::decay_t<decltype(data_)>>(data_);
    }
    const auto& operator*() const {
        return std::get<std::decay_t<decltype(data_)>>(data_);
    }
    auto* operator->() {
        return &std::get<std::decay_t<decltype(data_)>>(data_);
    }
    const auto* operator->() const {
        return &std::get<std::decay_t<decltype(data_)>>(data_);
    }
    template<SIMDVectorEquivalent T>
    T& data() {
        return std::get<T>(data_);
    }
    template<SIMDVectorEquivalent T>
    const T& data() const {
        return std::get<T>(data_);
    }
    template<ValidZType T>
    T& get(size_t index) {
        return std::get<std::decay_t<decltype(data_)>>(data_)[index];
    }
    template<ValidZType T>
    const T& get(size_t index) const {
        return std::get<std::decay_t<decltype(data_)>>(data_)[index];
    }
    template<ValidZType T>
    T& operator[](size_t index) {
        return std::get<std::decay_t<decltype(data_)>>(data_)[index];
    }
    template<ValidZType T>
    const T& operator[](size_t index) const {
        return std::get<std::decay_t<decltype(data_)>>(data_)[index];
    }
    template<ValidZType T>
    T* operator&() {
        return &std::get<std::decay_t<decltype(data_)>>(data_);
    }
    template<ValidZType T>
    const T* operator&() const {
        return &std::get<std::decay_t<decltype(data_)>>(data_);
    }
    template<SIMDVectorEquivalent T>
    operator T&() {
        return std::get<T>(reinterpret_cast<std::variant<T>&>(data_));
    }
    template<SIMDVectorEquivalent T>
    operator const T&() const {
        return std::get<T>(reinterpret_cast<const std::variant<T>&>(data_));
    }
    template<ValidZType T>
    z_vector& operator=(T value) {
        *this = z_vector(value);
        return *this;
    }
    template<ValidZType T>
    z_vector& operator=(const T* ptr) {
        *this = z_vector(ptr);
        return *this;
    }
    template<ValidZType U>
    z_vector& operator+=(const U& scalar) {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType scalar_vec(scalar);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        data_ += scalar_vec;
        return *this;
    }
    template<SIMDVectorEquivalent U>
    z_vector& operator+=(const z_vector& other) {
        using CommonSIMDType = decltype(find_largest_simd_type_for(T{}));
        const CommonSIMDType& other_data = std::get<CommonSIMDType>(other.data_);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        data_ += other_data;
        return *this;
    }
    z_vector operator+(const z_vector& b) const noexcept {
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            const CommonSIMDType& b_data = std::get<CommonSIMDType>(b.data_);
            result_data = a_data + b_data;
        }, data_);
        z_vector result;
        result.data_ = result_data;
        return result;
    }
    template<ValidZType U>
    z_vector& operator-=(const U& scalar) {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType scalar_vec(scalar);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        data_ -= scalar_vec;
        return *this;
    }
    z_vector operator-(const z_vector& b) const noexcept {
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            const CommonSIMDType& b_data = std::get<CommonSIMDType>(b.data_);
            result_data = a_data - b_data;
        }, data_);
        z_vector result;
        result.data_ = result_data;
        return result;
    }
    template<ValidZType U>
    z_vector& operator*=(const U& scalar) {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType scalar_vec(scalar);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        data_ *= scalar_vec;
        return *this;
    }
    z_vector operator*(const z_vector& b) const noexcept {
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            const CommonSIMDType& b_data = std::get<CommonSIMDType>(b.data_);
            result_data = a_data * b_data;
        }, data_);
        z_vector result;
        result.data_ = result_data;
        return result;
    }
    template<ValidZType U>
    z_vector& operator/=(const U& scalar) {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType scalar_vec(scalar);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        data_ /= scalar_vec;
        return *this;
    }
    z_vector operator/(const z_vector& b) const noexcept {  
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            const CommonSIMDType& b_data = std::get<CommonSIMDType>(b.data_);
            result_data = a_data / b_data;
        }, data_);
        z_vector result;
        result.data_ = result_data;
        return result;
    }
    z_vector& operator>>=(int shift) {
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            result_data = a_data >> shift;
        }, data_);
        data_ = result_data;
        return *this;
    }
    z_vector operator>>(int shift) const noexcept {
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            result_data = a_data >> shift;
        }, data_);
        z_vector result;
        result.data_ = result_data;
        return result;
    }
    z_vector& operator<<=(int shift) {
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            result_data = a_data << shift;
        }, data_);
        data_ = result_data;
        return *this;
    }
    z_vector operator<<(int shift) const noexcept {
        decltype(data_) result_data;
        std::visit([&](auto&& a_vec) {
            using CommonSIMDType = std::decay_t<decltype(a_vec)>;
            const CommonSIMDType& a_data = a_vec;
            result_data = a_data << shift;
        }, data_);
        z_vector result;
        result.data_ = result_data;
        return result;
    }
};
} // namespace ane
namespace std {
template<Valid
ane::z_vector max(const ane::z_vector& a, const ane::z_vector& b) {
    return simd_max(static_cast<a, b);
}
ane::z_vector min(const ane::z_vector& a, const ane::z_vector& b) {
    return std::min(a, b);
}
ane::z_vector operator+(const ane::z_vector& a, const ane::z_vector& b) {

}
ane::z_vector operator-(const ane::z_vector& a, const ane::z_vector& b) {
    return a - b;
}
} // namespace std
#endif // __aarch64__ && __APPLE__