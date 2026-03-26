#pragma once
/** --------------------------------------------------------------------------------------------------------- z_vector
 * @file z_vector.hpp
 * @brief Defines the z_vector struct, which represents a single 64-byte aligned z-vector register
 */
#include <ane/tiles/concepts.hpp>

namespace ane {
template<typename> constexpr bool always_false = false;

/** --------------------------------------------------------------------------------------------------------- z_vector
 * @struct z_vector
 * @brief A vector struct modeling the SVE vector register (SVL=512-bit) for various element types. Provides
 * vectorized operations using assembly kernels.
 * @tparam T The element type stored in the vector (must satisfy ValidZType)
 */
template<ValidZType T>
struct alignas(64) z_vector {
private:
    static constexpr size_t num_elements = 64 / sizeof(T);
    T data_[num_elements] = {0}; // 4 * 64 bytes = 64 bytes total (512 bits)
public:
    z_vector() = default;
    z_vector(T* ptr) {
        *reinterpret_cast<simd_char64*>(data_) = *reinterpret_cast<const simd_char64*>(ptr);
    }
    z_vector(T value) {
        if constexpr (std::same_as<T, float>) {
            *reinterpret_cast<simd_float16*>(data_) = simd_float16(value);
        } else
        if constexpr (std::same_as<T, int32_t>) {
            *reinterpret_cast<simd_int16*>(data_) = simd_int16(value);
        } else
        if constexpr (std::same_as<T, uint32_t>) {
            *reinterpret_cast<simd_uint16*>(data_) = simd_uint16(value);
        } else
        if constexpr (std::same_as<T, int16_t> || std::same_as<T, int>) {
            *reinterpret_cast<simd_short32*>(data_) = simd_short32(value);
        } else
        if constexpr (std::same_as<T, uint16_t> || std::same_as<T, unsigned int>) {
            *reinterpret_cast<simd_ushort32*>(data_) = simd_ushort32(value);
        } else
        if constexpr (std::same_as<T, int8_t> || std::same_as<T, char>) {
            *reinterpret_cast<simd_char64*>(data_) = simd_char64(value);
        } else
        if constexpr (std::same_as<T, uint8_t> || std::same_as<T, unsigned char>) {
            *reinterpret_cast<simd_uchar64*>(data_) = simd_uchar64(value);
        } else
        if constexpr (std::same_as<T, bfloat16_t>) {
            *reinterpret_cast<simd_half32*>(data_) = simd_half32(value);
        } else
        if constexpr (std::same_as<T, double>) {
            *reinterpret_cast<simd_double8*>(data_) = simd_double8(value);
        } else
        if constexpr (std::same_as<T, int64_t> || std::same_as<T, long long>) {
            *reinterpret_cast<simd_long8*>(data_) = simd_long8(value);
        } else
        if constexpr (std::same_as<T, uint64_t> || std::same_as<T, unsigned long long>) {
            *reinterpret_cast<simd_ulong8*>(data_) = simd_ulong8(value);
        } else static_assert(always_false<T>, "Unsupported type for z_vector");
    }
    z_vector& operator=(T value) {
        *this = z_vector(value);
        return *this;
    }
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
        for (size_t i = 0; i < 4; i++) {
            data[i] += scalar_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector operator+(const z_vector<U>& b) const noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        constexpr size_t num_simd_elements = sizeof(data_) / sizeof(CommonSIMDType);
        CommonSIMDType result[num_simd_elements];
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            result[i] = a_vec + b_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator-=(const U& scalar) {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType scalar_vec(scalar);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        for (size_t i = 0; i < 4; i++) {
            data[i] -= scalar_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator-=(const z_vector<U>& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        constexpr size_t num_simd_elements = sizeof(data_) / sizeof(CommonSIMDType);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        const CommonSIMDType* b_data = reinterpret_cast<const CommonSIMDType*>(&b.data_[0]);
        for (size_t i = 0; i < num_simd_elements; i++) {
            data[i] -= b_data[i];
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator*=(const U& scalar) {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType scalar_vec(scalar);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        for (size_t i = 0; i < 4; i++) {
            data[i] *= scalar_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator*=(const z_vector<U>& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        constexpr size_t num_simd_elements = sizeof(data_) / sizeof(CommonSIMDType);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        const CommonSIMDType* b_data = reinterpret_cast<const CommonSIMDType*>(&b.data_[0]);
        for (size_t i = 0; i < num_simd_elements; i++) {
            data[i] *= b_data[i];
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator/=(const U& scalar) {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType scalar_vec(scalar);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        for (size_t i = 0; i < 4; i++) {
            data[i] /= scalar_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator/=(const z_vector<U>& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        constexpr size_t num_simd_elements = sizeof(data_) / sizeof(CommonSIMDType);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        const CommonSIMDType* b_data = reinterpret_cast<const CommonSIMDType*>(&b.data_[0]);
        for (size_t i = 0; i < num_simd_elements; i++) {
            data[i] /= b_data[i];
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator&=(const U& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType b_vec(b);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        for (size_t i = 0; i < 4; i++) {
            data[i] &= b_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator&=(const z_vector<U>& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        constexpr size_t num_simd_elements = sizeof(data_) / sizeof(CommonSIMDType);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        const CommonSIMDType* b_data = reinterpret_cast<const CommonSIMDType*>(&b.data_[0]);
        for (size_t i = 0; i < num_simd_elements; i++) {
            data[i] &= b_data[i];
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator|=(const U& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType b_vec(b);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        for (size_t i = 0; i < 4; i++) {
            data[i] |= b_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator|=(const z_vector<U>& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        constexpr size_t num_simd_elements = sizeof(data_) / sizeof(CommonSIMDType);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        const CommonSIMDType* b_data = reinterpret_cast<const CommonSIMDType*>(&b.data_[0]);
        for (size_t i = 0; i < num_simd_elements; i++) {
            data[i] |= b_data[i];
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator^=(const U& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        const CommonSIMDType b_vec(b);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        for (size_t i = 0; i < 4; i++) {
            data[i] ^= b_vec;
        }
        return *this;
    }
    template<ValidZType U>
    z_vector& operator^=(const z_vector<U>& b) noexcept {
        using CommonType = decltype(find_convertable_type<T, U>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        constexpr size_t num_simd_elements = sizeof(data_) / sizeof(CommonSIMDType);
        CommonSIMDType* data = reinterpret_cast<CommonSIMDType*>(data_);
        const CommonSIMDType* b_data = reinterpret_cast<const CommonSIMDType*>(&b.data_[0]);
        for (size_t i = 0; i < num_simd_elements; i++) {
            data[i] ^= b_data[i];
        }
        return *this;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator+(const z_vector<A>& a, const z_vector<B>& b) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec + b_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator+(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec + scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator-(const z_vector<A>& a, const z_vector<B>& b) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec - b_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator-(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec - scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator*(const z_vector<A>& a, const z_vector<B>& b) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec * b_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator*(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec * scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator/(const z_vector<A>& a, const z_vector<B>& b) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec / b_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator/(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec / scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator&(const z_vector<A>& a, const z_vector<B>& b) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec & b_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator&(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec & scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator|(const z_vector<A>& a, const z_vector<B>& b) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec | b_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator|(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec | scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator^(const z_vector<A>& a, const z_vector<B>& b) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec ^ b_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<decltype(find_convertable_type<A, B>())> operator^(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonType = decltype(find_convertable_type<A, B>());
        using CommonSIMDType = decltype(find_largest_simd_type_for(CommonType{}));
        z_vector<CommonType> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec ^ scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<A> operator>>(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonSIMDType = decltype(find_largest_simd_type_for(A{}));
        z_vector<A> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec >> scalar_vec;
        }
        return result;
    }
    template<ValidZType A, ValidZType B>
    friend z_vector<A> operator<<(const z_vector<A>& a, const B& scalar) noexcept {
        using CommonSIMDType = decltype(find_largest_simd_type_for(A{}));
        z_vector<A> result;
        const CommonSIMDType scalar_vec(scalar);
        constexpr size_t num_simd_elements = sizeof(a.data_) / sizeof(CommonSIMDType);
        for (size_t i = 0; i < num_simd_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = a_vec << scalar_vec;
        }
        return result;
    }
    template<SIMDVectorEquivalent U>
    operator std::array<U, 64 / sizeof(U)>&() {
        return reinterpret_cast<std::array<U, 64 / sizeof(U)>&>(data_);
    }
    template<SIMDVectorEquivalent U>
    operator const std::array<U, 64 / sizeof(U)>&() const {
        return reinterpret_cast<const std::array<U, 64 / sizeof(U)>&>(data_);
    }
    template<ValidZType U>
    operator std::array<U, 64 / sizeof(U)>&() {
        return reinterpret_cast<std::array<U, 64 / sizeof(U)>&>(data_);
    }
    template<ValidZType U>
    operator const std::array<U, 64 / sizeof(U)>&() const {
        return reinterpret_cast<const std::array<U, 64 / sizeof(U)>&>(data_);
    }
    template<ValidZType U = T>
    U& operator[](size_t idx) {
        if constexpr (std::same_as<U, T>) {
            return data_[idx];
        } else {
            return reinterpret_cast<std::array<U, 64 / sizeof(U)>&>(data_)[idx];
        }
    }
    template<ValidZType U = T>
    const U& operator[](size_t idx) const {
        if constexpr (std::same_as<U, T>) {
            return data_[idx];
        } else {
            return reinterpret_cast<const std::array<U, 64 / sizeof(U)>&>(data_)[idx];
        }
    }
    template<ValidZType U>
    friend z_vector<U> max(const z_vector<U>& a, const z_vector<U>& b) noexcept {
        using CommonSIMDType = decltype(find_largest_simd_type_for(U{}));
        z_vector<U> result;
        for (size_t i = 0; i < num_elements; i++) {
            CommonSIMDType a_vec = reinterpret_cast<const CommonSIMDType*>(&a.data_[0])[i];
            CommonSIMDType b_vec = reinterpret_cast<const CommonSIMDType*>(&b.data_[0])[i];
            reinterpret_cast<CommonSIMDType*>(&result.data_[0])[i] = simd_max(a_vec, b_vec);
        }
        return result;
    }
};
}