#pragma once
/** --------------------------------------------------------------------------------------------------------- LUT Helpers
 * @file lut.hpp
 * @brief Helper structs for dealing with lookup tables in the SME matrix unit
 * 
 * @author Josh Morgan (@joshmorgan1000 on GitHub)
 * Released under the MIT License
 */
#include <ane/tiles/concepts.hpp>
#include <array>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <initializer_list>
#include <algorithm>


namespace ane {
/** --------------------------------------------------------------------------------------------------------- luti4
 * @struct luti4
 * @brief A helper utility struct for working with the LUTI4 instruction, which performs a 4-bit indexed
 * lookup into a 4-element or 16-element table.
 */
template<LUTCompatible T = uint32_t>
struct alignas(64) luti4 {
    /// @brief The underlying data array for the lookup table, aligned to 64 bytes.
    std::array<T, 16> data;
    /** --------------------------------------------------------------------------------- Default Constructor
     * @brief Default constructor initializes the lookup table with zeros
     */
    luti4(T init_val = 0) {
        data.fill(init_val);
    }
    /** --------------------------------------------------------------------------------- Construct from Array
     * @brief Constructor that initializes the lookup table with the provided array of
     * values. The input array must have exactly 16 elements, which will be copied into
     * the internal data array.
     * @param init_data The array of values to initialize the lookup table with.
     */
    luti4(const std::array<T, 16>& init_data) : data(init_data) {}
    /**
     * @brief Constructor that initializes the lookup table with the provided vector of
     * values. The input vector must have exactly 16 elements, which will be copied into
     * the internal data array.
     * @param init_data The vector of values to initialize the lookup table with
     * (must have 16 elements)
     */
    luti4(const std::vector<T>& init_data) {
        if (init_data.size() != 16) [[unlikely]] {
            throw std::invalid_argument("Initializer vector must have exactly"
                " 16 elements for luti4");
        }
        std::memcpy(data.data(), init_data.data(), 16 * sizeof(T));
    }
    /** --------------------------------------------------------------------------------- Construct from Initializer List
     * @brief Constructor that initializes the lookup table with the provided initializer
     * list of values. The input initializer list must have exactly 16 elements, which
     * will be copied into the internal data array.
     * @param init_list The initializer list of values to initialize the lookup table
     * with (must have 16 elements)
     */
    luti4(std::initializer_list<T> init_list) {
        if (init_list.size() != 16) [[unlikely]] {
            throw std::invalid_argument("Initializer list must have exactly"
                " 16 elements for luti4");
        }
        std::copy(init_list.begin(), init_list.end(), data.begin());
    }
    /** --------------------------------------------------------------------------------- Construct from Individual Values
     * @brief Constructor that initializes the lookup table with the provided individual
     * values. The constructor takes 16 individual values as parameters, which will be
     * copied into the internal data array.
     * @param val0 The first value to initialize the lookup table with
     * @param val1 The second value to initialize the lookup table with
     * @param val2 The third value to initialize the lookup table with
     * @param val3 The fourth value to initialize the lookup table with
     * @param val4 The fifth value to initialize the lookup table with
     * @param val5 The sixth value to initialize the lookup table with
     * @param val6 The seventh value to initialize the lookup table with
     * @param val7 The eighth value to initialize the lookup table with
     * @param val8 The ninth value to initialize the lookup table with
     * @param val9 The tenth value to initialize the lookup table with
     * @param val10 The eleventh value to initialize the lookup table with
     * @param val11 The twelfth value to initialize the lookup table with
     * @param val12 The thirteenth value to initialize the lookup table with
     * @param val13 The fourteenth value to initialize the lookup table with
     * @param val14 The fifteenth value to initialize the lookup table with
     * @param val15 The sixteenth value to initialize the lookup table with
     */
    luti4(
        T val0, T val1, T val2, T val3, T val4, T val5, T val6, T val7,
        T val8, T val9, T val10, T val11, T val12, T val13, T val14, T val15
    ) : data{val0, val1, val2, val3, val4, val5, val6, val7, val8, val9,
        val10, val11, val12, val13, val14, val15} {}
    /** --------------------------------------------------------------------------------- Get Value
     * @brief Gets a reference to the value at the specified index in the lookup table.
     * The index should be in the range [0, 15], and the function will return a reference
     * to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 15])
     * @return A reference to the value at the specified index in the lookup table
     */
    T& operator[](size_t index) {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti4");
        }
        return data[index];
    }
    /** --------------------------------------------------------------------------------- Get Value (const)
     * @brief Gets a const reference to the value at the specified index in the lookup
     * table. The index should be in the range [0, 15], and the function will return a
     * const reference to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 15])
     * @return A const reference to the value at the specified index in the lookup table
     */
    const T& operator[](size_t index) const {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti4");
        }
        return data[index];
    }
    /** --------------------------------------------------------------------------------- Test Function
     * @brief A test function that simulates the behavior of the LUTI4 instruction.
     * NOTE: In case it isn't obvious, this is NOT the same as the actual LUTI4
     * instruction, and will exeute far slower than the real deal.
     * 
     * Call it from a bytecode program with the same table and input data you would use
     * for the LUTI4 instruction, and it will return a vector of output values that you
     * can compare against the output from the real LUTI4 to verify correctness of your
     * program.
     * 
     * @param ptr Pointer to the input data (should point to at least size_bytes of data)
     * @param size_bytes The size of the input data in bytes
     * @return A vector of output values produced by the LUTI4 lookup
     */
    std::vector<T> test(const void* ptr, size_t size_bytes) const {
        std::vector<T> result;
        result.resize(size_bytes * 2); // Each byte produces 2 output values
        for (size_t i = 0; i < size_bytes; i++) {
            uint8_t byte = reinterpret_cast<const uint8_t*>(ptr)[i];
            uint8_t idx0 = byte & 0x0F; // Lower 4 bits
            uint8_t idx1 = (byte >> 4) & 0x0F; // Upper 4 bits
            result[2 * i] = data[idx0];
            result[2 * i + 1] = data[idx1];
        }
        return result;
    }
};
/** --------------------------------------------------------------------------------------------------------- luti2
 * @struct luti2
 * @brief A helper utility struct for working with the LUTI2 instruction, which performs a 2-bit indexed
 * lookup into a 4-element table.
 */
template<LUTCompatible T>
struct luti2 {
    /// @brief The underlying data array for the lookup table, aligned to 64 bytes.
    std::array<T, 4> data;
    /** --------------------------------------------------------------------------------- Default Constructor
     * @brief Default constructor initializes the lookup table with zeros
     */
    luti2(T init_val = 0) {
        data.fill(init_val);
    }
    /** --------------------------------------------------------------------------------- Construct from Array
     * @brief Constructor that initializes the lookup table with the provided array of
     * values. The input array must have exactly 4 elements, which will be copied into
     * the internal data array.
     * @param init_data The array of values to initialize the lookup table with.
     */
    luti2(const std::array<T, 4>& init_data) : data(init_data) {}
    /**
     * @brief Constructor that initializes the lookup table with the provided vector of
     * values. The input vector must have exactly 4 elements, which will be copied into
     * the internal data array.
     * @param init_data The vector of values to initialize the lookup table with
     * (must have 4 elements)
     */
    luti2(const std::vector<T>& init_data) {
        if (init_data.size() != 4) [[unlikely]] {
            throw std::invalid_argument("Initializer vector must have exactly"
                " 4 elements for luti2");
        }
        std::memcpy(data.data(), init_data.data(), 4 * sizeof(T));
    }
    /** --------------------------------------------------------------------------------- Construct from Initializer List
     * @brief Constructor that initializes the lookup table with the provided initializer
     * list of values. The input initializer list must have exactly 4 elements, which
     * will be copied into the internal data array.
     * @param init_list The initializer list of values to initialize the lookup table
     * with (must have 4 elements)
     */
    luti2(std::initializer_list<T> init_list) {
        if (init_list.size() != 4) [[unlikely]] {
            throw std::invalid_argument("Initializer list must have exactly"
                " 4 elements for luti2");
        }
        std::copy(init_list.begin(), init_list.end(), data.begin());
    }
    /** --------------------------------------------------------------------------------- Construct from Individual Values
     * @brief Constructor that initializes the lookup table with the provided individual
     * values. The constructor takes 4 individual values as parameters, which will be
     * copied into the internal data array.
     * @param val0 The first value to initialize the lookup table with
     * @param val1 The second value to initialize the lookup table with
     * @param val2 The third value to initialize the lookup table with
     * @param val3 The fourth value to initialize the lookup table with
     */
    luti2(T val0, T val1, T val2, T val3) : data{val0, val1, val2, val3} {}
    /** --------------------------------------------------------------------------------- Get Value
     * @brief Gets a reference to the value at the specified index in the lookup table.
     * The index should be in the range [0, 3], and the function will return a reference
     * to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 3])
     * @return A reference to the value at the specified index in the lookup table
     */
    T& operator[](size_t index) {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti2");
        }
        return data[index];
    }
    /** --------------------------------------------------------------------------------- Get Value (const)
     * @brief Gets a const reference to the value at the specified index in the lookup
     * table. The index should be in the range [0, 3], and the function will return a const reference to the corresponding value in the internal data array.
     * @param index The index of the value to get (should be in the range [0, 3])
     * @return A const reference to the value at the specified index in the lookup table
     */
    const T& operator[](size_t index) const {
        if (index >= data.size()) [[unlikely]] {
            throw std::out_of_range("Index out of range for luti2");
        }
        return data[index];
    }
    /** --------------------------------------------------------------------------------- Test Function
     * @brief A test function that simulates the behavior of the LUTI2 instruction.
     * NOTE: In case it isn't obvious, this is NOT the same as the actual LUTI2
     * instruction, and will exeute far slower than the real deal.
     * 
     * Call it from a bytecode program with the same table and input data you would use
     * for the LUTI2 instruction, and it will return a vector of output values that you
     * can compare against the output from the real LUTI2 to verify correctness of your
     * program.
     * 
     * @param ptr Pointer to the input data (should point to at least size_bytes of data)
     * @param size_bytes The size of the input data in bytes
     * @return A vector of output values produced by the LUTI2 lookup
     */
    std::vector<T> test(const void* ptr, size_t size_bytes) const {
        std::vector<T> result;
        result.resize(size_bytes * 4); // Each byte produces 4 output values
        for (size_t i = 0; i < size_bytes; i++) {
            uint8_t byte = reinterpret_cast<const uint8_t*>(ptr)[i];
            for (size_t j = 0; j < 4; j++) {
                uint8_t idx = (byte >> (2 * j)) & 0x03; // Extract 2 bits for index
                result[4 * i + j] = data[idx];
            }
        }
        return result;
    }
};
} // namespace ane