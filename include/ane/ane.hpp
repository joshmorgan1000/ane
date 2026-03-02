#pragma once
/** --------------------------------------------------------------------------------------------------------- Apple Neural Engine
 * @file ane.hpp
 * @brief Header file for the ane library.
 * Provides declarations for the lookup_table, nibble_pack, and crumb_pack utility structs, as well as
 * includes the assembly-optimized kernel function declarations from asm.hpp. This file serves as the main
 * public interface for the ane library, allowing users to access the optimized kernels and utility functions.
 * 
 * @author Josh Morgan
 * https://github.com/joshmorgan1000/ane
 * 
 * License: MIT License
 */
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <ane/extern/asm.hpp>

namespace ane {
/** --------------------------------------------------------------------------------------------------------- lookup_table
 * @struct lookup_table
 * @brief A helper utility struct.
 * This is a simple wrapper around a 64-entry lookup table, a utility struct to make the
 * LUTI2 and LUTI4 instructions easier to work with.
 */
struct lookup_table {
    /// @brief The raw data of the lookup table, aligned to 64 bytes for optimal access (32-bits per entry)
    alignas(64) uint8_t data[64];
    /** --------------------------------------------------------------------------------- Default Constructor
     * @brief Default constructor initializes the lookup table with zeros
     */
    lookup_table() {
        std::memset(data, 0, sizeof(data));
    }
    /** --------------------------------------------------------------------------------- Set 8-bit Value
     * @brief Sets the value at the specified index in the lookup table for uint8_t
     * @param i The index of the entry to set (0-15)
     * @param v The uint8_t value to set at the specified index
     */
    void set_u8(size_t i, uint8_t v) noexcept {
        data[i << 2] = v;
    }
    /** --------------------------------------------------------------------------------- Get 8-bit Value
     * @brief Gets the value at the specified index in the lookup table as a uint8_t
     * @param i The index of the entry to get (0-15)
     * @return The uint8_t value at the specified index in the lookup table
     */
    uint8_t get_u8(size_t i) const noexcept {
        return data[i << 2];
    }
    /** --------------------------------------------------------------------------------- Set 16-bit Unsigned Integer Value
     * @brief Sets the value at the specified index in the lookup table for uint16_t.
     * The value is stored as a 16-bit unsigned integer in the table, occupying
     * the first two bytes of a 32-bit entry.
     * @param i The index of the entry to set (0-15)
     * @param v The uint16_t value to set at the specified index
     */
    void set_u16(size_t i, uint16_t v) noexcept {
        std::memcpy(&data[i << 2], &v, 2);
    }
    /** --------------------------------------------------------------------------------- Get 16-bit Unsigned Integer Value
     * @brief Gets the value at the specified index in the lookup table as a uint16_t.
     * @param i The index of the entry to get (0-15)
     * @return The uint16_t value at the specified index in the lookup table
     */
    uint16_t get_u16(size_t i) const noexcept {
        uint16_t v;
        std::memcpy(&v, &data[i << 2], 2);
        return v;
    }
    /** --------------------------------------------------------------------------------- Set 32-bit Unsigned Integer Value
     * @brief Sets the value at the specified index in the lookup table for uint32_t.
     * The value is stored as a 32-bit unsigned integer in the table.
     * @param i The index of the entry to set (0-15)
     * @param v The uint32_t value to set at the specified index
     */
    void set_u32(size_t i, uint32_t v) noexcept {
        std::memcpy(&data[i << 2], &v, 4);
    }
    /** --------------------------------------------------------------------------------- Get 32-bit Unsigned Integer Value
     * @brief Gets the value at the specified index in the lookup table as a uint32_t.
     * @param i The index of the entry to get (0-15)
     * @return The uint32_t value at the specified index in the lookup table
     */
    uint32_t get_u32(size_t i) const noexcept {
        uint32_t v;
        std::memcpy(&v, &data[i << 2], 4);
        return v;
    }
    /** --------------------------------------------------------------------------------- Set 32-bit Float Value
     * @brief Sets the value at the specified index in the lookup table for float.
     * The float value is stored as a 32-bit entry in the table.
     * @param i The index of the entry to set (0-15)
     * @param v The float value to set at the specified index
     */
    void set_f32(size_t i, float v) noexcept {
        std::memcpy(&data[i << 2], &v, 4);
    }
    /** --------------------------------------------------------------------------------- Get 32-bit Float Value
     * @brief Gets the value at the specified index in the lookup table as a float.
     * @param i The index of the entry to get (0-15)
     * @return The float value at the specified index in the lookup table
     */
    float get_f32(size_t i) const noexcept {
        float v;
        std::memcpy(&v, &data[i << 2], 4);
        return v;
    }
    /** --------------------------------------------------------------------------------- Factory Method from uint8_t array
     * @brief Factory method to create a lookup table from an array of 16 uint8_t values.
     * Each value is stored in the first byte of a 32-bit entry in the table, with the remaining bytes set to zero.
     * @param entries An array of 16 uint8_t values to populate the lookup table
     * @return A lookup_table instance populated with the provided uint8_t values
     */
    static lookup_table from_u8_16(const uint8_t entries[16]) {
        lookup_table t;
        for (int i = 0; i < 16; i++) {
            t.data[i << 2] = entries[i];
        }
        return t;
    }
    /** --------------------------------------------------------------------------------- Factory Method from uint8_t initializer list
     * @brief Factory method to create a lookup table from an initializer list of
     * uint8_t values. Each value is stored in the first byte of a 32-bit entry
     * in the table, with the remaining bytes set to zero.
     * @param entries An initializer list of uint8_t values to populate the lookup
     * table (up to 16 values)
     * @return A lookup_table instance populated with the provided uint8_t values
     */
    static lookup_table from_u8_16(std::initializer_list<uint8_t> entries) {
        lookup_table t;
        int i = 0;
        for (auto v : entries) {
            if (i >= 16) {
                break;
            }
            t.data[i << 2] = v;
            i++;
        }
        return t;
    }
    /** --------------------------------------------------------------------------------- Factory Method from uint8_t array
     * @brief Factory method to create a lookup table from an array of 4 uint8_t values.
     * Each value is stored in the first byte of a 32-bit entry in the table, with
     * the remaining bytes set to zero.
     * @param entries An array of 4 uint8_t values to populate the lookup table
     * @return A lookup_table instance populated with the provided uint8_t values
     */
    static lookup_table from_u8_4(const uint8_t entries[4]) {
        lookup_table t;
        for (int i = 0; i < 4; i++) {
            t.data[i << 2] = entries[i];
        }
        return t;
    }
    /** --------------------------------------------------------------------------------- Factory Method from uint8_t initializer list
     * @brief Factory method to create a lookup table from an initializer list of
     * uint8_t values.
     * @param entries An initializer list of uint8_t values to populate the lookup
     * table (up to 4 values)
     * @return A lookup_table instance populated with the provided uint8_t values
     */
    static lookup_table from_u8_4(std::initializer_list<uint8_t> entries) {
        lookup_table t;
        int i = 0;
        for (auto v : entries) {
            if (i >= 4) {
                break;
            }
            t.data[i << 2] = v;
            i++;
        }
        return t;
    }
    /** --------------------------------------------------------------------------------- Factory Method from uint32_t array
     * @brief Factory method to create a lookup table from an array of 16 uint32_t values.
     * Each value is stored as a 32-bit unsigned integer in the table.
     * @param entries An array of 16 uint32_t values to populate the lookup table
     * @return A lookup_table instance populated with the provided uint32_t values
     */
    static lookup_table from_u32_16(const uint32_t entries[16]) {
        lookup_table t;
        for (int i = 0; i < 16; i++) {
            std::memcpy(&t.data[i << 2], &entries[i], 4);
        }
        return t;
    }
    /** --------------------------------------------------------------------------------- Factory Method from float array
     * @brief Factory method to create a lookup table from an array of 16 float values.
     * Each value is stored as a 32-bit float in the table.
     * @param entries An array of 16 float values to populate the lookup table
     * @return A lookup_table instance populated with the provided float values
     */
    static lookup_table from_f32_16(const float entries[16]) {
        lookup_table t;
        for (int i = 0; i < 16; i++) {
            std::memcpy(&t.data[i << 2], &entries[i], 4);
        }
        return t;
    }
    /** --------------------------------------------------------------------------------- Pointer Access
     * @brief Provides pointer access to the raw data of the lookup table (const version)
     * @return A pointer to the raw data of the lookup table
     */
    const uint8_t* ptr() const {
        return data;
    }
    /** --------------------------------------------------------------------------------- Pointer Access
     * @brief Provides pointer access to the raw data of the lookup table
     * @return A pointer to the raw data of the lookup table
     */
    uint8_t* ptr() {
        return data;
    }
    /** --------------------------------------------------------------------------------- Implicit Conversion Operator
     * @brief Implicit conversion operator to const uint8_t pointer (const version)
     */
    operator const uint8_t*() const {
        return data;
    }
    /** --------------------------------------------------------------------------------- Implicit Conversion Operator
     * @brief Implicit conversion operator to uint8_t pointer
     */
    operator uint8_t*() {
        return data;
    }
};
/** --------------------------------------------------------------------------------------------------------- nibble_pack
 * @struct nibble_pack
 * @brief Utility struct for packing and unpacking 4-bit values (nibbles) into bytes.
 * Provides methods to pack two 4-bit values into a single byte, as well as to
 * unpack them back into their original values. Also includes methods for working
 * with arrays of nibbles.
 */
struct nibble_pack {
    /** --------------------------------------------------------------------------------- Pack Nibbles
     * @brief Packs two 4-bit values (nibbles) into a single byte. The lower nibble
     * (bits 0-3) of the resulting byte contains val0, and the upper nibble
     * (bits 4-7) contains val1.
     * @param val0 The first 4-bit value to pack (only the lower 4 bits are used)
     * @param val1 The second 4-bit value to pack (only the lower 4 bits are used)
     * @return A byte containing the two packed nibbles
     */
    static uint8_t pack(uint8_t val0, uint8_t val1) {
        return static_cast<uint8_t>((val1 << 4) | (val0 & 0x0F));
    }
    /** --------------------------------------------------------------------------------- Unpack Nibbles
     * @brief Unpacks a byte containing two packed 4-bit values (nibbles) into their
     * original values. The lower nibble (bits 0-3) is extracted as val0, and the
     * upper nibble (bits 4-7) is extracted as val1.
     * @param packed The byte containing the two packed nibbles
     * @param v0 Output parameter to store the unpacked lower nibble value
     * @param v1 Output parameter to store the unpacked upper nibble value
     */
    static void unpack(uint8_t packed, uint8_t& v0, uint8_t& v1) {
        v0 = packed & 0x0F;
        v1 = (packed >> 4) & 0x0F;
    }
    /** --------------------------------------------------------------------------------- Pack Array
     * @brief Packs an array of 4-bit values (nibbles) into an array of bytes. Each byte
     * in the output array contains two packed nibbles from the input array. The input
     * array should have an even number of values, and the output array should have
     * enough space to hold n/2 bytes, where n is the number of values in the input
     * array.
     * @param values The array of 4-bit values (nibbles) to pack
     * @param packed The output array to store the packed bytes (should have space for
     * n/2 bytes)
     * @param n The number of values in the input array (should be even)
     */
    static void pack_array(const uint8_t* values, uint8_t* packed, size_t n) {
        for (size_t i = 0; i < n / 2; i++) {
            packed[i] = pack(values[i * 2], values[i * 2 + 1]);
        }
    }
    /** --------------------------------------------------------------------------------- Unpack Array
     * @brief Unpacks an array of packed bytes into an array of 4-bit values (nibbles).
     * Each byte in the input array contains two packed nibbles, which are unpacked
     * and stored in the output array. The output array should have enough space to
     * hold 2*n values, where n is the number of bytes in the input array.
     * @param packed The array of packed bytes containing the nibbles
     * @param values The output array to store the unpacked 4-bit values (nibbles)
     * @param n The number of bytes in the input packed array (the output array should
     * have space for 2*n values)
     */
    static void unpack_array(const uint8_t* packed, uint8_t* values, size_t n) {
        for (size_t i = 0; i < n / 2; i++) {
            unpack(packed[i], values[i * 2], values[i * 2 + 1]);
        }
    }
    /** --------------------------------------------------------------------------------- Get Nibble Value
     * @brief Gets a 4-bit value from the specified index in the packed array. The value is
     * extracted from either the lower or upper nibble of the corresponding byte, depending on
     * whether the index is even or odd.
     * @param packed The array of packed bytes containing the nibbles
     * @param index The index of the nibble to get (0-based)
     * @return The 4-bit value at the specified index in the packed array
     */
    static uint8_t get(const uint8_t* packed, size_t index) {
        uint8_t byte = packed[index / 2];
        return (index % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    }
    /** --------------------------------------------------------------------------------- Set Nibble Value
     * @brief Sets a 4-bit value at the specified index in the packed array. The value is
     * stored in either the lower or upper nibble of the corresponding byte, depending on
     * whether the index is even or odd.
     * @param packed The array of packed bytes containing the nibbles
     * @param index The index of the nibble to set (0-based)
     * @param value The 4-bit value to set at the specified index
     * (only the lower 4 bits are used)
     */
    static void set(uint8_t* packed, size_t index, uint8_t value) {
        size_t byte_idx = index / 2;
        if (index % 2 == 0) {
            packed[byte_idx] = (packed[byte_idx] & 0xF0) | (value & 0x0F);
        } else {
            packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((value & 0x0F) << 4);
        }
    }
};
/** --------------------------------------------------------------------------------------------------------- crumb_pack
 * @struct crumb_pack
 * @brief Utility struct for packing and unpacking 2-bit values (crumbs) into bytes.
 * Provides methods to pack four 2-bit values into a single byte, as well as to
 * unpack them back into their original values. Also includes methods for working
 * with arrays of crumbs.
 */
struct crumb_pack {
    /** --------------------------------------------------------------------------------- Pack Crumbs
     * @brief Packs four 2-bit values (crumbs) into a single byte. Each 2-bit value is
     * stored in a specific position within the byte: bits 0-1 for v0, bits 2-3 for v1,
     * bits 4-5 for v2, and bits 6-7 for v3.
     * @param v0 The first 2-bit value to pack (only the lower 2 bits are used)
     * @param v1 The second 2-bit value to pack (only the lower 2 bits are used)
     * @param v2 The third 2-bit value to pack (only the lower 2 bits are used)
     * @param v3 The fourth 2-bit value to pack (only the lower 2 bits are used)
     * @return A byte containing the four packed crumbs
     */
    static uint8_t pack(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
        return static_cast<uint8_t>(
            (v0 & 0x03) | ((v1 & 0x03) << 2) | ((v2 & 0x03) << 4) | ((v3 & 0x03) << 6)
        );
    }
    /** --------------------------------------------------------------------------------- Unpack Crumbs
     * @brief Unpacks a byte containing four packed 2-bit values (crumbs) into their
     * original values. Each 2-bit value is extracted from the corresponding position
     * within the byte (bits 0-1 for v0, bits 2-3 for v1, bits 4-5 for v2, bits 6-7 for v3).
     * @param packed The byte containing the four packed crumbs
     * @param v0 Output parameter to store the unpacked first crumb value
     * @param v1 Output parameter to store the unpacked second crumb value
     * @param v2 Output parameter to store the unpacked third crumb value
     * @param v3 Output parameter to store the unpacked fourth crumb value
     */
    static void unpack(uint8_t packed, uint8_t& v0, uint8_t& v1, uint8_t& v2, uint8_t& v3) {
        v0 = packed & 0x03;
        v1 = (packed >> 2) & 0x03;
        v2 = (packed >> 4) & 0x03;
        v3 = (packed >> 6) & 0x03;
    }
    /** --------------------------------------------------------------------------------- Pack Array
     * @brief Packs an array of 2-bit values (crumbs) into an array of bytes. Each byte
     * in the output array contains four packed crumbs from the input array. The input
     * array should have a number of values that is a multiple of 4, and the output
     * array should have enough space to hold n/4 bytes, where n is the number of
     * values in the input array.
     * @param values The array of 2-bit values (crumbs) to pack
     * @param packed The output array to store the packed bytes (should have space for
     * n/4 bytes)
     * @param n The number of values in the input array (should be a multiple of 4)
     */
    static void pack_array(const uint8_t* values, uint8_t* packed, size_t n) {
        for (size_t i = 0; i < n >> 2; i++) {
            packed[i] = pack(
                values[i << 2],
                values[(i << 2) + 1],
                values[(i << 2) + 2],
                values[(i << 2) + 3]
            );
        }
    }
    /** --------------------------------------------------------------------------------- Unpack Array
     * @brief Unpacks an array of packed bytes into an array of 2-bit values (crumbs).
     * Each byte in the input array contains four packed crumbs, which are unpacked
     * and stored in the output array. The output array should have enough space to
     * hold 4*n values, where n is the number of bytes in the input array.
     * @param packed The array of packed bytes containing the crumbs
     * @param values The output array to store the unpacked 2-bit values (crumbs)
     * @param n The number of values in the output array (should be a multiple of 4)
     */
    static void unpack_array(const uint8_t* packed, uint8_t* values, size_t n) {
        for (size_t i = 0; i < n >> 2; i++) {
            unpack(
                packed[i],
                values[i << 2],
                values[(i << 2) + 1],
                values[(i << 2) + 2],
                values[(i << 2) + 3]
            );
        }
    }
    /** --------------------------------------------------------------------------------- Get Crumb Value
     * @brief Gets a 2-bit value from the specified index in the packed array. The value
     * is extracted from one of the four 2-bit positions within the corresponding byte,
     * depending on the index modulo 4.
     * @param packed The array of packed bytes containing the crumbs
     * @param index The index of the crumb to get (0-based)
     * @return The 2-bit value at the specified index in the packed array
     */
    static uint8_t get(const uint8_t* packed, size_t index) {
        uint8_t byte = packed[index >> 2];
        int shift = static_cast<int>((index & 3) << 1);
        return (byte >> shift) & 0x03;
    }
    /** --------------------------------------------------------------------------------- Set Crumb Value
     * @brief Sets a 2-bit value at the specified index in the packed array. The value is
     * stored in one of the four 2-bit positions within the corresponding byte, depending
     * on the index modulo 4.
     * @param packed The array of packed bytes containing the crumbs
     * @param index The index of the crumb to set (0-based)
     * @param value The 2-bit value to set at the specified index (only the lower 2 bits
     * are used)
     */
    static void set(uint8_t* packed, size_t index, uint8_t value) {
        size_t byte_idx = index >> 2;
        int shift = static_cast<int>((index & 3) << 1);
        packed[byte_idx] = (packed[byte_idx] & ~(0x03 << shift)) |
                           ((value & 0x03) << shift);
    }
};
/** --------------------------------------------------------------------------------------------------------- tbl_indices
 * @struct tbl_indices
 * @brief Utility struct for validating and manipulating indices used with lookup tables.
 * Provides methods to validate that indices are within the bounds of a specified table size,
 * count how many indices are out of range, and clamp indices to ensure they do not exceed
 * the maximum allowed value for the table.
 */
struct tbl_indices {
    /** --------------------------------------------------------------------------------- Validate Table Indices
     * @brief Validates that all indices in the given array are within the bounds of the
     * specified table size.
     * @param indices The array of indices to validate
     * @param n The number of indices in the array
     * @param n_table The size of the lookup table (default is 64)
     * @return true if all indices are valid (less than n_table), false if any index is
     * out of range
     */
    static bool validate_tbl(const uint8_t* indices, size_t n, size_t n_table = 64) {
        for (size_t i = 0; i < n; i++) {
            if (indices[i] >= n_table) {
                return false;
            }
        }
        return true;
    }
    /** --------------------------------------------------------------------------------- Validate Table Indices
     * @brief Validates that all indices in the given array are within the bounds of the
     * specified table size.
     * @param indices The array of indices to validate
     * @param n The number of indices in the array
     * @param n_table The size of the lookup table (default is 64)
     * @return true if all indices are valid (less than n_table), false if any index is
     * out of range
     */
    static bool validate_tbl2(const uint8_t* indices, size_t n, size_t n_table = 128) {
        for (size_t i = 0; i < n; i++) {
            if (indices[i] >= n_table) {
                return false;
            }
        }
        return true;
    }
    /** --------------------------------------------------------------------------------- Count Out-of-Range Indices
     * @brief Counts how many indices in the given array are out of range for the
     * specified table size.
     * @param indices The array of indices to check
     * @param n The number of indices in the array
     * @param n_table The size of the lookup table (default is 64)
     * @return The number of indices that are out of range for the table
     */
    static size_t count_out_of_range(const uint8_t* indices, size_t n, size_t n_table) {
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            if (indices[i] >= n_table) {
                count++;
            }
        }
        return count;
    }
    /** --------------------------------------------------------------------------------- Clamp Indices
     * @brief Clamps the values in the indices array to ensure they do not exceed the
     * maximum allowed value for the table.
     * @param indices The array of indices to clamp
     * @param n The number of indices in the array
     * @param n_table The size of the lookup table (default is 64).
     */
    static void clamp(uint8_t* indices, size_t n, size_t n_table) {
        for (size_t i = 0; i < n; i++) {
            if (indices[i] >= n_table) {
                indices[i] = 0;
            }
        }
    }
};
/** --------------------------------------------------------------------------------------------------------- luti4_u8
 * @brief Performs a LUTI4 (lookup table with 4-bit indices) operation using the provided lookup table and
 * packed indices.
 * @param lut64 The lookup table to use for the operation, containing 64 entries (256 bytes)
 * @param packed_indices A pointer to the array of packed indices, where each byte contains two 4-bit
 * indices
 * @param output A pointer to the output array where the results will be stored
 * @param n The number of output elements to produce (should be equal to the number of 4-bit indices in the
 * packed_indices array)
 */
inline void luti4_u8(const lookup_table& lut64, const uint8_t* packed_indices, uint8_t* output, long n) {
    ane::kernel::luti4_u8(lut64.ptr(), packed_indices, output, n);
}
/** --------------------------------------------------------------------------------------------------------- luti2_u8
 * @brief Performs a LUTI2 (lookup table with 2-bit indices) operation using the provided lookup table and
 * packed indices.
 * @param lut64 The lookup table to use for the operation, containing 64 entries (256 bytes)
 * @param packed_indices A pointer to the array of packed indices, where each byte contains four 2-bit
 * indices
 * @param output A pointer to the output array where the results will be stored
 * @param n The number of output elements to produce (should be equal to the number of 2-bit indices in the
 * packed_indices array)
 */
inline void luti2_u8(const lookup_table& lut64, const uint8_t* packed_indices, uint8_t* output, long n) {
    ane::kernel::luti2_u8(lut64.ptr(), packed_indices, output, n);
}
/** --------------------------------------------------------------------------------------------------------- lut_expand_4bit
 * @brief Expands packed 4-bit indices into their corresponding values from the lookup table.
 * @param lut64 The lookup table to use for the operation, containing 64 entries (256 bytes)
 * @param packed_indices A pointer to the array of packed indices, where each byte contains two 4-bit
 * indices
 * @param out A pointer to the output array where the expanded values will be stored
 * @param n The number of output elements to produce (should be equal to the number of 4-bit indices in the
 * packed_indices array)
 */
static inline void lut_expand_4bit(
    const uint8_t* lut64, const uint8_t* packed_indices, uint8_t* out, long n
) {
    ane::kernel::luti4_u8(lut64, packed_indices, out, n);
}
/** --------------------------------------------------------------------------------------------------------- lut_expand_2bit
 * @brief Expands packed 2-bit indices into their corresponding values from the lookup table.
 * @param lut64 The lookup table to use for the operation, containing 64 entries (256 bytes)
 * @param packed_indices A pointer to the array of packed indices, where each byte contains four 2-bit
 * indices
 * @param out A pointer to the output array where the expanded values will be stored
 * @param n The number of output elements to produce (should be equal to the number of 2-bit indices in the
 * packed_indices array)
 */
static inline void lut_expand_2bit(
    const uint8_t* lut64, const uint8_t* packed_indices, uint8_t* out, long n
) {
    ane::kernel::luti2_u8(lut64, packed_indices, out, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Addition
 * @brief Element-wise addition of two FP32 vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void add_fp32(const float* a, const float* b, float* c, long n) {
    ane::kernel::add_fp32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Subtraction
 * @brief Element-wise subtraction of two FP32 vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void sub_fp32(const float* a, const float* b, float* c, long n) {
    ane::kernel::sub_fp32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Multiplication
 * @brief Element-wise multiplication of two FP32 vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void mul_fp32(const float* a, const float* b, float* c, long n) {
    ane::kernel::mul_fp32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Division
 * @brief Element-wise division of two FP32 vectors
 * @param a Pointer to the first input vector (numerator)
 * @param b Pointer to the second input vector (denominator)
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void div_fp32(const float* a, const float* b, float* c, long n) {
    ane::kernel::div_fp32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Maximum
 * @brief Computes the element-wise maximum of two input vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vectors
 */
static inline void max_fp32(const float* a, const float* b, float* c, long n) {
    ane::kernel::max_fp32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Minimum
 * @brief Computes the element-wise minimum of two input vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vectors
 */
static inline void min_fp32(const float* a, const float* b, float* c, long n) {
    ane::kernel::min_fp32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Fused Multiply-Add
 * @brief Fused multiply-add operation: out = a * b + c
 * @param a Pointer to the first multiplicand vector
 * @param b Pointer to the second multiplicand vector
 * @param c Pointer to the addend vector
 * @param out Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void fma_fp32(const float* a, const float* b, const float* c, float* out, long n) {
    ane::kernel::fma_fp32(a, b, c, out, n);
}
/** --------------------------------------------------------------------------------------------------------- Scalar Multiplication
 * @brief Multiplies each element of the input vector by a scalar value
 * @param input Pointer to the input vector
 * @param scalar The scalar value to multiply each element by
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void scalar_mul_fp32(const float* input, float scalar, float* output, long n) {
    ane::kernel::scalar_mul_fp32(input, scalar, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Scalar Addition
 * @brief Adds a scalar value to each element of the input vector
 * @param input Pointer to the input vector
 * @param scalar The scalar value to add to each element
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void scalar_add_fp32(const float* input, float scalar, float* output, long n) {
    ane::kernel::scalar_add_fp32(input, scalar, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Clamp
 * @brief Clamps each element of the input vector to the range [lo, hi]
 * @param input Pointer to the input vector
 * @param lo The lower bound for clamping
 * @param hi The upper bound for clamping
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void clamp_fp32(const float* input, float lo, float hi, float* output, long n) {
    ane::kernel::clamp_fp32(input, lo, hi, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Fill
 * @brief Fills the output vector with a scalar value
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param value The scalar value to fill the vector with
 * @param n The number of elements in the vector
 */
static inline void fill_fp32(float* output, float value, long n) {
    ane::kernel::fill_fp32(output, value, n);
}
/** --------------------------------------------------------------------------------------------------------- Copy
 * @brief Copies elements from the source vector to the destination vector
 * @param src Pointer to the source vector
 * @param dst Pointer to the destination vector (must be pre-allocated)
 * @param n The number of elements to copy
 */
static inline void copy_fp32(const float* src, float* dst, long n) {
    ane::kernel::copy_fp32(src, dst, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Negation
 * @brief Element-wise negation of an FP32 vector
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void neg_fp32(const float* input, float* output, long n) {
    ane::kernel::neg_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Element-wise Absolute Value
 * @brief Element-wise absolute value of an FP32 vector
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void abs_fp32(const float* input, float* output, long n) {
    ane::kernel::abs_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Square Root
 * @brief Computes the element-wise square root of the input vector
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void sqrt_fp32(const float* input, float* output, long n) {
    ane::kernel::sqrt_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Reciprocal Square Root
 * @brief Computes the element-wise reciprocal square root of the input vector
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void rsqrt_fp32(const float* input, float* output, long n) {
    ane::kernel::rsqrt_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Exponential
 * @brief Computes the element-wise exponential (e^x)
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void exp_fp32(const float* input, float* output, long n) {
    ane::kernel::exp_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Natural Logarithm
 * @brief Computes the element-wise natural logarithm (ln(x))
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void log_fp32(const float* input, float* output, long n) {
    ane::kernel::log_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Rectified Linear Unit
 * @brief Computes the element-wise ReLU activation (max(x, 0))
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void relu_fp32(const float* input, float* output, long n) {
    ane::kernel::relu_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Sigmoid
 * @brief Computes the element-wise sigmoid activation (1 / (1 + exp(-x)))
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void sigmoid_fp32(const float* input, float* output, long n) {
    ane::kernel::sigmoid_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Hyperbolic Tangent
 * @brief Computes the element-wise hyperbolic tangent activation
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void tanh_fp32(const float* input, float* output, long n) {
    ane::kernel::tanh_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- GELU
 * @brief Computes the element-wise GELU (Gaussian Error Linear Unit) activation
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void gelu_fp32(const float* input, float* output, long n) {
    ane::kernel::gelu_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- SiLU
 * @brief Computes the element-wise SiLU (Swish) activation: x * sigmoid(x)
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void silu_fp32(const float* input, float* output, long n) {
    ane::kernel::silu_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Softmax
 * @brief Computes the softmax activation over a vector
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void softmax_fp32(const float* input, float* output, long n) {
    ane::kernel::softmax_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Softmax Batch
 * @brief Computes softmax over batches (applies softmax per sequence in a batch)
 * @param input Pointer to the input tensor (flattened, batch_size * seq_len elements)
 * @param output Pointer to the output tensor (must be pre-allocated)
 * @param batch_size The number of sequences in the batch
 * @param seq_len The length of each sequence
 */
static inline void softmax_batch_fp32(const float* input, float* output, long batch_size, long seq_len) {
    ane::kernel::softmax_batch_fp32(input, output, batch_size, seq_len);
}
/** --------------------------------------------------------------------------------------------------------- Power
 * @brief Computes element-wise power: output = base^exponent
 * @param base Pointer to the base vector
 * @param exponent Pointer to the exponent vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void pow_fp32(const float* base, const float* exponent, float* output, long n) {
    ane::kernel::pow_fp32(base, exponent, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Reduce Sum
 * @brief Computes the sum of all elements in the input vector
 * @param input Pointer to the input vector
 * @param n The total number of elements in the vector
 * @return The sum of all elements
 */
static inline float reduce_sum_fp32(const float* input, long n) {
    return ane::kernel::reduce_sum_fp32(input, n);
}
/** --------------------------------------------------------------------------------------------------------- Reduce Max
 * @brief Computes the maximum element in the input vector
 * @param input Pointer to the input vector
 * @param n The total number of elements in the vector
 * @return The maximum element
 */
static inline float reduce_max_fp32(const float* input, long n) {
    return ane::kernel::reduce_max_fp32(input, n);
}
/** --------------------------------------------------------------------------------------------------------- Reduce Min
 * @brief Computes the minimum element in the input vector
 * @param input Pointer to the input vector
 * @param n The total number of elements in the vector
 * @return The minimum element
 */
static inline float reduce_min_fp32(const float* input, long n) {
    return ane::kernel::reduce_min_fp32(input, n);
}
/** --------------------------------------------------------------------------------------------------------- Global Max Pool FP32
 * @brief Global max pooling operation: reduces over spatial dimensions for each batch and channel
 * @param input Pointer to input tensor [batch, spatial, channels]
 * @param output Pointer to output tensor [batch, channels]
 * @param batch Number of batches
 * @param spatial Total spatial size (H*W flattened)
 * @param channels Number of channels
 */
static inline void global_max_pool_fp32(
    const float* input, float* output, long batch, long spatial, long channels
) {
    return ane::kernel::global_max_pool_fp32(input, output, batch, spatial, channels);
}
/** --------------------------------------------------------------------------------------------------------- Global Avg Pool FP32
 * @brief Global average pooling operation: averages over spatial dimensions for each batch and channel
 * @param input Pointer to input tensor [batch, spatial, channels]
 * @param output Pointer to output tensor [batch, channels]
 * @param batch Number of batches
 * @param spatial Total spatial size (H*W flattened)
 * @param channels Number of channels
 */
static inline void global_avg_pool_fp32(
    const float* input, float* output, long batch, long spatial, long channels
) {
    return ane::kernel::global_avg_pool_fp32(input, output, batch, spatial, channels);
}
/** --------------------------------------------------------------------------------------------------------- Dot Product
 * @brief Computes the dot product of two vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param n The total number of elements in each vector
 * @return The dot product of a and b
 */
static inline float dot_fp32(const float* a, const float* b, long n) {
    return ane::kernel::dot_fp32(a, b, n);
}
/** --------------------------------------------------------------------------------------------------------- Sum of Squares
 * @brief Computes the sum of squared elements in the input vector
 * @param input Pointer to the input vector
 * @param n The total number of elements in the vector
 * @return The sum of squared elements
 */
static inline float sumsqr_fp32(const float* input, long n) {
    return ane::kernel::sumsqr_fp32(input, n);
}
/** --------------------------------------------------------------------------------------------------------- Matrix Multiplication
 * @brief Computes C = A * B (row-major)
 * @param A Pointer to matrix A (M x K, row-major)
 * @param B Pointer to matrix B (K x N, row-major)
 * @param C Pointer to output matrix C (M x N, row-major, must be pre-allocated)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
static inline void matmul_fp32(const float* A, const float* B, float* C, long M, long N, long K) {
    ane::kernel::matmul_fp32(A, B, C, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Matrix Multiplication (Transpose)
 * @brief Computes C = A^T * B (A is transposed)
 * @param A Pointer to matrix A (K x M, row-major, will be treated as transposed)
 * @param B Pointer to matrix B (K x N, row-major)
 * @param C Pointer to output matrix C (M x N, row-major, must be pre-allocated)
 * @param M Number of rows in A's transposed view (columns in original layout)
 * @param N Number of columns in B and C
 * @param K Number of rows in A and B
 */
static inline void matmul_fp32_tn(const float* A, const float* B, float* C, long M, long N, long K) {
    ane::kernel::matmul_fp32_tn(A, B, C, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Matrix Transpose
 * @brief Transposes a matrix from row-major to column-major (or vice versa)
 * @param src Pointer to the source matrix in row-major format
 * @param dst Pointer to the destination matrix (must be pre-allocated)
 * @param rows The number of rows in the source matrix
 * @param cols The number of columns in the source matrix
 */
static inline void transpose_fp32(const float* src, float* dst, long rows, long cols) {
    ane::kernel::transpose_fp32(src, dst, rows, cols);
}
/** --------------------------------------------------------------------------------------------------------- FP32 to BF16 Conversion
 * @brief Converts FP32 (32-bit float) to BF16 (bfloat16) format
 * @param input Pointer to the input FP32 vector
 * @param output Pointer to the output BF16 vector (must be pre-allocated)
 * @param n The number of elements to convert
 */
static inline void fp32_to_bf16(const float* input, bfloat16_t* output, long n) {
    ane::kernel::fp32_to_bf16(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- BF16 to FP32 Conversion
 * @brief Converts BF16 (bfloat16) to FP32 (32-bit float) format
 * @param input Pointer to the input BF16 vector
 * @param output Pointer to the output FP32 vector (must be pre-allocated)
 * @param n The number of elements to convert
 */
static inline void bf16_to_fp32(const bfloat16_t* input, float* output, long n) {
    ane::kernel::bf16_to_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- FP32 to INT32 Conversion
 * @brief Converts FP32 to INT32 (truncating towards zero)
 * @param input Pointer to the input FP32 vector
 * @param output Pointer to the output INT32 vector (must be pre-allocated)
 * @param n The number of elements to convert
 */
static inline void fp32_to_int32(const float* input, int32_t* output, long n) {
    ane::kernel::fp32_to_int32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- INT32 to FP32 Conversion
 * @brief Converts INT32 to FP32
 * @param input Pointer to the input INT32 vector
 * @param output Pointer to the output FP32 vector (must be pre-allocated)
 * @param n The number of elements to convert
 */
static inline void int32_to_fp32(const int32_t* input, float* output, long n) {
    ane::kernel::int32_to_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Quantize FP32 to INT8
 * @brief Quantizes FP32 values to INT8 using a linear scale
 * @param input Pointer to the input FP32 vector
 * @param output Pointer to the output INT8 vector (must be pre-allocated)
 * @param scale The quantization scale factor
 * @param n The number of elements to quantize
 */
static inline void quantize_fp32_int8(const float* input, int8_t* output, float scale, long n) {
    ane::kernel::quantize_fp32_int8(input, output, scale, n);
}
/** --------------------------------------------------------------------------------------------------------- Dequantize INT8 to FP32
 * @brief Dequantizes INT8 values back to FP32 using a linear scale
 * @param input Pointer to the input INT8 vector
 * @param output Pointer to the output FP32 vector (must be pre-allocated)
 * @param scale The dequantization scale factor
 * @param n The number of elements to dequantize
 */
static inline void dequantize_int8_fp32(const int8_t* input, float* output, float scale, long n) {
    ane::kernel::dequantize_int8_fp32(input, output, scale, n);
}
/** --------------------------------------------------------------------------------------------------------- Fused RMS Norm + Scale
 * @brief Fused RMS normalization and scaling: out = (input / rms) * weight
 * @param input Pointer to the input vector
 * @param weight Pointer to the scale/weight vector
 * @param out Pointer to the output vector (must be pre-allocated)
 * @param inv_rms The pre-computed inverse RMS value for the normalization
 * @param n The number of elements in each vector
 */
static inline void fused_rms_norm_scale_fp32(
    const float* input, const float* weight, float* out, float inv_rms, long n
) {
    ane::kernel::fused_rms_norm_scale_fp32(input, weight, out, inv_rms, n);
}
/** --------------------------------------------------------------------------------------------------------- Fused SiLU Gate Multiplication
 * @brief Fused SiLU gating and multiplication: out = SiLU(gate) * up
 * @param gate Pointer to the gate vector
 * @param up Pointer to the up-projection vector
 * @param out Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void fused_silu_gate_mul_fp32(const float* gate, const float* up, float* out, long n) {
    ane::kernel::fused_silu_gate_mul_fp32(gate, up, out, n);
}
/** --------------------------------------------------------------------------------------------------------- Fused Weighted Add
 * @brief Fused weighted accumulation: out = acc + w * expert_out
 * @param acc Pointer to the accumulator vector
 * @param expert_out Pointer to the expert output vector
 * @param out Pointer to the output vector (must be pre-allocated)
 * @param w The weight scalar for the expert output
 * @param n The number of elements in each vector
 */
static inline void fused_weighted_add_fp32(
    const float* acc, const float* expert_out, float* out, float w, long n
) {
    ane::kernel::fused_weighted_add_fp32(acc, expert_out, out, w, n);
}
/** --------------------------------------------------------------------------------------------------------- Fused SSM Gate
 * @brief Fused SSM (State Space Model) gating: out = gate * y
 * @param gate Pointer to the gate vector
 * @param y Pointer to the state/output vector
 * @param out Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void fused_ssm_gate_fp32(const float* gate, const float* y, float* out, long n) {
    ane::kernel::fused_ssm_gate_fp32(gate, y, out, n);
}
/** --------------------------------------------------------------------------------------------------------- Fused Residual + Sum of Squares
 * @brief Fused residual connection and sum of squares: hidden = residual + x, ss_out = sum(x^2)
 * @param residual Pointer to the residual vector
 * @param x Pointer to the input vector
 * @param hidden Pointer to the hidden output vector (must be pre-allocated)
 * @param ss_out Pointer to the sum of squares output
 * @param n The total number of elements in each vector
 */
static inline void fused_residual_sumsqr_fp32(
    const float* residual, const float* x, float* hidden, float* ss_out, long n
) {
    ane::kernel::fused_residual_sumsqr_fp32(residual, x, hidden, ss_out, n);
}
/** --------------------------------------------------------------------------------------------------------- Bitwise AND
 * @brief Element-wise bitwise AND of two uint32 vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void and_u32(const uint32_t* a, const uint32_t* b, uint32_t* c, long n) {
    ane::kernel::and_u32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Bitwise OR
 * @brief Element-wise bitwise OR of two uint32 vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void or_u32(const uint32_t* a, const uint32_t* b, uint32_t* c, long n) {
    ane::kernel::or_u32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Bitwise XOR
 * @brief Element-wise bitwise XOR of two uint32 vectors
 * @param a Pointer to the first input vector
 * @param b Pointer to the second input vector
 * @param c Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in each vector
 */
static inline void xor_u32(const uint32_t* a, const uint32_t* b, uint32_t* c, long n) {
    ane::kernel::xor_u32(a, b, c, n);
}
/** --------------------------------------------------------------------------------------------------------- Bitwise NOT
 * @brief Element-wise bitwise NOT of a uint32 vector
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements in the vector
 */
static inline void not_u32(const uint32_t* input, uint32_t* output, long n) {
    ane::kernel::not_u32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution
 * @brief Performs a 2D convolution on an NHWC format input tensor
 * @param input_NHWC Pointer to input tensor in NHWC format (batch, height, width, channels_in)
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * (kernel_height, kernel_width, channels_in, channels_out)
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_fp32(
    const float* input_NHWC, const float* weight_HWIO, float* output_NHWC,
    long N, long H, long W, long C_in, long KH, long KW, long C_out,
    long stride, long pad
) {
    ane::kernel::conv2d_fp32(
        input_NHWC, weight_HWIO, output_NHWC, N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with Bias
 * @brief Performs a 2D convolution with bias on an NHWC format input tensor
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param bias Pointer to bias vector (C_out elements)
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_bias_fp32(
    const float* input_NHWC, const float* weight_HWIO, const float* bias,
    float* output_NHWC, long N, long H, long W, long C_in, long KH,
    long KW, long C_out, long stride, long pad
) {
    ane::kernel::conv2d_bias_fp32(
        input_NHWC, weight_HWIO, bias, output_NHWC, N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with Bias and ReLU
 * @brief Performs a 2D convolution with bias and ReLU activation
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param bias_C Pointer to bias vector (C_out elements)
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_bias_relu_fp32(
    const float* input_NHWC, const float* weight_HWIO, const float* bias_C,
    float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW,
    long C_out, long stride, long pad
) {
    ane::kernel::conv2d_bias_relu_fp32(
        input_NHWC, weight_HWIO, bias_C, output_NHWC, N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with Bias and ReLU6
 * @brief Performs a 2D convolution with bias and ReLU6 activation (clamp to [0, 6])
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param bias_C Pointer to bias vector (C_out elements)
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_bias_relu6_fp32(
    const float* input_NHWC, const float* weight_HWIO, const float* bias_C,
    float* output_NHWC, long N, long H, long W, long C_in, long KH, long KW,
    long C_out, long stride, long pad
) {
    ane::kernel::conv2d_bias_relu6_fp32(
        input_NHWC, weight_HWIO, bias_C, output_NHWC, N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with ReLU
 * @brief Performs a 2D convolution with ReLU activation (no bias)
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_relu_fp32(
    const float* input_NHWC, const float* weight_HWIO, float* output_NHWC,
    long N, long H, long W, long C_in, long KH, long KW, long C_out,
    long stride, long pad
) {
    ane::kernel::conv2d_relu_fp32(
        input_NHWC, weight_HWIO, output_NHWC, N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with Batch Normalization
 * @brief Performs a 2D convolution with batch normalization (no bias)
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param bn_scale Pointer to batch norm scale vector (C_out elements)
 * @param bn_shift Pointer to batch norm shift/bias vector (C_out elements)
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_bn_fp32(
    const float* input_NHWC, const float* weight_HWIO, const float* bn_scale,
    const float* bn_shift, float* output_NHWC, long N, long H, long W,
    long C_in, long KH, long KW, long C_out, long stride, long pad
) {
    ane::kernel::conv2d_bn_fp32(
        input_NHWC, weight_HWIO, bn_scale, bn_shift, output_NHWC,
        N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with Batch Normalization and ReLU
 * @brief Performs a 2D convolution with batch normalization and ReLU activation
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param bn_scale Pointer to batch norm scale vector (C_out elements)
 * @param bn_shift Pointer to batch norm shift/bias vector (C_out elements)
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_bn_relu_fp32(
    const float* input_NHWC, const float* weight_HWIO, const float* bn_scale,
    const float* bn_shift, float* output_NHWC, long N, long H, long W,
    long C_in, long KH, long KW, long C_out, long stride, long pad
) {
    ane::kernel::conv2d_bn_relu_fp32(
        input_NHWC, weight_HWIO, bn_scale, bn_shift, output_NHWC,
        N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with SiLU
 * @brief Performs a 2D convolution with SiLU (Swish) activation
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_swish_fp32(
    const float* input_NHWC, const float* weight_HWIO, float* output_NHWC,
    long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad
) {
    ane::kernel::conv2d_swish_fp32(
        input_NHWC, weight_HWIO, output_NHWC, N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- 2D Convolution with GELU
 * @brief Performs a 2D convolution with GELU activation
 * @param input_NHWC Pointer to input tensor in NHWC format
 * @param weight_HWIO Pointer to weight tensor in HWIO format
 * @param output_NHWC Pointer to output tensor in NHWC format (must be pre-allocated)
 * @param N Batch size
 * @param H Input height
 * @param W Input width
 * @param C_in Number of input channels
 * @param KH Kernel height
 * @param KW Kernel width
 * @param C_out Number of output channels
 * @param stride Convolution stride
 * @param pad Zero-padding amount
 */
static inline void conv2d_gelu_fp32(
    const float* input_NHWC, const float* weight_HWIO, float* output_NHWC,
    long N, long H, long W, long C_in, long KH, long KW, long C_out, long stride, long pad
) {
    ane::kernel::conv2d_gelu_fp32(
        input_NHWC, weight_HWIO, output_NHWC, N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- Dense with ReLU
 * @brief Performs a dense (fully connected) layer with ReLU activation: output = max(0, input @ weight)
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_relu_fp32(
    const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_relu_fp32(input_MK, weight_KN, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Softmax
 * @brief Performs a dense (fully connected) layer with softmax activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_softmax_fp32(
    const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_softmax_fp32(input_MK, weight_KN, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Bias
 * @brief Performs a dense (fully connected) layer with bias: output = input @ weight + bias
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param bias_N Pointer to bias vector (N elements)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_bias_fp32(
    const float* input_MK, const float* weight_KN, const float* bias_N,
    float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_bias_fp32(input_MK, weight_KN, bias_N, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Bias and ReLU
 * @brief Performs a dense (fully connected) layer with bias and ReLU activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param bias_N Pointer to bias vector (N elements)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_bias_relu_fp32(
    const float* input_MK, const float* weight_KN, const float* bias_N,
    float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_bias_relu_fp32(input_MK, weight_KN, bias_N, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with GELU
 * @brief Performs a dense (fully connected) layer with GELU activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_gelu_fp32(
    const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_gelu_fp32(input_MK, weight_KN, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with SiLU
 * @brief Performs a dense (fully connected) layer with SiLU activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_silu_fp32(
    const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_silu_fp32(input_MK, weight_KN, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Tanh
 * @brief Performs a dense (fully connected) layer with Tanh activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_tanh_fp32(
    const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_tanh_fp32(input_MK, weight_KN, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Sigmoid
 * @brief Performs a dense (fully connected) layer with Sigmoid activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_sigmoid_fp32(
    const float* input_MK, const float* weight_KN, float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_sigmoid_fp32(input_MK, weight_KN, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Bias and GELU
 * @brief Performs a dense (fully connected) layer with bias and GELU activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param bias_N Pointer to bias vector (N elements)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_bias_gelu_fp32(
    const float* input_MK, const float* weight_KN, const float* bias_N,
    float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_bias_gelu_fp32(input_MK, weight_KN, bias_N, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Bias and SiLU
 * @brief Performs a dense (fully connected) layer with bias and SiLU activation
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param bias_N Pointer to bias vector (N elements)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_bias_silu_fp32(
    const float* input_MK, const float* weight_KN, const float* bias_N,
    float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_bias_silu_fp32(input_MK, weight_KN, bias_N, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Residual
 * @brief Performs a dense (fully connected) layer with residual connection:
 * output = input @ weight + residual
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param residual_MN Pointer to residual matrix (M x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_residual_fp32(
    const float* input_MK, const float* weight_KN, const float* residual_MN,
    float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_residual_fp32(input_MK, weight_KN, residual_MN, output_MN, M, N, K);
}
/** --------------------------------------------------------------------------------------------------------- Dense with Bias and Residual
 * @brief Performs a dense (fully connected) layer with bias and residual connection
 * @param input_MK Pointer to input matrix (M x K, row-major)
 * @param weight_KN Pointer to weight matrix (K x N, row-major)
 * @param bias_N Pointer to bias vector (N elements)
 * @param residual_MN Pointer to residual matrix (M x N, row-major)
 * @param output_MN Pointer to output matrix (M x N, row-major, must be pre-allocated)
 * @param M Batch size (number of rows)
 * @param N Number of output features
 * @param K Number of input features
 */
static inline void dense_bias_residual_fp32(
    const float* input_MK, const float* weight_KN, const float* bias_N,
    const float* residual_MN, float* output_MN, long M, long N, long K
) {
    ane::kernel::dense_bias_residual_fp32(
        input_MK, weight_KN, bias_N, residual_MN, output_MN, M, N, K
    );
}
/** --------------------------------------------------------------------------------------------------------- Fused Dense + LayerNorm
 * @brief Fused dense (matvec) + layer normalization: temp = W @ x, then layernorm(temp, gamma, beta, eps)
 * Avoids materializing intermediate matmul result to memory.
 * @param W Pointer to weight matrix (m x n, row-major)
 * @param m Number of output dimensions (rows in W)
 * @param n Number of input dimensions (columns in W)
 * @param x Pointer to input vector (n elements)
 * @param gamma Pointer to scale vector (m elements)
 * @param beta Pointer to shift vector (m elements)
 * @param eps Epsilon for numerical stability
 * @param out Pointer to output vector (m elements, must be pre-allocated)
 */
static inline void fused_dense_layernorm_fp32(
    const float* W, int m, int n, const float* x, const float* gamma,
    const float* beta, float eps, float* out
) {
    ane::kernel::fused_dense_layernorm_fp32(W, m, n, x, gamma, beta, eps, out);
}
/** --------------------------------------------------------------------------------------------------------- Fused Dense + Leaky ReLU
 * @brief Fused dense (matvec) + bias + leaky ReLU activation
 * out[i] = temp[i] >= 0 ? temp[i] : alpha * temp[i], where temp = W @ x + bias
 * @param W Pointer to weight matrix (m x n, row-major)
 * @param m Number of output dimensions
 * @param n Number of input dimensions
 * @param x Pointer to input vector (n elements)
 * @param bias Pointer to bias vector (m elements)
 * @param alpha Leak coefficient (typically 0.01 or 0.2)
 * @param out Pointer to output vector (m elements, must be pre-allocated)
 */
static inline void fused_dense_leaky_relu_fp32(
    const float* W, int m, int n, const float* x, const float* bias, float alpha, float* out
) {
    ane::kernel::fused_dense_leaky_relu_fp32(W, m, n, x, bias, alpha, out);
}
/** --------------------------------------------------------------------------------------------------------- Fused Dense + Mish
 * @brief Fused dense (matvec) + bias + Mish activation
 * mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
 * Uses identity: mish(x) = x * e*(e+2) / (e^2 + 2*e + 2) where e = exp(x)
 * @param W Pointer to weight matrix (m x n, row-major)
 * @param m Number of output dimensions
 * @param n Number of input dimensions
 * @param x Pointer to input vector (n elements)
 * @param bias Pointer to bias vector (m elements)
 * @param out Pointer to output vector (m elements, must be pre-allocated)
 */
static inline void fused_dense_mish_fp32(
    const float* W, int m, int n, const float* x, const float* bias, float* out
) {
    ane::kernel::fused_dense_mish_fp32(W, m, n, x, bias, out);
}
/** --------------------------------------------------------------------------------------------------------- Fused Dense + ELU
 * @brief Fused dense (matvec) + bias + ELU (Exponential Linear Unit) activation
 * ELU(x) = x >= 0 ? x : alpha * (exp(x) - 1)
 * @param W Pointer to weight matrix (m x n, row-major)
 * @param m Number of output dimensions
 * @param n Number of input dimensions
 * @param x Pointer to input vector (n elements)
 * @param bias Pointer to bias vector (m elements)
 * @param alpha ELU alpha parameter (typically 1.0)
 * @param out Pointer to output vector (m elements, must be pre-allocated)
 */
static inline void fused_dense_elu_fp32(
    const float* W, int m, int n, const float* x, const float* bias, float alpha, float* out
) {
    ane::kernel::fused_dense_elu_fp32(W, m, n, x, bias, alpha, out);
}
/** --------------------------------------------------------------------------------------------------------- Fused BatchNorm + ReLU
 * @brief Fused batch normalization + ReLU: out = max(0, gamma * (x - mean) * inv_std + beta)
 * Pre-computed mean and inv_std (1/sqrt(var+eps)) must be provided by caller.
 * @param x Pointer to input vector (n elements)
 * @param n Number of elements
 * @param mean Pre-computed mean
 * @param inv_std Pre-computed 1/sqrt(var + eps)
 * @param gamma Pointer to scale vector (n elements)
 * @param beta Pointer to shift vector (n elements)
 * @param out Pointer to output vector (n elements, must be pre-allocated)
 */
static inline void fused_batchnorm_relu_fp32(
    const float* x, long n, float mean, float inv_std, const float* gamma, const float* beta, float* out
) {
    ane::kernel::fused_batchnorm_relu_fp32(x, n, mean, inv_std, gamma, beta, out);
}
/** --------------------------------------------------------------------------------------------------------- Fused GLU (Gated Linear Unit)
 * @brief Fused GLU with sigmoid gate: out[i] = x[i] * sigmoid(x[i])
 * This is the single-input GLU variant. For two-input GLU (out = up * sigmoid(gate)),
 * use fused_silu_gate_mul_fp32.
 * @param x Pointer to input vector (n elements)
 * @param n Number of elements
 * @param out Pointer to output vector (n elements, must be pre-allocated)
 */
static inline void fused_glu_fp32(const float* x, long n, float* out) {
    ane::kernel::fused_glu_fp32(x, n, out);
}
/** --------------------------------------------------------------------------------------------------------- Argmax FP32
 * @brief Find index of maximum element in an FP32 vector
 * @param input Pointer to the input vector
 * @param n Number of elements
 * @return Index of the maximum element (first occurrence)
 */
static inline int32_t argmax_fp32(const float* input, long n) {
    return ane::kernel::argmax_fp32(input, n);
}
/** --------------------------------------------------------------------------------------------------------- SGD FP32
 * @brief SGD parameter update: output = params - scale * sum(gradient rows)
 * @param params Pointer to parameter vector (n elements)
 * @param gradients Pointer to contiguous gradient buffer (batch_size rows of n floats)
 * @param output Pointer to output vector (n elements, must be pre-allocated)
 * @param n Number of parameters
 * @param scale Learning rate scale factor
 * @param batch_size Number of gradient rows
 */
static inline void sgd_fp32(
    const float* params, const float* gradients, float* output, long n, float scale, long batch_size
) {
    ane::kernel::sgd_fp32(params, gradients, output, n, scale, batch_size);
}
/** --------------------------------------------------------------------------------------------------------- Param Update FP32
 * @brief SGD parameter update with multiple gradient sources: output = params - lr*inv * sum(gradients)
 * @param params Pointer to parameter vector (n elements)
 * @param gradients Array of pointers to gradient vectors (count pointers, each pointing to n floats)
 * @param output Pointer to output vector (n elements, must be pre-allocated)
 * @param n Number of parameters
 * @param learning_rate Learning rate
 * @param inv_updates Inverse of update count (1/count)
 * @param count Number of gradient sources
 */
static inline void param_update_fp32(
    const float* params, const float* const gradients[], float* output,
    long n, float learning_rate, float inv_updates, int count
) {
    ane::kernel::param_update_fp32(params, gradients, output, n, learning_rate, inv_updates, count);
}
/** --------------------------------------------------------------------------------------------------------- Conv2D Backward (Combined)
 * @brief Combined conv2d backward pass: computes both input and weight gradients
 * @param input_NHWC Input tensor [N, H, W, C_in]
 * @param weight_HWIO Weight tensor [KH, KW, C_in, C_out]
 * @param grad_out_NHWC Gradient of output [N, H_out, W_out, C_out]
 * @param grad_in_NHWC Output: gradient of input [N, H, W, C_in]
 * @param grad_weight_HWIO Output: gradient of weights [KH, KW, C_in, C_out]
 * @param grad_bias Output: gradient of bias [C_out] (can be nullptr)
 */
static inline void conv2d_backward(
    const float* input_NHWC, const float* weight_HWIO, const float* grad_out_NHWC,
    float* grad_in_NHWC, float* grad_weight_HWIO, float* grad_bias,
    long N, long H, long W, long C_in, long KH, long KW,
    long C_out, long stride, long pad
) {
    ane::kernel::conv2d_backward_input_fp32(
        grad_out_NHWC, weight_HWIO, grad_in_NHWC,
        N, H, W, C_in, KH, KW, C_out, stride, pad
    );
    ane::kernel::conv2d_backward_weight_fp32(
        input_NHWC, grad_out_NHWC, grad_weight_HWIO, grad_bias,
        N, H, W, C_in, KH, KW, C_out, stride, pad
    );
}
/** --------------------------------------------------------------------------------------------------------- SDP Attention Decode (Cached)
 * @brief Single-query decode attention against a KV cache
 * @param q Query vector [num_heads * head_dim]
 * @param k_cache Key cache [cache_len * num_heads * head_dim]
 * @param v_cache Value cache [cache_len * num_heads * head_dim]
 * @param output Output vector [num_heads * head_dim]
 * @param cache_len Number of cached key-value pairs
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param scale Attention scale factor (typically 1/sqrt(head_dim))
 */
static inline void sdp_attn_decode_cached_fp32(
    const float* q, const float* k_cache, const float* v_cache,
    float* output, long cache_len, long num_heads, long head_dim, float scale
) {
    ane::kernel::sdp_attn_decode_cached_fp32(
        q, k_cache, v_cache, output, cache_len, num_heads, head_dim, scale
    );
}
/** --------------------------------------------------------------------------------------------------------- KV Cache Append
 * @brief Append new key-value pair to the KV cache at a given position
 * @param k_cache Key cache to append to
 * @param v_cache Value cache to append to
 * @param new_k New key vector to append
 * @param new_v New value vector to append
 * @param pos Position index in the cache to write to
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 */
static inline void kv_cache_append_fp32(
    float* k_cache, float* v_cache, const float* new_k, const float* new_v,
    long pos, long num_heads, long head_dim
) {
    ane::kernel::kv_cache_append_fp32(k_cache, v_cache, new_k, new_v, pos, num_heads, head_dim);
}
/** --------------------------------------------------------------------------------------------------------- SDP Attention Backward
 * @brief Backward pass for scaled dot-product attention
 * @param dO Gradient of output
 * @param q Query tensor
 * @param k Key tensor
 * @param v Value tensor
 * @param attn_weights Forward attention weights
 * @param dQ Output: gradient of queries
 * @param dK Output: gradient of keys
 * @param dV Output: gradient of values
 * @param seq_len Sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param scale Attention scale factor
 */
static inline void sdp_attn_backward_fp32(
    const float* dO, const float* q, const float* k, const float* v,
    const float* attn_weights, float* dQ, float* dK, float* dV,
    long seq_len, long num_heads, long head_dim, float scale
) {
    ane::kernel::sdp_attn_backward_fp32(
        dO, q, k, v, attn_weights, dQ, dK, dV, seq_len, num_heads, head_dim, scale
    );
}
/** --------------------------------------------------------------------------------------------------------- GQA Attention Decode (Cached)
 * @brief Grouped-query attention decode against a KV cache
 * @param q Query vector [n_heads * head_dim]
 * @param k_cache Key cache [cache_len * n_kv_heads * head_dim]
 * @param v_cache Value cache [cache_len * n_kv_heads * head_dim]
 * @param output Output vector [n_heads * head_dim]
 * @param cache_len Number of cached key-value pairs
 * @param n_heads Number of query heads
 * @param n_kv_heads Number of key-value heads
 * @param head_dim Dimension per head
 * @param scale Attention scale factor
 */
static inline void gqa_attn_decode_cached_fp32(
    const float* q, const float* k_cache, const float* v_cache,
    float* output, long cache_len, long n_heads, long n_kv_heads,
    long head_dim, float scale
) {
    ane::kernel::gqa_attn_decode_cached_fp32(
        q, k_cache, v_cache, output, cache_len, n_heads, n_kv_heads, head_dim, scale
    );
}
/** --------------------------------------------------------------------------------------------------------- Cross Attention Prefill
 * @brief Cross-attention prefill (encoder-decoder attention)
 * @param q Query tensor [q_len * num_heads * head_dim]
 * @param k Key tensor [kv_len * num_heads * head_dim]
 * @param v Value tensor [kv_len * num_heads * head_dim]
 * @param output Output tensor [q_len * num_heads * head_dim]
 * @param q_len Query sequence length
 * @param kv_len Key-value sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param scale Attention scale factor
 */
static inline void cross_attn_prefill_fp32(
    const float* q, const float* k, const float* v, float* output,
    long q_len, long kv_len, long num_heads, long head_dim, float scale
) {
    ane::kernel::cross_attn_prefill_fp32(
        q, k, v, output, q_len, kv_len, num_heads, head_dim, scale
    );
}
/** --------------------------------------------------------------------------------------------------------- Cross Attention Decode (Cached)
 * @brief Cross-attention decode against a KV cache
 * @param q Query vector [num_heads * head_dim]
 * @param k Key cache [kv_len * num_heads * head_dim]
 * @param v Value cache [kv_len * num_heads * head_dim]
 * @param output Output vector [num_heads * head_dim]
 * @param kv_len Key-value sequence length
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param scale Attention scale factor
 */
static inline void cross_attn_decode_cached_fp32(
    const float* q, const float* k, const float* v, float* output,
    long kv_len, long num_heads, long head_dim, float scale
) {
    ane::kernel::cross_attn_decode_cached_fp32(
        q, k, v, output, kv_len, num_heads, head_dim, scale
    );
}
/** --------------------------------------------------------------------------------------------------------- Flash Attention Decode (Cached)
 * @brief Flash attention decode against a KV cache (memory-efficient)
 * @param q Query vector [num_heads * head_dim]
 * @param k_cache Key cache [cache_len * num_heads * head_dim]
 * @param v_cache Value cache [cache_len * num_heads * head_dim]
 * @param output Output vector [num_heads * head_dim]
 * @param cache_len Number of cached key-value pairs
 * @param num_heads Number of attention heads
 * @param head_dim Dimension per head
 * @param scale Attention scale factor
 */
static inline void flash_attn_decode_cached_fp32(
    const float* q, const float* k_cache, const float* v_cache,
    float* output, long cache_len, long num_heads, long head_dim, float scale
) {
    ane::kernel::flash_attn_decode_cached_fp32(
        q, k_cache, v_cache, output, cache_len, num_heads, head_dim, scale
    );
}
/** --------------------------------------------------------------------------------------------------------- Log1p FP32
 * @brief Computes elementwise log(1 + x) with improved numerical accuracy for small x
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements
 */
static inline void log1p_fp32(const float* input, float* output, long n) {
    ane::kernel::log1p_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Sign FP32
 * @brief Computes elementwise sign: -1.0 for negative, 0.0 for zero, +1.0 for positive
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements
 */
static inline void sign_fp32(const float* input, float* output, long n) {
    ane::kernel::sign_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Hard Sigmoid FP32
 * @brief Computes elementwise hard sigmoid: clamp(x/6 + 0.5, 0, 1)
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param n The number of elements
 */
static inline void hard_sigmoid_fp32(const float* input, float* output, long n) {
    ane::kernel::hard_sigmoid_fp32(input, output, n);
}
/** --------------------------------------------------------------------------------------------------------- Dropout FP32
 * @brief Applies dropout: output[i] = input[i] * mask[i] * inv_keep_prob
 * @param input Pointer to the input vector
 * @param mask Pointer to the binary mask vector (0.0 or 1.0)
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param inv_keep_prob Reciprocal of the keep probability (1.0 / keep_prob)
 * @param n The number of elements
 */
static inline void dropout_fp32(
    const float* input, const float* mask, float* output, float inv_keep_prob, long n
) {
    ane::kernel::dropout_fp32(input, mask, output, inv_keep_prob, n);
}
/** --------------------------------------------------------------------------------------------------------- Gaussian Noise FP32
 * @brief Adds scaled Gaussian noise: output[i] = input[i] + noise[i] * stddev
 * @param input Pointer to the input vector
 * @param output Pointer to the output vector (must be pre-allocated)
 * @param stddev Standard deviation of the Gaussian noise
 * @param n The number of elements
 * @param seed 64-bit seed for the deterministic hash-based noise generator
 */
static inline void gaussian_noise_fp32(
    const float* input, float* output, float stddev, long n, uint64_t seed
) {
    ane::kernel::gaussian_noise_fp32(input, output, stddev, n, seed);
}
} // namespace ane