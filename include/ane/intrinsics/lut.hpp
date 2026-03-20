#pragma once
/** --------------------------------------------------------------------------------------------------------- Apple Neural Engine
 * @file lut.hpp
 * @brief Header file for the lookup_table struct, a utility for working with the LUTI4
 * (luti2 is not supported by M4)
 */
#if defined(__aarch64__) && defined(__APPLE__)
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <initializer_list>
#include <ane/extern/asm.hpp>
#include <simd/simd.h>

namespace ane {
/** --------------------------------------------------------------------------------------------------------- lookup_table
 * @struct lookup_table
 * @brief A helper utility struct.
 * This is a simple wrapper around a 64-entry lookup table, a utility struct to make the
 * LUTI2 and LUTI4 instructions easier to work with.
 */
struct alignas(64) lookup_table {
    /// @brief The raw data of the lookup table, aligned to 64 bytes for optimal access (32-bits per entry)
    simd_uchar64 data;
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
} // namespace ane
#endif