#pragma once
/** --------------------------------------------------------------------------------------------------------- z_tiles
 * @file z_tiles.hpp
 * @brief Defines the z_tiles struct, which represents the 4 ZA tiles (za0.s-za3.s) as a single 4096-byte
 * aligned struct for easy capture of ZA state via store_tiles.
 * 
 *  - Each tile is 16 rows × 16 int32 columns on M4/M5 (GROUP_DIM=32, SVL=16), so total is 64 z-vector
 *    rows × 64 bytes = 4096 bytes.
 *  - Provides helper methods to access the tile data as int32 or float arrays, and to get the raw pointer
 *    for passing to store_tiles or load_bias operands.
 * 
 * @author Josh Morgan (@joshmorgan1000 on GitHub)
 * Released under the MIT License
 */
#include <ane/tiles/concepts.hpp>
#include <ane/tiles/z_vector.hpp>
#include <ane/tiles/z_stream.hpp>
#include <array>
#include <cstdint>
#include <cstring>

namespace ane {
/**
 * @brief Structs representing the ZA tile array state for different data types. Each struct is 4096 bytes
 * aligned to match the size of the full ZA tile state.
 */
struct alignas(4096) float32x16x16_t {
    float data[16][16][4];
};
struct alignas(4096) int32x16x16_t {
    int32_t data[16][16][4];
};
struct alignas(4096) uint32x16x16_t {
    uint32_t data[16][16][4];
};
struct alignas(4096) bfloat16x32x32_t {
    bfloat16_t data[32][32][2];
};
struct alignas(4096) uint16x32x32_t {
    uint16_t data[32][32][2];
};
struct alignas(4096) int16x32x32_t {
    int16_t data[32][32][2];
};
struct alignas(4096) int8x64x64_t {
    int8_t data[64][64];
};
struct alignas(4096) uint8x64x64_t {
    uint8_t data[64][64];
};
/** --------------------------------------------------------------------------------------------------------- z_tiles
 * @struct z_tiles
 * @brief A 4096-byte aligned struct that maps directly to the ZA tile array state.
 *  - Use with store_tiles to capture ZA state at the end of a program.
 *  - 4 tiles (za0.s-za3.s), each 16 rows × 16 int32 columns on M4/M5 (SVLs=16).
 *  - Total: 64 z-vector rows × 64 bytes = 4096 bytes.
 */
struct alignas(4096) z_tiles {
    uint8_t data[4096];  ///< Raw ZA tile bytes in store_tiles layout
    /** --------------------------------------------------------------------------------------------- As Int32
     * @brief Access the tile data as a flat array of int32 values.
     * @return Pointer to the first int32 element (1024 elements total)
     */
    int32_t* as_i32() { return reinterpret_cast<int32_t*>(data); }
    const int32_t* as_i32() const { return reinterpret_cast<const int32_t*>(data); }
    /** --------------------------------------------------------------------------------------------- As Float
     * @brief Access the tile data as a flat array of float32 values.
     * @return Pointer to the first float element (1024 elements total)
     */
    float* as_f32() { return reinterpret_cast<float*>(data); }
    const float* as_f32() const { return reinterpret_cast<const float*>(data); }
    /** --------------------------------------------------------------------------------------------- Pointer
     * @brief Get the raw pointer for passing to store_tiles or load_bias operands.
     */
    uint8_t* ptr() { return data; }
    const uint8_t* ptr() const { return data; }
};
} // namespace ane