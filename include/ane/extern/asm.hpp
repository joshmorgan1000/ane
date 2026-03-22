#pragma once
#if defined(__APPLE__)
#include <cstdint>
#include <cstring>
#if defined(__has_include) && __has_include(<arm_neon.h>)
#include <arm_neon.h>
#else
struct alignas(2) bfloat16_t {
    uint16_t value;
    bfloat16_t() : value(0) {}
    explicit bfloat16_t(float f) {
        value = static_cast<uint16_t>(reinterpret_cast<uint32_t&>(f) >> 16);
    }
    operator float() const {
        uint32_t temp = static_cast<uint32_t>(value) << 16;
        return reinterpret_cast<float&>(temp);
    }
};
#endif

namespace ane {
namespace kernel {
extern "C" {
void tile_execute(const uint8_t* bytecode, const void** input_streams, void** output_streams);
void stream_exec(const void* compiled_program);
} // extern "C"
} // namespace kernel
} // namespace ane
#endif  // defined(__APPLE__)