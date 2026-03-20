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
#if defined(__aarch64__) && defined(__APPLE__)
#include "extern/trampoline.hpp"
#include "extern/asm.hpp"
#include "intrinsics/lut.hpp"
#endif // defined(__aarch64__) && defined(__APPLE__)
