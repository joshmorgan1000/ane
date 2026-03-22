cmake_minimum_required(VERSION 3.19)
project(ane LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# ==============================================================================
# Apple platform libraries
# ==============================================================================
find_library(FOUNDATION Foundation REQUIRED)
find_library(METAL Metal)
find_library(MPS MetalPerformanceShaders)
find_library(COREML CoreML)
find_library(NETWORK Network)
find_library(ACCELERATE Accelerate)
find_library(COMPRESSION compression)
find_library(CRYPTOKIT CryptoKit)
set(PLATFORM_LIBS ${FOUNDATION})
if(METAL)
    list(APPEND PLATFORM_LIBS ${METAL})
endif()
if(MPS)
    list(APPEND PLATFORM_LIBS ${MPS})
endif()
if(COREML)
    list(APPEND PLATFORM_LIBS ${COREML})
endif()
if(NETWORK)
    list(APPEND PLATFORM_LIBS ${NETWORK})
endif()
if(ACCELERATE)
    list(APPEND PLATFORM_LIBS ${ACCELERATE})
endif()
if(COMPRESSION)
    list(APPEND PLATFORM_LIBS ${COMPRESSION})
endif()
# ==============================================================================
# Compiler Flags
# ==============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-finite-math-only>
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
        $<$<COMPILE_LANGUAGE:CXX>:-ftree-vectorize>
    )
    # Silence specific warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-Wno-nan-infinity-disabled" HAS_NO_NAN_INFINITY_DISABLED)
        if(HAS_NO_NAN_INFINITY_DISABLED)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-nan-infinity-disabled>)
        endif()
        # Silence anonymous struct warnings from Metal headers
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-gnu-anonymous-struct>)
        # Silence VLA warnings (used in some math kernels)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-vla-cxx-extension>)
    endif()
    # Build type specific flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-flto>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>)
        add_link_options(-flto -march=native)
        add_compile_definitions(NDEBUG)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O0> $<$<COMPILE_LANGUAGE:CXX>:-g>)
        # ARM64 requires -march=native for bfloat16 support via arm_bf16.h
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        endif()
        # Enable Cereal lifetime tracking in debug builds
        # add_compile_definitions(CEREAL_TRACK_LIFETIME)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /fp:fast /Oi)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /GL)
        add_link_options(/LTCG)
    else()
        add_compile_options(/Od /Zi)
    endif()
endif()
# ==============================================================================
# Platform capability detection
# ==============================================================================
message(STATUS "ane: Detecting platform capabilities...")
if(APPLE)
    set(ASM_SYM_PREFIX "_")
else()
    set(ASM_SYM_PREFIX "")
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
    message(FATAL_ERROR "ane requires ARM64. Detected: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
# Probe SME2 by compiling and executing smstart
file(WRITE ${CMAKE_BINARY_DIR}/try_sme2.s
    ".text\n"
    ".global ${ASM_SYM_PREFIX}sme2_test\n"
    ".p2align 4\n"
    "${ASM_SYM_PREFIX}sme2_test:\n"
    "    smstart sm\n"
    "    ptrue p0.b\n"
    "    smstop sm\n"
    "    mov w0, #0\n"
    "    ret\n"
)
file(WRITE ${CMAKE_BINARY_DIR}/try_sme2_main.c
    "extern int sme2_test(void);\n"
    "int main(void) { return sme2_test(); }\n"
)
try_compile(SME2_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}/try_sme2_build
    SOURCES ${CMAKE_BINARY_DIR}/try_sme2.s ${CMAKE_BINARY_DIR}/try_sme2_main.c
    COMPILE_DEFINITIONS -march=armv9-a+sme2+sve2
    OUTPUT_VARIABLE SME2_COMPILE_OUTPUT
)
set(ANE_HAS_SME2 0)
if(SME2_COMPILE_RESULT)
    try_run(SME2_RUN_RESULT SME2_RUN_COMPILE_RESULT
        ${CMAKE_BINARY_DIR}/try_sme2_run
        SOURCES ${CMAKE_BINARY_DIR}/try_sme2.s ${CMAKE_BINARY_DIR}/try_sme2_main.c
        COMPILE_DEFINITIONS -march=armv9-a+sme2+sve2
    )
    if(SME2_RUN_RESULT EQUAL 0)
        set(ANE_HAS_SME2 1)
        message(STATUS "  SME2: AVAILABLE")
    else()
        message(STATUS "  SME2: NOT AVAILABLE (execution failed)")
    endif()
else()
    message(STATUS "  SME2: NOT AVAILABLE (compilation failed)")
endif()

if(NOT ANE_HAS_SME2)
    message(FATAL_ERROR
        "ane requires ARM SME2 (Apple M4 or compatible).\n"
        "SME2 instruction probe failed on this system."
    )
endif()
# ── Header-only library (inline asm in ane.hpp) ──
add_library(ane INTERFACE)
target_include_directories(ane INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# ── Install rules ──
include(GNUInstallDirs)
install(TARGETS ane)
install(DIRECTORY include/ane
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# ── Tests ──
add_subdirectory(tests)
enable_testing()
message(STATUS "======================================================")
message(STATUS "______________________________________________________")
message(STATUS "     __     _____ ______  _______ __   _ _______")
message(STATUS "     |        |   |_____] |_____| | \  | |______")
message(STATUS "     |_____ __|__ |_____] |     | |  \_| |______")
message(STATUS "──────────────────────────────────────────────────────")
message(STATUS "     Bare-Metal Access to Apple's Neural Engine")
message(STATUS "======================================================")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(METAL_ENABLED)
message(STATUS "  Metal Framework:   ${METAL_ENABLED}")
endif()
message(STATUS "===================================================")