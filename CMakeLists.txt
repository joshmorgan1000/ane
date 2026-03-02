cmake_minimum_required(VERSION 3.19)
project(ane LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Platform detection: require ARM64 with SME2 ──
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

# ── Assembly kernels ──
file(GLOB ASM_SOURCES src/ane/asm/*.s)

set_source_files_properties(${ASM_SOURCES}
    PROPERTIES COMPILE_FLAGS "-march=armv9-a+sme2+sve2+sme-lutv2"
)

# ── Static library ──
add_library(ane STATIC ${ASM_SOURCES})
target_include_directories(ane PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Install rules ──
include(GNUInstallDirs)
install(TARGETS ane
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ane
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ── Tests ──
add_subdirectory(tests)
enable_testing()
