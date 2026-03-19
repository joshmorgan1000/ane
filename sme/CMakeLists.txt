cmake_minimum_required(VERSION 3.19)
project(kernels LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ── Platform detection: require ARM64 with SME2 ──
message(STATUS "kernels: Detecting platform capabilities...")
set(KERNELS_HAVE_AVX512 0)
set(KERNELS_HAVE_AVX2 0)
set(KERNELS_HAVE_SME2 0)
set(KERNELS_HAVE_NEON 0)
set(KERNELS_HAVE_XDNA 0)
if(APPLE)
    set(ASM_SYM_PREFIX "_")
else()
    set(ASM_SYM_PREFIX "")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
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
    set(KERNEL_HAS_SME2 0)
    if(SME2_COMPILE_RESULT)
        try_run(SME2_RUN_RESULT SME2_RUN_COMPILE_RESULT
            ${CMAKE_BINARY_DIR}/try_sme2_run
            SOURCES ${CMAKE_BINARY_DIR}/try_sme2.s ${CMAKE_BINARY_DIR}/try_sme2_main.c
            COMPILE_DEFINITIONS -march=armv9-a+sme2+sve2
        )
        if(SME2_RUN_RESULT EQUAL 0)
            set(KERNEL_HAS_SME2 1)
            message(STATUS "  SME2: AVAILABLE")
            set(KERNELS_HAVE_SME2 1)
        else()
            # Try SVE2 as a fallback for older Apple Silicon without SME2 support
            file(WRITE ${CMAKE_BINARY_DIR}/try_sve2.s
                ".text\n"
                ".global ${ASM_SYM_PREFIX}sve2_test\n"
                ".p2align 4\n"
                "${ASM_SYM_PREFIX}sve2_test:\n"
                "    ptrue p0.b\n"
                "    mov w0, #0\n"
                "    ret\n"
            )
            try_compile(SVE2_COMPILE_RESULT
                ${CMAKE_BINARY_DIR}/try_sve2_build
                SOURCES ${CMAKE_BINARY_DIR}/try_sve2.s ${CMAKE_BINARY_DIR}/try_sme2_main.c
                COMPILE_DEFINITIONS -march=armv9-a+sve2
                OUTPUT_VARIABLE SVE2_COMPILE_OUTPUT
            )
            if(SVE2_COMPILE_RESULT)
                message(STATUS "  SVE2: AVAILABLE")
                set(KERNELS_HAVE_SVE2 1)
            else()
                # Try NEON as a last resort for very old Apple Silicon models
                file(WRITE ${CMAKE_BINARY_DIR}/try_neon.s
                    ".text\n"
                    ".global ${ASM_SYM_PREFIX}neon_test\n"
                    ".p2align 4\n"
                    "${ASM_SYM_PREFIX}neon_test:\n"
                    "    mov v0.16b, #0\n"
                    "    mov w0, #0\n"
                    "    ret\n"
                )
                try_compile(NEON_COMPILE_RESULT
                    ${CMAKE_BINARY_DIR}/try_neon_build
                    SOURCES ${CMAKE_BINARY_DIR}/try_neon.s ${CMAKE_BINARY_DIR}/try_sme2_main.c
                    COMPILE_DEFINITIONS -march=armv8-a+neon
                    OUTPUT_VARIABLE NEON_COMPILE_OUTPUT
                )
                if(NEON_COMPILE_RESULT)
                    message(STATUS "  NEON: AVAILABLE")
                    set(KERNELS_HAVE_NEON 1)
                else()
                    message(FATAL_ERROR "kernels require SME2, SVE2, or NEON for ARM. Detected: ${CMAKE_SYSTEM_PROCESSOR}")
                endif()
            endif()
        endif()
    endif()
else()
    # Check for AVX512 etc...
endif()
# Build library based on detected capabilities
if(KERNELS_HAVE_SME2)
    message(STATUS "Platform supports ARM SME2. Building SME2 kernels.")
    # ── Assembly kernels - list all the files in the subdirectories of src/sme2/ ──
    file(GLOB_RECURSE ASM_SOURCES src/sme2/**/*.s)
    set_source_files_properties(${ASM_SOURCES}
        PROPERTIES COMPILE_FLAGS "-march=armv9-a+sme2+sve2+sme-lutv2"
    )
    # ── Static library ──
    add_library(kernels STATIC ${ASM_SOURCES})
    target_include_directories(kernels PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    # ── Install rules ──
    include(GNUInstallDirs)
    install(TARGETS kernels
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(DIRECTORY include/kernels
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    # ── Tests ──
    add_subdirectory(tests)
    enable_testing()
else()
    message(FATAL_ERROR "Only SME2 is supported at this time")
endif()