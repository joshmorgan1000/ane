# tests/CMakeLists.txt
file(GLOB TEST_SOURCES test_*.cpp)
file(GLOB BENCH_SOURCES bench_*.cpp)

foreach(src ${TEST_SOURCES} ${BENCH_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_compile_options(${name} PRIVATE -O0)
    target_link_libraries(${name} ane)
endforeach()
