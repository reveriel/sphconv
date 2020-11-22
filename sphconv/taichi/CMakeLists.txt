
cmake_minimum_required(VERSION 3.8)

set(CMAKE_BUILD_TYPE Debug)

project(ttt)


set(TAICHI_DIR "../../third_party/taichi/")
set(TAICHI_CMAKE_DIR "${TAICHI_DIR}/cmake/")

include(../../third_party/taichi/cmake/TaichiCXXFlags.cmake)

include_directories(/usr/local/cuda/include)
include_directories(../../third_party/taichi/include)
include_directories(../../third_party/taichi/lang/include/)
include_directories(../../third_party/taichi/external/include)
link_directories(../../third_party/taichi/build)

message("C++ Flags: ${CMAKE_CXX_FLAGS}")

add_executable(program)
target_sources(program PUBLIC program.cpp)
target_link_libraries(program taichi_core taichi_lang_core dl)

add_executable(test)
target_sources(test PUBLIC test.cpp)
target_link_libraries(test taichi_core taichi_lang_core dl)

add_executable(test_array)
target_sources(test_array PUBLIC test_array.cpp)
target_link_libraries(test_array taichi_core taichi_lang_core dl)