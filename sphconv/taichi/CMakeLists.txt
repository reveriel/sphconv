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

add_subdirectory(mp11)
link_libraries(Boost::mp11)

# set(CMAKE_CXX_COMPILER  clang++)

add_library(utils utils.cpp)
link_libraries(utils)

add_executable(program)
target_sources(program PUBLIC program.cpp)
target_link_libraries(program taichi_core taichi_lang_core dl ${Boost_LIBRARIES} utils)


add_executable(mp)
target_sources(mp PUBLIC mp.cpp)
target_link_libraries(mp  ${Boost_LIBRARIES})

add_executable(test)
target_sources(test PUBLIC test.cpp)
target_link_libraries(test taichi_core taichi_lang_core dl ${Boost_LIBRARIES})

add_executable(test_array)
target_sources(test_array PUBLIC test_array.cpp)
target_link_libraries(test_array taichi_core taichi_lang_core dl ${Boost_LIBRARIES})

pybind11_add_module(conv_taichi MODULE
    export.cpp
)

target_link_libraries(conv_taichi
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE ${TORCH_PYTHON_LIBRARY}
)
