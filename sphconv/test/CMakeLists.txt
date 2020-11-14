
cmake_minimum_required(VERSION 3.5)
project(test CXX CUDA)

find_package(Catch2 REQUIRED)

include_directories("../include")
include_directories("../cutlass/include")

add_executable(threadmap threadmap.cu)
target_include_directories(threadmap INTERFACE ${CATCH_INCLUDE_DIR})
target_link_libraries(threadmap Catch2::Catch2WithMain)

# add_executable(threadmap_test threadmap_test.cu)