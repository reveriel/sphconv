

add_library(sphconv
    src/all.cpp
    src/indice_conv.cu
    src/indice.cu
    src/to_dense.cu
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


target_include_directories(sphconv PUBLIC
    include/
)

find_package(Catch2 REQUIRED)
Add_executable(feature
    test/feature.cpp
    )

Target_link_libraries(feature PRIVATE Catch2::Catch2 sphconv)

