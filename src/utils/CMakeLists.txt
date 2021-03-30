pybind11_add_module(sphconv_utils all.cpp)
target_include_directories(sphconv_utils PRIVATE ${PROJECT_SOURCE_DIR}/include/)
