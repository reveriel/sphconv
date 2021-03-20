pybind11_add_module(sphconv_utils all.cpp)
# target_link_libraries(sphconv_utils pybind11)
target_include_directories(sphconv_utils PRIVATE ${PROJECT_SOURCE_DIR}/include/)
