GTEST_DIR=../cutlass/build/_deps/googletest-src/googletest
USER_DIR=.
CPPFLAGS += -isystem $(GTEST_DIR)/include

DEBUG=1

CXX := g++

## all unit test executables
TESTS = threadmap iterator

## environment variables, save to a file to speedup the make proccess
MAKE_CACHE = cache.make

# CUDA ROOT DIR that contains bin/ lib64/ and include/
CUDA_DIR := /usr/local/cuda

$(MAKE_CACHE):
	echo "" > $@
	echo "PYTHON_HEADER_DIR := $(shell python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')" >> $@
	echo "PYTORCH_INCLUDES := $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')" >> $@
	echo "PYTORCH_LIBRARIES := $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')" >> $@
	# Assume pytorch > v1.1
	echo "WITH_ABI := $(shell python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')" >> $@

	echo "NVCCFLAGS += $(shell python -c 'from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; [print(p) for p in COMMON_NVCC_FLAGS]')" >> $@

include $(MAKE_CACHE)

all: $(TESTS)

clean :
	rm -f $(TESTS) gtest.a gtest_main.a *.o $(MAKE_CACHE)

# PYTHON Header path
# PYTHON_HEADER_DIR := $(shell python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')
# PYTORCH_INCLUDES := $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
# PYTORCH_LIBRARIES := $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')


INCLUDE_DIRS := ./ $(CUDA_DIR)/include

INCLUDE_DIRS += $(PYTHON_HEADER_DIR)
INCLUDE_DIRS += $(PYTORCH_INCLUDES)
INCLUDE_DIRS += ../../sphconv/include
INCLUDE_DIRS += ../../sphconv/cutlass/include

# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

###############################################################################

###############################################################################
# SRC_DIR := ./sphconv/src
# OBJ_DIR := ./sphconv/objs
# CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)
# OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
# CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(CU_SRCS))
# STATIC_LIB := $(OBJ_DIR)/libmake_pytorch.a

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_75,code=sm_75
		# -gencode arch=compute_60,code=sm_60 \
		# -gencode arch=compute_61,code=sm_61 \
		# -gencode arch=compute_30,code=sm_30 \
		# -gencode arch=compute_35,code=sm_35 \
		# -gencode arch=compute_50,code=sm_50 \
		# -gencode arch=compute_52,code=sm_52 \
		# -gencode arch=compute_61,code=compute_61 \
		# -gencode arch=compute_70,code=sm_70 \
		# -gencode arch=compute_75,code=sm_75

# We will also explicitly add stdc++ to the link target.
LIBRARIES += stdc++ cudart c10 torch torch_python
# caffe2_gpu  caff2

# from torch.utils.cpp_extension import library_paths give empty results
# use absolute path
# this is bad
LDFLAGS += -L/home/user/anaconda3/envs/sphconv/lib/python3.7/site-packages/torch/lib
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))
# $(foreach librarydir,$(PYTORCH_LIBRARIES),-L$(librarydir)) $(PKG_CONFIG) \

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	# https://gcoe-dresden.de/reaching-the-shore-with-a-fog-warning-my-eurohack-day-4-morning-session/
	NVCCFLAGS += -g -G # -rdc true
else
	COMMON_FLAGS += -DNDEBUG -O3
endif

WARNINGS := -Wall -Wno-sign-compare -Wcomment

INCLUDE_DIRS += $(BLAS_INCLUDE)

# Automatic dependency generation (nvcc is handled separately)
CXXFLAGS += -MMD -MP

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir)) \
    -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=$(WITH_ABI)
CXXFLAGS += -pthread -fPIC -fwrapv -std=c++14 $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -std=c++14 -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)



GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
$(GTEST_DIR)/include/gtest/internal/*.h


# Builds gtest.a and gtest_main.a.
# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)
# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc
gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc
gtest.a : gtest-all.o
	$(AR) $(ARFLAGS) $@ $^
gtest_main.a : gtest-all.o gtest_main.o
	$(AR) $(ARFLAGS) $@ $^


# Builds a sample test.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.
threadmap.o : $(USER_DIR)/threadmap.cu $(GTEST_HEADERS)
	@ echo NVCC $<
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

iterator.o : $(USER_DIR)/iterator.cu $(GTEST_HEADERS)
	@ echo NVCC $<
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -M $< -o ${@:.o=.d} \
		-odir $(@D)
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

# segmentation fault on gtest version 1.7.0 , @0fe96607
# check out 1.8.1
threadmap : threadmap.o gtest_main.a
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -I$(GTEST_DIR)  $(CPPFLAGS) $(LDFLAGS) -o $@  $^

iterator : iterator.o gtest_main.a
	$(Q)nvcc $(NVCCFLAGS) $(CUDA_ARCH) -I$(GTEST_DIR)  $(CPPFLAGS) $(LDFLAGS) -o $@  $^


simple.o : simple.cpp
	nvcc $(NVCCFLAGS) $(CUDA_ARCH) -I$(GTEST_DIR)  $(CPPFLAGS) $(LDFLAGS)   -o simple.o -c simple.cpp

simple : simple.o gtest_main.a
	nvcc $(NVCCFLAGS) $(CUDA_ARCH) -I$(GTEST_DIR)  $(CPPFLAGS) $(LDFLAGS)  -o simple simple.o gtest_main.a -lpthread
