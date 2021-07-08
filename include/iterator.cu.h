#pragma once

#include <torch/extension.h>

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

// https://github.com/NVIDIA/cutlass/blob/master/media/docs/tile_iterator_concept.md

#include "debug_utils.h"

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

namespace threadblock
{

// concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator
template <typename Element_,
          typename ThreadMap_,
          int Alignment,
          int In, // rule[tile, k, In]
          bool Transpose = false,
          int AdvanceRank = 1
          >
struct InputTileIterator;

// for input feature in global memeory
// load from global memory to Fragment
// feature, [NNZ, C]
template <typename Element_,
          typename ThreadMap_,
          int Alignment,
          int In> // rule[tile, k, In]
struct InputTileIterator<Element_, ThreadMap_, Alignment, In, false, 1> {
    using Element = Element_;

    using ThreadMap = ThreadMap_;
    static int const kAlignment = Alignment;

    /// < Shape type describing extent of tile. pitchlinear
    using Shape = typename ThreadMap::Shape;

    ///< fragment object derived from cutlass::Array<Element, N>
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kContiguous * ThreadMap::Iterations::kStrided>;

    using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

    using Layout = cutlass::layout::PitchLinear;
    // using TensorRef = TensorRef<Element, Layout>;
    // using ConstTensorRef = typename TensorRef::ConstTensorRef;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    const static int kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
    static_assert(kAccessesPerVector == 1, "accesse per vector must be 1");

    struct Params {
        GpuTensor<Element, 2> inFeature_;  // [NNZ, C]
        GpuTensor<int32_t, 4> Rule_;       // [Ntile, KKK, 2, tile_n_max]
        GpuTensor<int32_t, 2> ruleSize_;   // [Ntile, KKK]

        Params() : inFeature_(0, zeros, zeros),
                   Rule_(0, zeros, zeros),
                   ruleSize_(0, zeros, zeros)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 2>& inFeature,
               const GpuTensor<int32_t, 4>& Rule,
               const GpuTensor<int32_t, 2>& ruleSize)
            : inFeature_(inFeature),
              Rule_(Rule),
              ruleSize_(ruleSize)
        {
        }
    };

    struct Mask {
        static int const kCount = ThreadMap::Iterations::kStrided;
        int global_offset[kCount];

        CUTLASS_DEVICE
        Mask() {}

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                global_offset[i] = -1;
            }
        }
    };

private:
    Mask mask_;
    Params const& params_;

    /// A threas's starting v
    Index thread_start_v_;
    Index thread_start_c_;
    int tile_idx_;
    int kernel_offset_;
    int rule_size_;

    /// Iteration along vectors implied by the thread map
    // int iteration_vector_;

    /// Iteration in the contiguous dimension
    // int iteration_contiguous_;

    /// Iteration in the strided dimension
    // int iteration_strided_;

public:
    CUTLASS_DEVICE void clear_mask()
    {
        mask_.clear();
    }

    CUTLASS_DEVICE void enable_mask()
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void get_mask(Mask& mask)
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void set_mask(Mask const& mask)
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE
    InputTileIterator(Params const& params, int thread_idx, int tile_idx, int vbegin, int kernel_offset)
        : params_(params),
          tile_idx_(tile_idx),
          kernel_offset_(kernel_offset)
    {

        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_v_ = thread_offset.strided() + vbegin;
        thread_start_c_ = thread_offset.contiguous();
        rule_size_ = params.ruleSize_[tile_idx][kernel_offset];

        // if (thread_idx == 0) {
        //     printf(
        //         "input tile iterator: "
        //         "shape = Shape<%02d,%02d>,"
        //     " iterations = Shape<%02d,%02d>"
        //     " delta = Shape<%02d,%02d>\n",
        //         Shape::kContiguous, Shape::kStrided,
        //         ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided,
        //         ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided
        //     );
        // }

        // Initialize mask
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            int v = thread_start_v_ + s * ThreadMap::Delta::kStrided;
            mask_.global_offset[s] = (v < rule_size_)
                                         ? params_.Rule_[tile_idx_][kernel_offset_][In][v]
                                         : -1;
        }

        // printf("T%03d, start v, c = (%02d, %02d)\n", threadIdx.x, thread_start_v_, thread_start_c_);
    }

    CUTLASS_HOST_DEVICE InputTileIterator& operator++()
    {
        thread_start_c_ += Shape::kContiguous;
        return *this;
    }

    CUTLASS_HOST_DEVICE InputTileIterator operator++(int)
    {
        InputTileIterator self(*this);
        operator++();
        return self;
    }

    // loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment& frag)
    {
        int channel_size = params_.inFeature_.size(1);
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

            int global_offset = mask_.global_offset[s];

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

                int idx = c + s * ThreadMap::Iterations::kContiguous;

                int channel = thread_start_c_ + c * ThreadMap::Delta::kContiguous;

                // printf("input channel = %d\n", channel);
                AccessType const* access_ptr =
                    reinterpret_cast<AccessType const*>(
                        &(params_.inFeature_[global_offset][channel]));

                bool is_valid = (global_offset >= 0); // && channel ?

                // if (is_valid && channel < channel_size && channel == 0) {
                //     int v = thread_start_v_ + s * ThreadMap::Delta::kStrided;
                //     printf("in read T%03d, v, c = (%02d - %02d,%02d) = %f\n",
                //            threadIdx.x, v, global_offset, channel,
                //            params_.inFeature_[global_offset][channel]);
                // }

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[idx], access_ptr, is_valid && channel < channel_size);
            }
        }
    }
};



/// featuire [NNZ, C]

// specitalization for transposed iteartor
// for  feature.T
template <typename Element_,
          typename ThreadMap_,
          int Alignment,
          int In // rule[tile, k, In]
          >
struct InputTileIterator<Element_, ThreadMap_, Alignment, In, true, 0>  {
    using Element = Element_;

    using ThreadMap = ThreadMap_;
    static int const kAlignment = Alignment;

    /// < Shape type describing extent of tile. pitchlinear
    using Shape = typename ThreadMap::Shape;
    // strided, contiguous
    //  iC,  NNZ(V)

    ///< fragment object derived from cutlass::Array<Element, N>
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kContiguous * ThreadMap::Iterations::kStrided>;
        // M, N
        //

    using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

    using Layout = cutlass::layout::PitchLinear;
    // using TensorRef = TensorRef<Element, Layout>;
    // using ConstTensorRef = typename TensorRef::ConstTensorRef;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    const static int kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
    static_assert(kAccessesPerVector == 1, "accesse per vector must be 1");

    struct Params {
        GpuTensor<Element, 2> inFeature_;
        GpuTensor<int32_t, 4> Rule_;
        GpuTensor<int32_t, 2> ruleSize_;

        Params() : inFeature_(0, zeros, zeros),
                   Rule_(0, zeros, zeros),
                   ruleSize_(0, zeros, zeros)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 2>& inFeature,
               const GpuTensor<int32_t, 4>& Rule,
               const GpuTensor<int32_t, 2>& ruleSize)
            : inFeature_(inFeature),
              Rule_(Rule),
              ruleSize_(ruleSize)
        {
        }
    };

    struct Mask {
        static int const kCount = ThreadMap::Iterations::kContiguous;
        int global_offset[kCount];

        CUTLASS_DEVICE
        Mask() {}

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                global_offset[i] = -1;
            }
        }
    };

private:
    Mask mask_;
    Params const& params_;

    /// A threas's starting v
    Index thread_start_v_;
    Index thread_start_c_;
    int tile_idx_;
    int kernel_offset_;
    int rule_size_;

    /// Iteration along vectors implied by the thread map
    // int iteration_vector_;

    /// Iteration in the contiguous dimension
    // int iteration_contiguous_;

    /// Iteration in the strided dimension
    // int iteration_strided_;

public:
    CUTLASS_DEVICE void clear_mask()
    {
        mask_.clear();
    }

    CUTLASS_DEVICE void enable_mask()
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void get_mask(Mask& mask)
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void set_mask(Mask const& mask)
    {
        gpuUnimplementedErr;
    }


    CUTLASS_DEVICE
    InputTileIterator(Params const& params, int thread_idx, int tile_idx, int cbegin, int kernel_offset)
        : params_(params),
          tile_idx_(tile_idx),
          kernel_offset_(kernel_offset)
    {

        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_c_ = thread_offset.strided() + cbegin;
        thread_start_v_ = thread_offset.contiguous();
        rule_size_ = params.ruleSize_[tile_idx][kernel_offset];

        // Initialize mask
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            int v = thread_start_v_ + c * ThreadMap::Delta::kContiguous;
            mask_.global_offset[c] = (v < rule_size_)
                                         ? params_.Rule_[tile_idx_][kernel_offset_][In][v]
                                         : -1;
        }

        // printf("T%03d, start v, c = (%02d, %02d)\n", threadIdx.x, thread_start_v_, thread_start_c_);
    }

    CUTLASS_HOST_DEVICE InputTileIterator& operator++()
    {
        thread_start_v_ += Shape::kContiguous;

        // update mask
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            int v = thread_start_v_ + c * ThreadMap::Delta::kContiguous;
            mask_.global_offset[c] = (v < rule_size_)
                                         ? params_.Rule_[tile_idx_][kernel_offset_][In][v]
                                         : -1;
        }

        return *this;
    }

    CUTLASS_HOST_DEVICE InputTileIterator operator++(int)
    {
        InputTileIterator self(*this);
        operator++();
        return self;
    }

    // loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment& frag)
    {
        int channel_size = params_.inFeature_.size(1);
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        // TODO: remove this clear;
        // why do I need this?
        // because!, cutlass's mma compute the cast fragment first !
        // this cuases a bug
        frag.clear();

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

            int channel = thread_start_c_ + s * ThreadMap::Delta::kStrided;

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

                int global_offset = mask_.global_offset[c];

                // assume kAccessesPerVector = 1
                //             int idx = c + s * ThreadMap::Iterations::kContiguous;

                // iteration_contiguous_ = idx % ThreadMap::Iterations::kContiguous;
                // iteration_strided_ = idx / ThreadMap::Iterations::kContiguous;
                //     int channel = c;
                int idx = c + s * ThreadMap::Iterations::kContiguous;

                // printf("input channel = %d\n", channel);
                AccessType const* access_ptr =
                    reinterpret_cast<AccessType const*>(
                        &(params_.inFeature_[global_offset][channel]));

                bool is_valid = (global_offset >= 0); // && channel ?

                // if (is_valid && channel < channel_size && channel % 8 == 0) {
                //     int v = thread_start_v_ + c * ThreadMap::Delta::kContiguous;
                //     printf("A in read T%03d, v, c = (%02d - %02d,%02d) = %f\n",
                //            threadIdx.x, v, global_offset, channel,
                //            params_.inFeature_[global_offset][channel]);
                // }

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[idx], access_ptr, is_valid && channel < channel_size);
            }
        }
    }
};


/// featuire [NNZ, C]
// for , d_FeatureOut
// specitalization for transposed iteartor
template <typename Element_,
          typename ThreadMap_,
          int Alignment,
          int In // rule[tile, k, In]
          >
struct InputTileIterator<Element_, ThreadMap_, Alignment, In, false, 0>  {
    using Element = Element_;

    using ThreadMap = ThreadMap_;
    static int const kAlignment = Alignment;

    /// < Shape type describing extent of tile. pitchlinear
    using Shape = typename ThreadMap::Shape;
    // strided, contiguous
    //  iC,  NNZ(V)

    ///< fragment object derived from cutlass::Array<Element, N>
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kContiguous * ThreadMap::Iterations::kStrided>;
        // M, N
        //

    using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

    using Layout = cutlass::layout::PitchLinear;
    // using TensorRef = TensorRef<Element, Layout>;
    // using ConstTensorRef = typename TensorRef::ConstTensorRef;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    const static int kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
    static_assert(kAccessesPerVector == 1, "accesse per vector must be 1");

    struct Params {
        GpuTensor<Element, 2> inFeature_;
        GpuTensor<int32_t, 4> Rule_;
        GpuTensor<int32_t, 2> ruleSize_;

        Params() : inFeature_(0, zeros, zeros),
                   Rule_(0, zeros, zeros),
                   ruleSize_(0, zeros, zeros)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 2>& inFeature,
               const GpuTensor<int32_t, 4>& Rule,
               const GpuTensor<int32_t, 2>& ruleSize)
            : inFeature_(inFeature),
              Rule_(Rule),
              ruleSize_(ruleSize)
        {
        }
    };

    struct Mask {
        static int const kCount = ThreadMap::Iterations::kStrided;
        int global_offset[kCount];

        CUTLASS_DEVICE
        Mask() {}

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                global_offset[i] = -1;
            }
        }
    };

private:
    Mask mask_;
    Params const& params_;

    /// A threas's starting v
    Index thread_start_v_;
    Index thread_start_c_;
    int tile_idx_;
    int kernel_offset_;
    int rule_size_;

    /// Iteration along vectors implied by the thread map
    // int iteration_vector_;

    /// Iteration in the contiguous dimension
    // int iteration_contiguous_;

    /// Iteration in the strided dimension
    // int iteration_strided_;

public:
    CUTLASS_DEVICE void clear_mask()
    {
        mask_.clear();
    }

    CUTLASS_DEVICE void enable_mask()
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void get_mask(Mask& mask)
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void set_mask(Mask const& mask)
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE
    InputTileIterator(Params const& params, int thread_idx, int tile_idx, int vbegin, int kernel_offset)
        : params_(params),
          tile_idx_(tile_idx),
          kernel_offset_(kernel_offset)
    {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_v_ = thread_offset.strided() + vbegin;
        thread_start_c_ = thread_offset.contiguous();
        rule_size_ = params.ruleSize_[tile_idx][kernel_offset];

        // if (thread_idx == 0) {
        //     printf(
        //         "input tile iterator: "
        //         "shape = Shape<%02d,%02d>,"
        //     " iterations = Shape<%02d,%02d>"
        //     " delta = Shape<%02d,%02d>\n",
        //         Shape::kContiguous, Shape::kStrided,
        //         ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided,
        //         ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided
        //     );
        // }

        // Initialize mask
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            int v = thread_start_v_ + s * ThreadMap::Delta::kStrided;
            mask_.global_offset[s] = (v < rule_size_)
                                         ? params_.Rule_[tile_idx_][kernel_offset_][In][v]
                                         : -1;
        }

        // printf("T%03d, start v, c = (%02d, %02d)\n", threadIdx.x, thread_start_v_, thread_start_c_);
    }

    CUTLASS_HOST_DEVICE InputTileIterator& operator++()
    {
        thread_start_v_ += Shape::kStrided;

        // update mask
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            int v = thread_start_v_ + s * ThreadMap::Delta::kStrided;
            mask_.global_offset[s] = (v < rule_size_)
                                         ? params_.Rule_[tile_idx_][kernel_offset_][In][v]
                                         : -1;
        }
        return *this;
    }

    CUTLASS_HOST_DEVICE InputTileIterator operator++(int)
    {
        InputTileIterator self(*this);
        operator++();
        return self;
    }

    // loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment& frag)
    {
        int channel_size = params_.inFeature_.size(1);
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

            int global_offset = mask_.global_offset[s];

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {


                int idx = c + s * ThreadMap::Iterations::kContiguous;
                int channel = thread_start_c_ + c * ThreadMap::Delta::kContiguous;

                // printf("input channel = %d\n", channel);
                AccessType const* access_ptr =
                    reinterpret_cast<AccessType const*>(
                        &(params_.inFeature_[global_offset][channel]));

                bool is_valid = (global_offset >= 0); // && channel ?

                // if (is_valid && channel < channel_size && channel == 0) {
                //     int v = thread_start_v_ + s * ThreadMap::Delta::kStrided;
                //     printf("B in read T%03d, v, c = (%02d - %02d,%02d) = %f\n",
                //            threadIdx.x, v, global_offset, channel,
                //            params_.inFeature_[global_offset][channel]);
                // }

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[idx], access_ptr, is_valid && channel < channel_size);
            }
        }
    }
};





// concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator

// for kernel in global memeory
// load from global memory to Fragment
template <
    typename Element_,
    typename ThreadMap_,
    int Aligment>
struct KernelTileIterator {
    using Element = Element_;
    using ThreadMap = ThreadMap_;
    using Layout = cutlass::layout::PitchLinear;
    static int const kAlignment = Aligment;
    /// < Shape type describing extent of tile.
    // Ci * Co
    using Shape = typename ThreadMap::Shape;
    ///< fragment object derived from cutlass::Array<Element, N>
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kContiguous * ThreadMap::Iterations::kStrided>;

    using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

    // using TensorRef = TensorRef<Element, Layout>;
    // using ConstTensorRef = typename TensorRef::ConstTensorRef;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    struct Params {
        GpuTensor<Element, 3> weight_;

        Params()
            : weight_(0, zeros, zeros)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 3>& weight)
            : weight_(weight) {}
    };

private:
    Params const& params_;
    /// A threas's starting v
    Index thread_start_ci_;
    Index thread_start_co_;
    Index kernel_offset_;
    bool mask_;

public:
    CUTLASS_DEVICE
    KernelTileIterator(Params const& params, int thread_idx, int kernel_offset)
        : params_(params),
          kernel_offset_(kernel_offset)
    {
        // if (thread_idx == 0) {
        //     printf("kernel tile iterator shape = Shape<%02d,%02d>,"
        //     " iterations = Shape<%02d,%02d>"
        //     " delta = Shape<%02d,%02d>\n",
        //         Shape::kContiguous, Shape::kStrided,
        //         ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided,
        //         ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided
        //     );
        // }

        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_ci_ = thread_offset.strided();
        thread_start_co_ = thread_offset.contiguous();
        // init mask
        int oC = params.weight_.size(2);
        mask_ = thread_start_co_ < oC;

        // printf("T%03d, start s,c = (%02d, %02d) %d\n", threadIdx.x, thread_start_ci_, thread_start_co_, mask_);
    }

    CUTLASS_DEVICE KernelTileIterator& operator++()
    {
        thread_start_ci_ += Shape::kStrided;
        return *this;
    }

    CUTLASS_DEVICE KernelTileIterator operator++(int)
    {
        KernelTileIterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_DEVICE void clear_mask()
    {
        mask_ = false;
    }

    // loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment& frag)
    {
        int in_channel_size = params_.weight_.size(1);

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

                int idx = c + s * ThreadMap::Iterations::kContiguous;

                int ci = thread_start_ci_ + s * ThreadMap::Delta::kStrided;
                int co = thread_start_co_ + c * ThreadMap::Delta::kContiguous;

                // if (mask_ && ci < in_channel_size && co == 0) {
                //     printf("read kernerl T%03d, ci, co = (%02d,%02d) = %f\n",
                //            threadIdx.x, ci, co,
                //            params_.weight_[kernel_offset_][ci][co]);
                // }

                AccessType const* access_ptr =
                    reinterpret_cast<AccessType const*>(
                        &(params_.weight_[kernel_offset_][ci][co]));

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[idx], access_ptr, mask_ &&  ci < in_channel_size);
            }
        }
    }
};





// nTile d_weight

template <
    typename Element_,
    typename ThreadMap_,
    int Aligment>
struct NKernelTileIterator {
    using Element = Element_;
    using ThreadMap = ThreadMap_;
    static int const kAlignment = Aligment;
    /// < Shape type describing extent of tile.
    // Ci * Co
    using Shape = typename ThreadMap::Shape;
    ///< fragment object derived from cutlass::Array<Element, N>

    using Layout = cutlass::layout::RowMajor;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");
    //   static_assert( ThreadMap::Iterations::kGroup == 1,"ThreadMap::Iterations::kGroup must be  1");
    //   static_assert( ThreadMap::Iterations::kCluster== 1,"ThreadMap::Iterations::kCluster must be  1");

    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kColumn *
            ThreadMap::Iterations::kRow *
            ThreadMap::Iterations::kGroup *
            ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    struct Params {
        GpuTensor<Element, 4> weight_;

        Index increment_row;     ///< increment quantity (in rows) to advance when moving between rows
        Index increment_group;   ///< increment quantity (in rows) to advance when moving to the next group
        Index increment_cluster; ///< increment quantity (in rowas) to advance when moving to the next cluster

        Index advance_row;     ///< amount to add to move to the next 'row' position
        Index advance_group;   ///< amount to add to move to the next 'group' position
        Index advance_cluster; ///< amount to add to move to the next 'cluster' position
        Index advance_tile;    ///< amount to add to move to the next 'tile'

        CUTLASS_HOST_DEVICE
        void initialize(int stride, cutlass::epilogue::threadblock::OutputTileThreadMapDesc thread_map)
        {
            increment_row = stride * thread_map.delta.row;

            increment_group = stride * thread_map.delta.group -
                              stride * thread_map.delta.row *
                                  (thread_map.iterations.row - 1);

            increment_cluster = stride * thread_map.delta.cluster -
                                stride * thread_map.delta.group *
                                    (thread_map.iterations.group - 1) -
                                stride * thread_map.delta.row *
                                    (thread_map.iterations.row - 1);

            advance_row = stride * thread_map.shape.row;

            advance_group = stride * (thread_map.shape.group - 1) *
                            thread_map.shape.row * thread_map.count.row;

            advance_cluster =
                stride *
                thread_map.count.group *
                thread_map.shape.group *
                thread_map.count.row *
                thread_map.shape.row;

            advance_tile =
                stride *
                thread_map.shape.group *
                thread_map.shape.row *
                thread_map.shape.cluster *
                thread_map.shape.tile;
        }

        Params()
            : weight_(0, zeros, zeros)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 4>& weight)
            : weight_(weight)
        {
            initialize(
                // outFeature.stride(0),
                Shape::kColumn,
                cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>());
        }
    };

    struct Mask {
        static int const kCount = ThreadMap::Iterations::kRow *
                                  ThreadMap::Iterations::kGroup *
                                  ThreadMap::Iterations::kCluster;

        bool row_guard[kCount];

        CUTLASS_DEVICE
        Mask()
        {
        }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                row_guard[i] = false;
            }
        }
    };

    CUTLASS_DEVICE
    void init_mask(int tile_idx, int kernel_offset)
    {
        int iC = params_.weight_.size(2);
        // if (threadIdx.x == 0)
        //     printf(" thread_start_ci_ :%d, iC :%d \n", thread_start_ci_, iC);
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow *
                                   (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = (row_offset + thread_start_ci_) < iC;

                    mask_.row_guard[frag_row_idx] = row_guard;
                }
            }
        }
    }


private:
    Mask mask_;
    Params const& params_;
    /// A threas's starting v
    int thread_start_ci_;
    int thread_start_co_;
    int kernel_offset_;
    int tile_idx_;

    // element offset
    int offset_;

    /// Internal state counter
    int state_[3];

public:
    CUTLASS_DEVICE
    NKernelTileIterator(Params const& params, int tile_idx,
                        int cbegin,
                        int thread_idx, int kernel_offset)
        : params_(params),
          tile_idx_(tile_idx),
          kernel_offset_(kernel_offset)
    {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_ci_ = thread_offset.row() + cbegin;
        thread_start_co_ = thread_offset.column();
        // init mask
        init_mask(tile_idx, kernel_offset);

        // Initialize pointer
        offset_ = thread_offset.row() * Shape::kColumn + thread_offset.column();
        state_[0] = state_[1] = state_[2] = 0;
    }

    CUTLASS_DEVICE NKernelTileIterator& operator++()
    {
        ++state_[0];

        offset_ += params_.advance_row;
        thread_start_ci_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {

            state_[0] = 0;
            ++state_[1];
            offset_ += params_.advance_group;

            thread_start_ci_ += (ThreadMap::Shape::kGroup - 1) *
                               ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {

                state_[1] = 0;
                ++state_[2];
                offset_ += params_.advance_cluster;

                thread_start_ci_ += ThreadMap::Count::kGroup *
                                   ThreadMap::Shape::kGroup *
                                   ThreadMap::Count::kRow *
                                   ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    offset_ += params_.advance_tile;
                }
            }
        }

        // init mask
        init_mask(tile_idx_, kernel_offset_);

        return *this;
    }

    CUTLASS_DEVICE NKernelTileIterator operator++(int)
    {
        NKernelTileIterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_DEVICE void clear_mask()
    {
        mask_ = false;
    }

    // loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment& frag)
    {

    }


    CUTLASS_DEVICE
    void store(Fragment const& frag)
    {
        int offset = offset_;
        int oC = params_.weight_.size(3);

        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow *
                                   (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset =
                        row * ThreadMap::Delta::kRow +
                        group * ThreadMap::Delta::kGroup +
                        cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = mask_.row_guard[frag_row_idx];
                    int ic = row_offset + thread_start_ci_;

                    // TODO optimize out these division
                    // int ic = offset / Shape::kColumn;
                    int oc = offset % Shape::kColumn;

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

                        bool is_valid = (oc + column * ThreadMap::Delta::kColumn < oC) &&
                                        row_guard;

                        // if (is_valid)
                        //     printf("out T%03d, write icoc(%d, %d), row_offset:%d, thread_start_ci_:%d,  column:%02d, kcolumn:%02d, offset%d   :%d\n",
                        //         threadIdx.x,
                        //         ic , oc,
                        //         row_offset,
                        //         thread_start_ci_,
                        //         column, ThreadMap::Delta::kColumn,
                        //         offset,
                        //         is_valid);

                        AccessType* memory_pointer =
                            const_cast<AccessType*>(
                                reinterpret_cast<const AccessType*>(
                                    &(params_.weight_[tile_idx_][kernel_offset_][ic][oc])));

                        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                            is_valid);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        offset += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    offset += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                offset += params_.increment_cluster;
            }
        }
    }


};



// template<
//   typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
//   typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
//   int PartitionsK,                          ///< Number of partitions of the K dimension
//   typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
//   typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
//   typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
//   typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
//   typename OutputOp_,                       ///< Output operator
//   typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
//   int FragmentsPerPartition = 1                 ///< Used to coarsten the epilogue granularity

/// Satisfies: writableTileIterator | PredicatedTileIterator | ForwardTileIterator

//  ThreadMap to be used by epilogue::PredicatedTileIterator satisfying concept OutputTileThreadMap

template <
    typename Element_,
    typename ThreadMap_, //    < Thread map (conept: OutputTileThreadMap)
    int In>              // rule[t][k][In]
class OutTileIterator
{
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;
    using Element = Element_;

    using Layout = cutlass::layout::RowMajor;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using ConstTensorRef = typename TensorRef::ConstTensorRef;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = typename Layout::TensorCoord;

    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
    static int const kThreads = ThreadMap::kThreads;
    static int const kIterations = ThreadMap::Count::kTile;

    static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
    static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
    static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
    static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");
    //   static_assert( ThreadMap::Iterations::kGroup == 1,"ThreadMap::Iterations::kGroup must be  1");
    //   static_assert( ThreadMap::Iterations::kCluster== 1,"ThreadMap::Iterations::kCluster must be  1");

    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kColumn *
            ThreadMap::Iterations::kRow *
            ThreadMap::Iterations::kGroup *
            ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

    /// Memory access size
    using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess>;

    struct Params {
        GpuTensor<Element, 2> outFeature_;
        GpuTensor<int32_t, 4> Rule_;
        GpuTensor<int32_t, 2> ruleSize_;

        Index increment_row;     ///< increment quantity (in rows) to advance when moving between rows
        Index increment_group;   ///< increment quantity (in rows) to advance when moving to the next group
        Index increment_cluster; ///< increment quantity (in rowas) to advance when moving to the next cluster

        Index advance_row;     ///< amount to add to move to the next 'row' position
        Index advance_group;   ///< amount to add to move to the next 'group' position
        Index advance_cluster; ///< amount to add to move to the next 'cluster' position
        Index advance_tile;    ///< amount to add to move to the next 'tile'

        CUTLASS_HOST_DEVICE
        void initialize(int stride, cutlass::epilogue::threadblock::OutputTileThreadMapDesc thread_map)
        {
            increment_row = stride * thread_map.delta.row;

            increment_group = stride * thread_map.delta.group -
                              stride * thread_map.delta.row *
                                  (thread_map.iterations.row - 1);

            increment_cluster = stride * thread_map.delta.cluster -
                                stride * thread_map.delta.group *
                                    (thread_map.iterations.group - 1) -
                                stride * thread_map.delta.row *
                                    (thread_map.iterations.row - 1);

            advance_row = stride * thread_map.shape.row;

            advance_group = stride * (thread_map.shape.group - 1) *
                            thread_map.shape.row * thread_map.count.row;

            advance_cluster =
                stride *
                thread_map.count.group *
                thread_map.shape.group *
                thread_map.count.row *
                thread_map.shape.row;

            advance_tile =
                stride *
                thread_map.shape.group *
                thread_map.shape.row *
                thread_map.shape.cluster *
                thread_map.shape.tile;
        }

        Params()
            : outFeature_(0, zeros, zeros),
              Rule_(0, zeros, zeros),
              ruleSize_(0, zeros, zeros)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 2>& outFeature,
               const GpuTensor<int32_t, 4>& Rule,
               const GpuTensor<int32_t, 2>& ruleSize)
            : outFeature_(outFeature),
              Rule_(Rule),
              ruleSize_(ruleSize)
        {
            initialize(
                // outFeature.stride(0),
                Shape::kColumn,
                cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>());
        }
    };

    struct Mask {
        static int const kCount = ThreadMap::Iterations::kRow *
                                  ThreadMap::Iterations::kGroup *
                                  ThreadMap::Iterations::kCluster;

        int global_offset[kCount];

        CUTLASS_DEVICE
        Mask()
        {
            // printf("kCount = %d\n", kCount);
        }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear()
        {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                global_offset[i] = -1;
            }
        }
    };

    CUTLASS_DEVICE
    void init_mask(int tile_idx, int kernel_offset)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow *
                                   (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow +
                                     group * ThreadMap::Delta::kGroup +
                                     cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = (row_offset + thread_start_v_) < rule_size_;

                    mask_.global_offset[frag_row_idx] =
                        row_guard
                            ? params_.Rule_[tile_idx][kernel_offset][In][row_offset + thread_start_v_]
                            : -1;
                }
            }
        }
    }




private:
    Mask mask_;
    const Params &params_;

    /// Byte-level pointer

    int rule_size_;
    int tile_idx_;
    int kernel_offset_;

    int thread_start_v_;
    int thread_start_c_;

    /// Byte-level pointer
    // int pointer_;

    // element offset
    int offset_;

    /// Internal state counter
    int state_[3];


public:
    CUTLASS_DEVICE
    OutTileIterator(
        Params const& params,
        // TensorCoord extent,  // V x oC
        int tile_idx,
        int v_begin,
        int thread_idx,
        int kernel_offset)
        : params_(params),
          tile_idx_(tile_idx),
          kernel_offset_(kernel_offset)
    {

        // if (tile_idx == 0 && thread_idx == 0) {
        //     printf(
        //         "out tile iterator: "
        //         "shape = Shape<%d,%d,%d,%d,%d>, iterations = Shape<%d,%d,%d,%d,%d>"
        //         " delta = Shape<%d,%d,%d,%d,%d>\n",
        //         Shape::kColumn, Shape::kRow, Shape::kGroup, Shape::kCluster, Shape::kTile,
        //         ThreadMap::Iterations::kColumn, ThreadMap::Iterations::kRow, ThreadMap::Iterations::kGroup, ThreadMap::Iterations::kCluster, ThreadMap::Iterations::kTile,
        //         ThreadMap::Delta::kColumn, ThreadMap::Delta::kRow, ThreadMap::Delta::kGroup, ThreadMap::Delta::kCluster, ThreadMap::Delta::kTile);
        // }

        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        // printf(" inital offset = %d, %d\n", thread_offset.row(), thread_offset.column());
        thread_start_v_ = v_begin + thread_offset.row();
        thread_start_c_ = thread_offset.column();
        // printf("T%3d, startvc(%2d,%2d)\n", thread_idx, thread_start_v_, thread_start_c_);

        // if (((int)&params_ % 4) != 0) {
        //     printf("params_ (%p) not aligned\n", &params_);
        //     printf("tielidx:%d, v_begin:%d, thread_id:x%d, koffset:%d\n", tile_idx, v_begin, thread_idx, kernel_offset);
        // }

        rule_size_ = params.ruleSize_[tile_idx][kernel_offset];

        // Initialize mask
        init_mask(tile_idx, kernel_offset);

        // Initialize pointer
        offset_ = thread_offset.row() * Shape::kColumn + thread_offset.column();
        // LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
        // LongIndex(thread_offset.column())

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    CUTLASS_DEVICE OutTileIterator& operator++()
    {
        // if (threadIdx.x == 0)
        //     printf(" operator ++ \n");
        ++state_[0];

        offset_ += params_.advance_row;
        thread_start_v_ += ThreadMap::Shape::kRow;

        if (state_[0] == ThreadMap::Count::kRow) {

            state_[0] = 0;
            ++state_[1];
            offset_ += params_.advance_group;

            thread_start_v_ += (ThreadMap::Shape::kGroup - 1) *
                               ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {

                state_[1] = 0;
                ++state_[2];
                offset_ += params_.advance_cluster;

                thread_start_v_ += ThreadMap::Count::kGroup *
                                   ThreadMap::Shape::kGroup *
                                   ThreadMap::Count::kRow *
                                   ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                    offset_ += params_.advance_tile;
                }
            }
        }

        // printf("- T%03d   old vc(%02d, %02d), new vc(%02d, %02d),  startv (%02d  ->  %02d)\n",
        //         threadIdx.x,
        //        old_offset / Shape::kColumn, old_offset % Shape::kColumn,
        //        offset_ / Shape::kColumn, offset_ % Shape::kColumn,
        //        old_thread_start_v, thread_start_v_);

        // init mask
        init_mask(tile_idx_, kernel_offset_);

        return *this;
    }

    CUTLASS_DEVICE void clear_mask()
    {
        mask_.clear();
    }

    CUTLASS_DEVICE void enable_mask()
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void get_mask(Mask& mask)
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void set_mask(Mask const& mask)
    {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE
    void load(Fragment& frag)
    {

        int offset = offset_;
        int channel_size = params_.outFeature_.size(1);
        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow *
                                   (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset =
                        row * ThreadMap::Delta::kRow +
                        group * ThreadMap::Delta::kGroup +
                        cluster * ThreadMap::Delta::kCluster;

                    // printf("T:%03d  row_offset:%02d, thread_start_v:%02d \n",
                    //        threadIdx.x, row_offset, thread_start_v_);

                    int global_offset = mask_.global_offset[frag_row_idx];

                    //   AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

                    // TODO optimize out these division
                    // int v = offset / Shape::kColumn;
                    int c = offset % Shape::kColumn;
                    // v should =  thread_start_v + row_offset

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

                        // bool guard = row_guard && mask_.predicates[column];

                        bool is_valid = (global_offset >= 0) &&
                                        (c + column * ThreadMap::Delta::kColumn < channel_size);

                        // if (is_valid)
                        //     printf("out T%03d, read vc(%02d - %02d,%02d),  column:%02d, kcolumn:%02d, offset:%d   :%d\n",
                        //         threadIdx.x,
                        //         row_offset + thread_start_v_, global_offset,
                        //         c + column * ThreadMap::Delta::kColumn,
                        //         column, ThreadMap::Delta::kColumn,
                        //         offset,
                        //         is_valid);

                        AccessType* memory_pointer =
                            const_cast<AccessType*>(
                                reinterpret_cast<const AccessType*>(
                                    &(params_.outFeature_[global_offset][c])));

                        cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                            is_valid);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        offset += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    offset += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                offset += params_.increment_cluster;
            }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag)
    {
        // int start_c = thread_start_c_; // + column * ThreadMap::Delta::kColumn;
        int offset = offset_;
        int channel_size = params_.outFeature_.size(1);
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow *
                                   (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset =
                        row * ThreadMap::Delta::kRow +
                        group * ThreadMap::Delta::kGroup +
                        cluster * ThreadMap::Delta::kCluster;

                    int global_offset = mask_.global_offset[frag_row_idx];

                    // int v = offset / Shape::kColumn;
                    int c = offset % Shape::kColumn;

                    // printf("T%03d, out write vc(%02d, %02d), row_offset:%02d, thread_start_v:%02d\n",
                    //     threadIdx.x, v, c, row_offset, thread_start_v_
                    // );

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

                        // printf("output channel = %d\n", channel);
                        // printf("mem ptr = %p\n", memory_pointer);
                        bool is_valid = (global_offset >= 0) &&
                                        (c + column * ThreadMap::Delta::kColumn < channel_size);

                        // if (is_valid)
                        //     printf("out T%03d, writeto vc(%02d - %02d,%02d), column:%02d, kcolumn:%02d\n",
                        //         threadIdx.x,
                        //         row_offset + thread_start_v_, global_offset,
                        //         c + column * ThreadMap::Delta::kColumn,
                        //         column, ThreadMap::Delta::kColumn
                        //         );

                        AccessType* memory_pointer =
                            const_cast<AccessType*>(
                                reinterpret_cast<const AccessType*>(
                                    &(params_.outFeature_[global_offset][c])));

                        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                            is_valid);
                    }

                    if (row + 1 < ThreadMap::Iterations::kRow) {
                        offset += params_.increment_row;
                    }
                }

                if (group + 1 < ThreadMap::Iterations::kGroup) {
                    offset += params_.increment_group;
                }
            }

            if (cluster + 1 < ThreadMap::Iterations::kCluster) {
                offset += params_.increment_cluster;
            }
        }
    }
};

} // namespace threadblock
} // namespace sphconv
