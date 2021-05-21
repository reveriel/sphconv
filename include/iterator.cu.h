#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/memory.h"

// https://github.com/NVIDIA/cutlass/blob/master/media/docs/tile_iterator_concept.md

#include "debug_utils.h"

namespace sphconv {

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

namespace threadblock {


template<
        typename Shape_,  // pitchlinear
        int Threads,
        int ElementsPerAccess = 1>
struct InThreadMap {

    // TODO: split Rule to input and output rule
    /// 2d coordinate, RuLe[tile][k][0][v];
    //  the (v, ic) coordinate
    using TensorCoord = cutlass::layout::PitchLinearCoord;
    /// Tile shape, V x iC
    using Shape = Shape_;

    /// Number of threads total
    static int const kThreads = Threads;

    static int const kElementsPerAccess = ElementsPerAccess;

    static int const kWarpSize = 32;

    static int const kVStride = kThreads / kWarpSize;
    static int const kWarpCount = kThreads / kWarpSize;

    struct Detail {
        static_assert(!(Shape::kStrided % kWarpCount),
                      " shape on V dim must be divisible by warp count ");
    };

    /// Shape of access by each thread
    using ThreadAccessShape = cutlass::layout::PitchLinearShape<kElementsPerAccess, 1>;

    /// Number of iterations by each thread
    ///< Iterations along each dimension (concept: PitchLinearShape)
    using Iterations = cutlass::layout::PitchLinearShape<Shape::kContiguous / kWarpSize, Shape::kStrided / kWarpCount>;

    ///< Delta betweeen accesses (units of elements, concept: PitchLinearShape)
    using Delta = cutlass::layout::PitchLinearShape<kWarpSize, kWarpCount> ;

    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space,  (v, ic)
    CUTLASS_HOST_DEVICE
    static TensorCoord initial_offset(int thread_id) {
        int warp_id = (thread_id / kWarpSize);
        int lane_id = (thread_id % kWarpSize);
        return TensorCoord(lane_id, warp_id);
    }
};


// concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator

// for input feature in global memeory
// load from global memory to Fragment
template <typename Element_,
        typename ThreadMap_,
        int Alignment>
struct InputTileIterator {
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

    using Layout = cutlass::layout::RowMajor;
    // using TensorRef = TensorRef<Element, Layout>;
    // using ConstTensorRef = typename TensorRef::ConstTensorRef;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = cutlass::layout::PitchLinearCoord;

  const static int kChannelSize = Shape::kContiguous;

  struct Params  {
    const GpuTensor<Element, 2> &inFeature_;
    const GpuTensor<Index, 4> &Rule_;
    const GpuTensor<Index, 2> &ruleSize_;

    // CUTLASS_HOST_DEVICE
    Params(torch::Tensor &inFeature,
           torch::Tensor &localRules,
           torch::Tensor &ruleSize)
           : inFeature_ ( inFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
        Rule_ ( localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>()),
        ruleSize_ ( ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>())
    {
    }
  };


    struct Mask {
        static int const kCount = ThreadMap::Iterations::kStrided;
        int global_offset[kCount];

        CUTLASS_DEVICE
        Mask() { }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                global_offset[i] = -1;
            }
        }
    };

private:
    Mask mask_;
    Params const &params_;

    /// A threas's starting v
    Index thread_start_v_;
    Index thread_start_c_;
    int tile_idx_;
    int kernel_offset_;
    int rule_size_;


public:
    CUTLASS_DEVICE void clear_mask() {
        mask_.clear();
    }

    CUTLASS_DEVICE void enable_mask() {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void get_mask(Mask &mask) {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void set_mask(Mask const &mask) {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE
    InputTileIterator(Params const &params, int thread_idx, int tile_idx, int vbegin, int kernel_offset )
    : params_(params),
      tile_idx_(tile_idx),
      kernel_offset_(kernel_offset)
     {

        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_v_ = thread_offset.strided() + vbegin;
        thread_start_c_ = thread_offset.contiguous();
        rule_size_ = params.ruleSize_[tile_idx][kernel_offset];

        // Initialize mask
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kStrided; ++c) {
            int v = thread_start_v_ + ThreadMap::Delta::kStrided * c;
            mask_.global_offset[c] = (v < rule_size_) ? params.Rule_[tile_idx][kernel_offset][0][v] : -1;
        }
    }

    CUTLASS_HOST_DEVICE InputTileIterator &operator++() {
        thread_start_v_ += Shape::kStrided;

        // update mask
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kStrided; ++c) {
            int v = thread_start_v_ + ThreadMap::Delta::kStrided * c ;
            mask_.global_offset[c] = (v < rule_size_) ? params_.Rule_[tile_idx_][kernel_offset_][0][v] : -1;
        }
        return *this;
    }

    CUTLASS_HOST_DEVICE InputTileIterator operator++(int) {
        InputTileIterator self(*this);
        operator++();
        return self;
    }


    // loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment &frag) {

        AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

                int channel = thread_start_c_ + c * ThreadMap::Delta::kContiguous;
                int global_offset = mask_.global_offset[s];
                AccessType const *access_ptr =
                    reinterpret_cast<AccessType const *>(
                        &(params_.inFeature_[global_offset][channel]));

                bool is_valid = (global_offset >= 0) && (channel < kChannelSize);

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[s], access_ptr, is_valid);
            }
        }
    }
};


// concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator

// for kernel in global memeory
// load from global memory to Fragment
template <typename Element_,
          typename ThreadMap_,
          int Aligment>
struct KernelTileIterator
{
    using Element = Element_;

    using ThreadMap = ThreadMap_;
    static int const kAlignment = Aligment;

    /// < Shape type describing extent of tile.
    // Ci * Co
    using Shape = typename ThreadMap::Shape;

    ///< fragment object derived from cutlass::Array<Element, N>
    using Fragment = cutlass::Array<
        Element,
        ThreadMap::Iterations::kContiguous * ThreadMap::Iterations::kStrided>;

    using AccessType = cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

    using Layout = cutlass::layout::RowMajor;
    // using TensorRef = TensorRef<Element, Layout>;
    // using ConstTensorRef = typename TensorRef::ConstTensorRef;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorCoord = cutlass::layout::PitchLinearCoord;

    struct Params {
        const GpuTensor<Element, 3> &weight_;

        // CUTLASS_HOST_DEVICE
        Params(torch::Tensor &weight)
        : weight_(weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>())
        {
        }
    };

private:

    Params const &params_;
    /// A threas's starting v
    Index thread_start_co_;
    Index thread_start_ci_;
    Index kernel_offset_;
    bool mask_;

public:
    CUTLASS_DEVICE
    KernelTileIterator(Params const &params, int thread_idx, int kernel_offset)
    :params_(params),
    kernel_offset_(kernel_offset),
    mask_(true)
     {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_ci_ = thread_offset.strided();
        thread_start_co_ = thread_offset.contiguous();
    }

    CUTLASS_DEVICE KernelTileIterator &operator++() {
        return *this;
    }

    CUTLASS_DEVICE KernelTileIterator operator++(int) {
        KernelTileIterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_DEVICE void clear_mask() {
        mask_ = false;
    }


    // loads a fragment from memory
    CUTLASS_DEVICE void load(Fragment &frag) {

        AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

                int ci = thread_start_co_ + c * ThreadMap::Delta::kContiguous;
                int co = thread_start_ci_ + s * ThreadMap::Delta::kStrided;

                AccessType const *access_ptr =
                    reinterpret_cast<AccessType const *>(
                        &(params_.weight_[kernel_offset_][ci][co]));

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[s], access_ptr, mask_);
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
typename ThreadMap_,   //    < Thread map (conept: OutputTileThreadMap)
typename Element_
>
class OutTileIterator  {
public:
    using ThreadMap = ThreadMap_;
    using Shape = typename ThreadMap::Shape;
  using Element = Element_;

  using Layout = cutlass::layout::RowMajor;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = cutlass::MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");
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

  const static int kChannelSize = Shape::kColumn;

  struct Params  {
    const GpuTensor<Element, 2> &outFeature_;
    const GpuTensor<Index, 4> &Rule_;
    const GpuTensor<Index, 2> &ruleSize_;

    // CUTLASS_HOST_DEVICE
    Params(torch::Tensor &outFeature,
           torch::Tensor &localRules,
           torch::Tensor &ruleSize)
        :
        outFeature_(outFeature.packed_accessor32<float, 2, torch::RestrictPtrTraits>()),
        Rule_(localRules.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>()),
        ruleSize_(ruleSize.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>())
    {
    }
  };

    struct Mask {
        static int const kCount = ThreadMap::Iterations::kRow;

        int global_offset[kCount];

        CUTLASS_DEVICE
        Mask() { }

        ///< Efficiently disables all accesses guarded by mask
        CUTLASS_HOST_DEVICE void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i)
            {
                global_offset[i] = -1;
            }
        }
    };


private:

    Mask mask_;
    Params const &params_;

    /// Byte-level pointer

    int rule_size_;
    int tile_idx_;
    int kernel_offset_;

    Index thread_start_c_;

    Index thread_start_v_;


public:
    CUTLASS_DEVICE
    OutTileIterator(Params const &params, int tile_idx, int v_begin, int thread_idx, int kernel_offset)
    : params_(params),
     tile_idx_(tile_idx),
     kernel_offset_(kernel_offset)
     {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_v_ = v_begin + thread_offset.row();
        thread_start_c_ = thread_offset.column();
        rule_size_ = params.ruleSize_[tile_idx][kernel_offset];

        // Initialize mask
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            int v = thread_start_v_ + ThreadMap::Delta::kRow * row;
            mask_.global_offset[v] = (v < rule_size_) ?  params_.Rule_[tile_idx][kernel_offset][1][v] : -1;
        }

    }

    CUTLASS_DEVICE OutTileIterator &operator++() {
        thread_start_v_ += Shape::kRow;

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
            int v = thread_start_v_ + ThreadMap::Delta::kRow * row;
            mask_.global_offset[v] = (v < rule_size_) ?  params_.Rule_[tile_idx_][kernel_offset_][1][v] : -1;
        }

        return *this;
    }

    CUTLASS_DEVICE void clear_mask() {
        mask_.clear();
    }

    CUTLASS_DEVICE void enable_mask() {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void get_mask(Mask &mask) {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE void set_mask(Mask const &mask) {
        gpuUnimplementedErr;
    }


    CUTLASS_DEVICE
    void load(Fragment &frag) {
        gpuUnimplementedErr;
    }

    CUTLASS_DEVICE
    void store(Fragment const &frag) {
        AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow
                        + group + ThreadMap::Delta::kGroup
                        + cluster * ThreadMap::Delta::kCluster;

                    int global_offset = mask_.global_offset[row_offset];

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

                        int channel = thread_start_c_ + column * ThreadMap::Delta::kColumn;

                        AccessType *memory_pointer =
                            const_cast<AccessType *>(
                                reinterpret_cast<const AccessType *>(
                                    &(params_.outFeature_[global_offset][channel])));

                        bool is_valid = (mask_.global_offset[row_offset] >= 0) && (channel < kChannelSize);

                        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void *)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                            is_valid);

                    }
                }

            }
        }
    }
};



} // threadblock
} // sphconv