#pragma once

#include <torch/extension.h>

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/tensor_ref.h"

// https://github.com/NVIDIA/cutlass/blob/master/media/docs/tile_iterator_concept.md

#include "debug_utils.h"

namespace sphconv
{

template <typename T, int N>
using GpuTensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

namespace threadblock
{


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

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::kElements;
    static_assert(kAccessesPerVector == 1, "accesse per vector must be 1");

    struct Params {
        const GpuTensor<Element, 2> inFeature_;
        const GpuTensor<int32_t, 4> Rule_;
        const GpuTensor<int32_t, 2> ruleSize_;

        //   Index increment_row;     ///< increment quantity (in rows) to advance when moving between rows
        //   Index increment_group;   ///< increment quantity (in rows) to advance when moving to the next group
        //   Index increment_cluster; ///< increment quantity (in rowas) to advance when moving to the next cluster

        //   Index advance_row;     ///< amount to add to move to the next 'row' position
        //   Index advance_group;   ///< amount to add to move to the next 'group' position
        //   Index advance_cluster; ///< amount to add to move to the next 'cluster' position
        //   Index advance_tile;    ///< amount to add to move to the next 'tile'

        //   CUTLASS_HOST_DEVICE
        //   void initialize(OutputtTileThreadMapDesc thread_map) {
        //       increment_row = thread_map.delta.row;

        //       increment_group = thread_map.delta.group - thread_map.delta.row * (thread_map.iterations.row - 1);

        //       increment_cluster = thread_map.delta.cluster - thread_map.delta.group * (thread_map.iterations.group - 1) - thread_map.delta.row * (thread_map.iterations.row - 1);

        //       advance_row = thread_map.shape.row;

        //       advance_group =
        //           (thread_map.shape.group - 1) * thread_map.shape.row * thread_map.count.row;

        //       advance_cluster =
        //           thread_map.count.group *
        //           thread_map.shape.group *
        //           thread_map.count.row *
        //           thread_map.shape.row;

        //       advance_tile =
        //           thread_map.shape.group *
        //           thread_map.shape.row *
        //           thread_map.shape.cluster *
        //           thread_map.shape.tile;
        //   }

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 2>& inFeature,
               const GpuTensor<int32_t, 4>& Rule,
               const GpuTensor<int32_t, 2>& ruleSize) : inFeature_(inFeature),
                                                        Rule_(Rule),
                                                        ruleSize_(ruleSize)
        {
            //   initialize(make_OutputTileThreadMapDesc<ThreadMap>());
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

        // Initialize mask
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kStrided; ++c) {
            int v = thread_start_v_ + ThreadMap::Delta::kStrided * c;
            mask_.global_offset[c] = (v < rule_size_) ? params.Rule_[tile_idx][kernel_offset][0][v] : -1;
        }
    }

    CUTLASS_HOST_DEVICE InputTileIterator& operator++()
    {
        thread_start_v_ += Shape::kStrided;

        // update mask
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kStrided; ++c) {
            int v = thread_start_v_ + ThreadMap::Delta::kStrided * c;
            mask_.global_offset[c] = (v < rule_size_) ? params_.Rule_[tile_idx_][kernel_offset_][0][v] : -1;
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

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

                // assume kAccessesPerVector = 1
                //             int idx = c + s * ThreadMap::Iterations::kContiguous;

                // iteration_contiguous_ = idx % ThreadMap::Iterations::kContiguous;
                // iteration_strided_ = idx / ThreadMap::Iterations::kContiguous;
                //     int channel = c;
                int idx = c + s * ThreadMap::Iterations::kContiguous;

                int channel = thread_start_c_ + c * ThreadMap::Delta::kContiguous;
                int global_offset = mask_.global_offset[s];

                // printf("input channel = %d\n", channel);
                AccessType const* access_ptr =
                    reinterpret_cast<AccessType const*>(
                        &(params_.inFeature_[global_offset][channel]));

                bool is_valid = (global_offset >= 0) && (channel < kChannelSize);

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[idx], access_ptr, is_valid);
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
struct KernelTileIterator {
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
        const GpuTensor<Element, 3> weight_;

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
          kernel_offset_(kernel_offset),
          mask_(true)
    {
        TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);
        thread_start_ci_ = thread_offset.strided();
        thread_start_co_ = thread_offset.contiguous();
    }

    CUTLASS_DEVICE KernelTileIterator& operator++()
    {
        thread_start_co_ += Shape::kContiguous;
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

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {

            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

                int idx = c + s * ThreadMap::Iterations::kContiguous;

                int ci = thread_start_ci_ + s * ThreadMap::Delta::kStrided;
                int co = thread_start_co_ + c * ThreadMap::Delta::kContiguous;

                // printf("kernel, ci, co = (%d,%d)\n", ci, co);
                AccessType const* access_ptr =
                    reinterpret_cast<AccessType const*>(
                        &(params_.weight_[kernel_offset_][ci][co]));

                cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                    frag_ptr[idx], access_ptr, mask_);
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
    typename ThreadMap_, //    < Thread map (conept: OutputTileThreadMap)
    typename Element_>
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
    using TensorCoord = cutlass::MatrixCoord;

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

    const static int kChannelSize = Shape::kColumn; // oC

    struct Params {
        GpuTensor<Element, 2> outFeature_;
        const GpuTensor<int32_t, 4> Rule_;
        const GpuTensor<int32_t, 2> ruleSize_;

        CUTLASS_HOST_DEVICE
        Params(const GpuTensor<Element, 2>& outFeature,
               const GpuTensor<int32_t, 4>& Rule,
               const GpuTensor<int32_t, 2>& ruleSize)
            : outFeature_(outFeature),
              Rule_(Rule),
              ruleSize_(ruleSize)
        {
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

private:
    Mask mask_;
    const Params const& params_;

    /// Byte-level pointer

    int rule_size_;
    int tile_idx_;
    int kernel_offset_;

    Index thread_start_c_;

    Index thread_start_v_;

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

        // v x ic      ic x oc
        // v x oc
        // 32 x 32
        // if (tile_idx == 0 && thread_idx == 0) {
        //     printf("shape = Shape<%d,%d,%d,%d,%d>, iterations = Shape<%d,%d,%d,%d,%d>"
        //     " delta = Shape<%d,%d,%d,%d,%d>\n",
        //         Shape::kColumn, Shape::kRow, Shape::kGroup, Shape::kCluster, Shape::kTile,
        //         ThreadMap::Iterations::kColumn, ThreadMap::Iterations::kRow, ThreadMap::Iterations::kGroup, ThreadMap::Iterations::kCluster, ThreadMap::Iterations::kTile,
        //         ThreadMap::Delta::kColumn, ThreadMap::Delta::kRow, ThreadMap::Delta::kGroup, ThreadMap::Delta::kCluster, ThreadMap::Delta::kTile
        //     );
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
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
                    // printf("row_offset = %2d, cluster,group,row:(%2d,%2d,%2d),\n", row_offset,
                    //     cluster, group, row);

                    bool row_guard = (row_offset + thread_start_v_) < rule_size_;

                    // printf("row_offset:%d, rowoff+start_v=%d, frag_row_idx=%d, start_c=%d \n",
                    //     row_offset, row_offset+thread_start_v_, frag_row_idx , thread_start_c_
                    // );

                    mask_.global_offset[frag_row_idx] =
                        row_guard ? params_.Rule_[tile_idx][kernel_offset][1][row_offset + thread_start_v_]
                                  : -1;

                    // if (thread_start_c_ == 0) {
                    //     printf("frag_row:%d, row_offset:%d, start_v:%d\n", frag_row_idx, row_offset, thread_start_v_);
                    // }
                }
            }
        }

        // Initialize internal state counter
        state_[0] = state_[1] = state_[2] = 0;
    }

    CUTLASS_DEVICE OutTileIterator& operator++()
    {
        ++state_[0];

        // row_idx_ += params_.advance_row;
        thread_start_v_ += ThreadMap::Shape::kRow;
        if (state_[0] == ThreadMap::Count::kRow) {

            state_[0] = 0;
            ++state_[1];

            thread_start_v_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

            if (state_[1] == ThreadMap::Count::kGroup) {

                state_[1] = 0;
                ++state_[2];

                thread_start_v_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

                if (state_[2] == ThreadMap::Count::kCluster) {
                    state_[2] = 0;
                }
            }
        }
        // init mask
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = (row_offset + thread_start_v_) < rule_size_;

                    mask_.global_offset[frag_row_idx] =
                        row_guard ? params_.Rule_[tile_idx_][kernel_offset_][1][row_offset + thread_start_v_]
                                  : -1;

                    // if (thread_start_c_ == 0) {
                    //     printf("update, frag_row:%d, row_offset:%d, start_v:%d\n", frag_row_idx, row_offset, thread_start_v_);
                    // }
                }
            }
        }

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

        AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = (row_offset + thread_start_v_) < rule_size_;

                    int global_offset = mask_.global_offset[frag_row_idx];

                    //   AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                        int channel = thread_start_c_; // + column * ThreadMap::Delta::kColumn;

                        // bool guard = row_guard && mask_.predicates[column];

                        bool is_valid = (global_offset >= 0) && (channel + column * ThreadMap::Delta::kColumn < kChannelSize) && row_guard;

                        AccessType* memory_pointer =
                            const_cast<AccessType*>(
                                reinterpret_cast<const AccessType*>(
                                    &(params_.outFeature_[global_offset][channel])));

                        cutlass::arch::global_load<
                            AccessType,
                            sizeof(AccessType)>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                                     column],
                            (void*)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                                   kElementsPerAccess],
                            is_valid);
                    }

                    //   if (row + 1 < ThreadMap::Iterations::kRow) {
                    //     byte_pointer += params_.increment_row;
                    //   }
                }

                // if (group + 1 < ThreadMap::Iterations::kGroup) {
                //   byte_pointer += params_.increment_group;
                // }
            }

            //   if (cluster + 1 < ThreadMap::Iterations::kCluster) {
            //     byte_pointer += params_.increment_cluster;
            //   }
        }
    }

    CUTLASS_DEVICE
    void store(Fragment const& frag)
    {
        // int row_idx = row_idx_;
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                    int frag_row_idx =
                        (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                    int row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

                    bool row_guard = (row_offset + thread_start_v_) < rule_size_;

                    int global_offset = mask_.global_offset[frag_row_idx];

                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

                        int channel = thread_start_c_;// + column * ThreadMap::Delta::kColumn;

                        // printf("output channel = %d\n", channel);
                        // printf("mem ptr = %p\n", memory_pointer);
                        bool is_valid = (global_offset >= 0) && (channel + column * ThreadMap::Delta::kColumn < kChannelSize) && row_guard;

                        // printf("T%03d, writeto vc(%02d,%02d)\n",
                        //        threadIdx.x, row_offset + thread_start_v_, channel);
                        AccessType* memory_pointer =
                            const_cast<AccessType*>(
                                reinterpret_cast<const AccessType*>(
                                    &(params_.outFeature_[global_offset][channel])));

                        cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                            frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                            (void*)&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                            is_valid);
                    }
                    // if (row + 1 < ThreadMap::Iterations::kRow) {
                    // }
                }
            }
        }
    }
};

} // namespace threadblock
} // namespace sphconv