#pragma once

#include "cutlass/cutlass.h"
#include "sphconv/sphconv.h"

namespace sphconv
{

namespace threadblock
{


/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
  typename Shape_,
  typename Policy_
  // int Stages
  >
class MmaBase {
  public:
  using Shape = Shape_;
  using Policy = Policy_;

  //
  // Dependent types
  //

  /// warp-level Mma
  using Operator = typename Policy::Operator;

  /// Shape describing the conv computed from shared memory
  /// by each warp
  using WarpConv = typename Policy::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  // the input is of shape B * T * H * W * C
  //  gather,
  //  get (B * T' * H * W) * C
  // T' is dynamic,
  using WarpCount = ?; // TODO

  /// Number of warp-level GEMM operations
  static int const kWarpGemmIterations = ?;

  /// Tensor I
  using TensorRefI = torch::PackedTensorAccessor32<ElementI, 5, torch::RestrictPtrTraits>;
  using TensorRefInRule = torch::PackedTensorAccessor32<ElementI, 5, torch::RestrictPtrTraits>;
  using TensorRefOutRule = torch::PackedTensorAccessor32<ElementI, 5, torch::RestrictPtrTraits>;
  using TensorRefNumIn = torch::PackedTensorAccessor32<ElementI, 5, torch::RestrictPtrTraits>;

  /// Tensor K
  using TensorRefK = torch::PackedTensorAccessor32<ElementK, 5, torch::RestrictPtrTraits>;

  //
  // Nested structs
  //

  /// Shared storage boject needed by threadblock-scoped
  class SharedStorage {
  public:
    /// shape of I ?

  public:
    //
    // Data members
    //

    /// Buffer for I operand
    cutlass::AlignedBuffer<typename Operator::ElementI, shapeI::kCount> operand_I;

    /// Buffer for K operand
    cutlass::AligendBuffer<typename Operator::ElementK, shapeK::kCount> operand_K;

  public:

    //
    // Methods
    //
  }

  protected:
  //
  // Data members
  //

  /// Iterator to load warp-scoped tile of I operand from shared memory
  typename Operator::IteratorI warp_tile_iterator_I_;

  /// Iterator to load warp-scoped tile of K operand from shared memory
  typename Operator::IteratorK warp_tile_iterator_K_;

  public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
      warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) { }

  )
};

template <
    /// Size of the Conv problem
    typename Shape_,
    /// Iterates over tiles of I operand in global memory
    typename IteratorI_,
    /// Iterates over tiles of I operand in shared memory
    typename SmemIteratorI_,
    /// Iterates over tiles of K  operand in global memory
    typename IteratorK_,
    /// Iterats over tiles of K operand in shared memory
    typename SmemIteratorK_,
    /// Data type of output O
    typename ElementO_,
    /// Policy describing tuning details
    typename Policy_,
    /// Number of stages
    int Stages>
class Mma: public MmaBase<Shape_, Policy_, 2>
{
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorI = IteratorI_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorK = IteratorK_;
  ///< Data type of accumulator matrix
  using ElementO = ElementO_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorI = SmemIteratorI_;
  using SmemIteratorK = SmemIteratorK_;

  //
  // Dependent types
  //

  /// Fragment of operand I loaded from global memory
  using FragmentI = typename IteratorI::Fragment;

  /// Fragment of operand K loaded from global memory
  using FragmentK = typename IteratorK::Fragment;

  /// Fragment
  using FragmentO = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  // staticaly assert kStages for Mma is two (Double-buffered pipeline)
  // static_assert((Base::kStages == 2), "MmaPipelined requires kStages set to value 2");

private:
  using WarpFragmentI = typename Operator::FragmentI;
  using WarpFragmentK = typename Operator::FragmentK;

protected:
  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_I_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_K_;

public:

  /// Contrcut from tensor referecnes
  CUTLASS_DEVICE
  Mma(
      typename Base::SharedStorage &shared_storage,
      int thread_idx,
      int warp_idx,
      int lane_idx) : Base(shared_storage, thread_idx, warp_idx, lane_idx),
                      smem_iterator_I_(shared_storage.operand_I_ref(), thread_idx),
                      smem_iterator_K_(shared_storage.operand_K_ref(), thread_idx)
  {
    // Compute warp location within thradblock tile by mapping the warp_id to
    // ?
    // ???j
    // move warp tile iterator to its position
    int warp_idx = 0;
    this->warp_tile_iterator_K_.add_tile_offset({warp_idx  });
  }

  /// Perform a threadblock-scoped indice-conv
  CUTLASS_DEVICE
  void operator()(
      int batch_iterations,
      FragmentO &output,
      IteratorI iterator_I,
      IteratorK iterator_K)
  {
    //
    // Prologue
    //

    FragmentI tb_frag_I;
    FragmentK tb_frag_K;

    tb_frag_I.clear();
    tb_frag_K.clear();

    // The lask block is loaded in the prolog
    iterator_I.load(tb_frag_I);
    iterator_K.load(tb_frag_K);

    ++iterator_I;
    ++iterator_K;

  }
};

template<typename T=void >
struct MmaCore {

};

struct Mma {
  // typename ElementI_
  using MmaCore = threadblock::MmaCore;

  using IteratorI = threadblock::PredicatedTileIterator<>;
  using IteratorK = threadblock::FilterIterator<>;

  using ThreadblockMma = threadblock::Mma<>;

};


} // namespace threadblock


} // namespace sphconv