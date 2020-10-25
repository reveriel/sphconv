#pragma once

#include "sphconv/sphconv.h"
#include "sphconv/indice_conv.h"

namespace sphconv
{

namespace kernel
{

template<
typename ElementFeature_,
typename ElementRuleMap_,
typename ElementKernel_,
typename ElementNumIn_,
typename ThreadblockShape,
typename WarpShape,
typename InstructionShape,
typename ThreadblockSwizzle_,
typename Operator,
typename Mma_,
typename Gatherer,
typename Scatterer
>
class IndiceConv {
  using Mma = Mma_;

  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using SharedStorage = threadblock::SharedStorage;

  struct Params {
    TileCoord problem_size; // the tile of Input feature's size (h, w)
    TileCoord grid_tiled_shape; // shape of tiles
    typename Mma::IteratorI::Params params_I;
    typename Mma::IteratorI::TensorRef ref_I;
    typename Mma::IteratorK::Params params_K;
    typename Mma::IteratorK::TensorRef ref_K;
    typename Mma::IteratorRule::Params params_inrule;
    typename Mma::IteratorRule::TensorRef ref_inrule;
    typename Mma::IteratorRule::Params params_outrule;
    typename Mma::IteratorRule::TensorRef ref_outrule;
    typename Mma::IteratorNumIn::Params params_numin
    typename Mma::IteratorNumIn::TensorRef ref_numin;

    typename Mma::OutputTileIterator::Params params_O;
    torch::PackedTensorAccessor32 ref_O;

    CUTLASS_HOST_DEVICE
    Params(
    ): () {
    }
  };


  // struct SharedStorage {
  typename Mma::SharedStorage main_loop;
  // };

  CUTLASS_HOST_DEVICE
  IndiceConv() { }

  /// Executes one indiceConv
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    ThreadblockSwizzle threadblock_swizzle;

    // the threadblock offset (in units of threadblock-scoped tiles)
    BatchedTileCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset();

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.h() <= threadblock_tile_offset.h() ||
      params.grid_tiled_shape.w() <= threadblock_tile_offset.w()) {

      return;
    }


    // Each CTA handles multiple batch indices to accomodate
    // in my case, the grid's Z dimension should be large enough
    for (int batch_idx = threadblock_swizzle.get_batch_idx();
      batch_idx < params.batch_count;
      batch_idx += gridDim.z) {

      // Compute inital location in logical coordinates
      cutlass::MatrixCoord tb_offset{
        threadblock_tile_offset.h
        0
      };

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      //
      typename Mma::IteratorI iterator_I(
        params.params_I,
        params.ref_I,
        params.problem_size
        thread_idx,
        tb_offset
      );
      iterator_I.add_pointer_offset();

      typename Mma::IteratorK iterator_K(
        params.params_K,
        params.ref_K,
        thread_idx,
        tb_offset
      );
      typename Mma::IteratorRule iterator_inrule(
        thread_idx,
        tb_offset
      );

      typename Mma::IteratorRule iterator_outrule(
        thread_idx,
        tb_offset
      );

      typename Mma::IteratorNumIn iterator_numin(
        thread_idx,
        tb_offset
      );


      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      // Main loop

      Mma mma(shared_storage, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentO out;

      out.clear();

      // Compute threadblock-scoped indice-conv
      mma(
        iterator_I,
        iterator_K,
        iterator_inrule,
        iterator_outrule,
        iterator_numin,
        out);

      typename OutputTileIterator iterator_O(
        params.params_O,
        params.ref_O,
      );

      iterator_O.add_pointer_offset( );
      }

  }

};





} // namespace kernel

} // namespace sphconv