
#pragma once

#include "cutlass/cutlass.h"
#include "sphconv/sphconv.h"
#include "sphconv/indice_conv/layout.cu.h"

namespace sphconv
{

namespace threadblock
{


template <typename Shape, typename Element, typename Layout, int AdvanceRank,
        typename ThreadMap, typename AccessType>
class TileAccessIterator;

template<typename Shape_, typename Element_, int AdvanceRank,
    typename ThreadMap_, typename AccessType_>
class TileAccessIterator<Shape_, Element_, sphconv::layout::TensorNTHWC,
      AdvanceRank, ThreadMap_, AccessType_> {
public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNTHWC;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using Index = typename Layout::LongIndex;

  using TensorRef = TorchTensor<Element, 5>;
  using TensorCoord = typename Layout::TensorCoord;
  using Pointer = Element *;
  using NonConstPointer = Pointer;

private:
  /// Data
public:
CUTLASS_HOST_DEVICE
TileAccessIterator(TensorRef ref, int thread_id)
{
}

void set_iteration_index(int index) {}

void add_pointer_offset(LongIndex pointer_offset) {}

AccessType *get() const { }

TileAccessIterator &operator++() {
}
TileAccessIterator operator++(int) {}

void add_tile_offset(TensorCoord const&coord) {

}
};



template<
  typename Shape,
  typename Element,
  typename Layout,
  int AdvanceRank,
  typename ThreadMap,
  int AccessSize = ThreadMap::kElementsPerAccess
>
class TileIterator;

template<
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize>
class TileIterator<Shape_, Element_, layout::TensorNTHWC, AdvanceRank,
  ThreadMap_, AccessSize>
{
  public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNTHWC;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using Index = typename Layout::Index;
  using Index = typename Layout::LongIndex;

  using TensorRef = TorchTensor<Element, 5>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = Pointer;
  //TODO: alignment for interal

  // N, T, H, W, C
  // access  [:, : i_i, j_j, :]
  // raw data
  using AccessType =
    cutlass::layout::Array<Element, ThreadMap::kElementsPerAccess >;

  // using TileAccessIterator =
  //   TileAccessIterator<Shape, Element, Layout, kAdvanceRank,
  //                ThreadMap, AccessType>;

  // static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;
  using Fragment = Array<Element,
  ThreadMap::Iterations::kH *
  ThreadMap::Iterations::kW *
  ThreadMap::Iterations::kC *
  ThreadMap::kElementsPerAccess>;

  // using Mask = typename TileAccessIterator::Mask;

  public:
  TileIterator(
    // Params const &params,
    Pointer pointer,
    int thread_id,
    TensorCoord const&threadblock_offset)
    : address_iterator_(ref, thread_id) {
    }

    /// Loads a fragment
    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
      AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
      uint8_t const *byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

    // access  [:, :, i_i, j_j, :]
    // when C is small
    // when C is
      for (int n = 0; n < ThreadMap::Iterations::kN; ++n)
      {
        for (int t = 0; n < it.t; ++t)
        {
          for (int h = 0; h < ThreadMap::Iterations::kH; ++h)
          {

            for (int w = 0; w < ThreadMap::Iterations::kW; ++w)
            {

              for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s)
              {
                AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_pointer);

                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < ThreadMap::Iterations::kC; ++c)
                {
                  int idx = c + s * ThreadMap::Iterations::kContiguous;
                  frag_ptr[idx] = access_ptr[c + s * ThreadMap::Iterations::kContiguous / ThreadMap::ThreadAccessShape::kStrided];
                }
                // if  (s + 1 <)
              }
            }
          }
        }
      }
    }

    /// Loads a fragment
    CUTLASS_HOST_DEVICE
    void load(Fragment &frag, TensorCoord const &tile_offset) {
      load_with_pointer_offset(
        frag,

        //
      );
    }
    /// Loads a fragment
    CUTLASS_HOST_DEVICE
    void load(Fragment &frag) {
      load_with_pointer_offset(frag, 0);
    }

    CUTLASS_HOST_DEVICE
    void store_with_pointer_offset(Fragment const&frag, Index pointer_offset)
    {
      // load
    }
    /// Stores a fragment
    CUTLASS_HOST_DEVICE
    void store(Fragment const &frag, TensorCoord const& tile_offset) {
      store_with_pointer_offset(
        frag,
        //
      );
    }

  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Advances the pointer
  TileIterator &operator++() {
    pointer_ +=
  }

};

template<typename A>
struct FilterIterator {

};



} // namespace threadblock


} // namespace sphconv

