
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "sphconv/sphconv.h"
#include "sphconv/indice_conv/coord.cu.h"
#include "sphconv/indice_conv/layout.cu.h"

using cutlass::sizeof_bits;

namespace sphconv {

namespace threadblock {

template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, typename AccessType>
class PredicatedTileAccessIterator;

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_>
class PredicatedTileAccessIterator<Shape_, Element_, sphconv::layout::TensorNTHWC,
                                    AdvanceRank, ThreadMap_, AccessType_> {
public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNTHWC;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TorchTensor<Element, 5>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = Pointer;

  static int const kAdvanceRank = AdvanceRank;

  static int const kAccessesPerVector = ThreadMap::kElementsPermAccess / AccessType::kElements;

  static int const kPredicatesPerByte = 4;
  static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

  static int const kPredicateCount = ThreadMap::Iterations::kCount * kAccessesPerVector;

  /// Number of 32b words containing predicates
  static int const kPredicateByteCount =
    (kPredicateCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
  static int const kPredicateWordCount =  (kPredicateByteCount + 3) / 4;

  static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

  using Mask = cutlass::Array<uint32_t, kPredicateWordCount>;

  class Params {
    public:
    friend PredicatedTileAccessIterator;

    private:
    /// stride of pitch-linear layout (units of Element)
    int stride_;
    /// amount (in byte) to increment pointer to move to next access along
    /// strided dimension
    LongIndex inc_strided_;
    /// amount (in byte) to increment pointer from last access to first access
    /// of next tile
    LongIndex inc_next_;
    /// amount (in byte) to increment pointer from first access of current tile
    /// to first access of next tile
    LongIndex inc_advance_;

    public:
    CUTLASS_HOST_DEVICE
    Params(): stride_(0), inc_strided_(0), inc_next_(0), inc_advance_(0) { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)  {
      // in unit of element
      stride_ = layout.stride(0) * layout.stride(1);

      // in byte
      inc_strided_ = (LongIndex(stride_) * ThreadMap::Delta::kStrided)
        * sizeof_bits<Elements>::value / 8;

      if (kAdvanceRank) {
        inc_advance_ = Shape::kStrided * stride_ * int(sizeof(Element));
      } else {
        inc_advance_ = Shape::kContiguous * int(sizeof(Element));
      }

      inc_next = inc_advance_ - (ThreadMap::Iterations::kStrided - 1)
        * ThreadMap::Delta::kStrided * stride_ * int(sizeof(Element));
    }
  };

private:
  using BytePointer = char *;

private:
  Params const &params_;

  BytePointer pointer_;

  uint32_t predicates_[kPredicateWordCount];

  TensorCoord extent_;

  /// Inital offset for each thread
  TensorCoord thread_offset_;

  /// Offset to the first steady-state tile
  TensorCoord residue_offset_;

  /// Used for out-of-order visitation
  bool is_residue_tile_;

  /// Iteration along vectors implied by the thread map
  int iteration_vector_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_HOST_DEVICE
  void compute_predicates_(
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0u;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided; ts++) {

          TensorCoord iteration_coord(c * ThreadMap::Delta::kContiguous,
                                      ts + s * ThreadMap::Delta::kStrided);

          TensorCoord coord = thread_offset_ + iteration_coord;

          bool guard;

          if (is_steady_state) {
            if (kAdvanceRank == 0) {
              guard = (coord.h() < extent_.h());
            } else {
              guard = (coord.w() < extent_.w());
            }
          } else {
            guard = (coord.h() < extent_.h() &&
                     coord.w() < extent_.w());
          }

          int pred_idx = ts + c *  ThreadMap::ThreadAccessShape::kStrided + s * ThreadMap::Iterations::kContiguous *  ThreadMap::ThreadAccessShape::kStrided;
          int word_idx = pred_idx / kPredicatesPerWord;
          int residual = pred_idx % kPredicatesPerWord;
          int byte_idx = residual / kPredicatesPerByte;
          int bit_idx = residual % kPredicatesPerByte;

          predicates_[word_idx] |= (unsigned(guard) << (byte_idx * 8 + bit_idx));

        }
      }
    }
  }

public:
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator(
    Params const &params,
    Pointer pointer,
    TensorCoord extent,
    int thread_id,
    TensorCoord const &threadblock_offset)
    : params_(params),
    pointer_(pointer),
    extent_(extent),
    is_residue_tile_(true)
  {
    TensorCoord residue_extent;
    if (kAdvanceRank) {
      // n t h w c
      //
      Index residue_size = (extent_[kAdvanceRank + 1] - threadblock_offset.stride()) %
        Shape::kStrided;
      if (!residue_size) {
        residue_size = Shape::kStirded;
      }

      // TODO !
      residue_offset_ = make_Coord(0, 0, 0, residue_size);
      residue_extent = make_Coord( 0, 0, 0, extent_.w(),  extent_.h());
    } else {

      //TODO
    }
      // Per-thread offset in logical coordinates of tensor
    thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);

    // update internal pointers
    Layout layout(params_.stride_);
    add_pointer_offset(layout(thread_offset_));

    compute_predicates_(residue_extent, false);

    set_iteration_index(0);

  }

  void set_iteration_index(int index) {}

  void add_pointer_offset(LongIndex pointer_offset) {}

  AccessType *get() const { return 0; }

  PredicatedTileAccessIterator &operator++() {
    return *this;
  }

  PredicatedTileAccessIterator operator++(int) {
    return *this;
  }

  void add_tile_offset(TensorCoord const &coord)
  {
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
class PredicatedTileIterator;

template<
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize>
class PredicatedTileIterator<Shape_, Element_, layout::TensorNTHWC, AdvanceRank,
  ThreadMap_, AccessSize>
{
  public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::TensorNTHWC;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TorchTensor<Element, 5>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = Pointer;
  //TODO: alignment for interal

  // N, T, H, W, C
  // access  [:, : i_i, j_j, :]
  // raw data
  using AccessType =
      cutlass::AlignedArray<
          Element, AccessSize, (AccessSize * sizeof_bits<Element>::value / 8)>;

  using TileAccessIterator =
      PredicatedTileAccessIterator<
          Shape, Element, Layout, kAdvanceRank, ThreadMap, AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  using Fragment = cutlass::Array<
    Element,
    ThreadMap::Iterations::kH *
    ThreadMap::Iterations::kW *
    ThreadMap::Iterations::kC *
    ThreadMap::kElementsPerAccess>;

  using Mask = typename TileAccessIterator::Mask;
//
  class Params {
    friend PredicatedTileIterator;

    private:
    typename TileAccessIterator::Params params_;

    public:
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : params_(layout) { }

    CUTLASS_HOST_DEVICE
    Params() { }
  };

private:
  /// Interal pointer type permits fast address arithmetic
  using BytePointer = char *;

private:
  TileAccessIterator address_iterator_;

public:
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,
      /// Extent of tensor, nthwc, on 'w' and 'h' dimension
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset)
      : address_iterator_(params.params_, pointer, extent, thread_id,
                          threadblock_offset) {}

  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,
    Pointer pointer,
    TensorCoord extent,
    int thread_id)
    : PredicatedTileIterator(params, pointer, extent, thread_id,
        make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});
    return *this;
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask() { address_iterator_.clear_mask(); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { address_iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { address_iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { address_iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset)
  {
    // n t h w c
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    for (int n = 0; n < extent_.n() ; ++n) {
      for (int t = 0; t < extent_.t(); ++t) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < ThreadMap::Iterations::kH ; ++h) {
          CUTLASS_PRAGMA_UNROLL
          for (int w = 0; w < ThreadMap::Iterations::kW ; ++w) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kC; ++c) {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < kAccessesPerVector; ++v) {
                int idx = v + kAccessesPerVector *
                  (c + ThreadMap::Iterations::kC *
                  (w +  ThreadMap::Iterations::kW *
                  (t + ThreadMap::Iterations::kH *
                  (n + extent_.t()))));
                address_iterator_.set_iteration_index(idx);
                char const *byte_ptr =
                  reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;

                AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

                cutlass::arch::global_load<
                  AccessType
                  sizeof(AccessType)>(
                    frag_ptr[idx], access_ptr, address_iterator_.valid());
                ++address_iterator_;
              }
            }
          }
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_byte_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }


  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    // n t h w c
    for (int n = 0; n < extent_.n() ; ++n) {
      for (int t = 0; t < extent_.t() ; ++t) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < ThreadMap::Iterations::kH ; ++h) {
          CUTLASS_PRAGMA_UNROLL
          for (int w = 0; w < ThreadMap::Iterations::kW ; ++w) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kC; ++c) {
              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < kAccessesPerVector; ++v) {
                int idx = v + kAccessesPerVector *
                  (c + ThreadMap::Iterations::kC *
                  (w +  ThreadMap::Iterations::kW *
                  (t + ThreadMap::Iterations::kH *
                  (n + extent_.t()))));

                address_iterator_.set_iteration_index(idx);

                char *byte_ptr =
                 reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
                AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

                if (address_iterator_.valid()) {
                  *access_ptr = frag_ptr[idx];
                }
                ++address_iterator_;
              }
            }
          }
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }
};

template<typename A>
struct FilterIterator {

};


} // namespace threadblock
} // namespace sphconv

