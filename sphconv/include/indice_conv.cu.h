///////////////////////////////////////////////////////////////////
// indice conv kernels
///////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/coord.h>

using torch::RestrictPtrTraits;

namespace sphconv {

template <typename Index>
__global__ void indice_conv_kernel(
  const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    feature,
  torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    new_feature,
  const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
    InRuleMap,
  const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
    OutRuleMap,
  const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
    NumIn,
  const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    weight,
  int N,
  int in_channels,
  int out_channels,
  int kernel_volume,
  int KD, int KH, int KW,
  int sH, int sW,
  int padH, int padW,
  int dH, int dW,
  int oH, int oW,
  int H, int W)
{
  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= H || y >= W) return;

  for (Index b = 0; b < N; b++) {
    for (Index k = 0; k < kernel_volume ; k++) {

      Index k_D = k / (KH * KW);
      Index k_H = (k / KW) % KH;
      Index k_W = k % KW;

      Index oX = OutSpatial(k_H, x, sH, dH, padH);
      Index oY = OutSpatial(k_W, y, sW, dW, padW);

      if (oX >= oH || oX < 0 || oY >= oW || oY < 0 ) continue;

      for (int ic = 0; ic < in_channels; ic++) {
        for (int i = 0; i < NumIn[b][k][x][y]; i++) {

          Index oc = threadIdx.z;
          while (oc < out_channels) {

            // input thickness
            int it = InRuleMap[b][k][x][y][i];
            // output thickness
            int ot = OutRuleMap[b][k][x][y][i];

            atomicAdd(&new_feature[b][oc][ot][oX][oY], weight[oc][ic][k_D][k_H][k_W] * feature[b][ic][it][x][y]);

            oc += blockDim.z;
          }// while
        } // for i
      } // for ic
    } // for k
  } // for b
}


template <typename Index>
__global__ void indice_conv_backward_kernel(
  const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    d_featureOut,
  torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    d_feature,
  torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    d_weight,
  const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
    InRuleMap,
  const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
    OutRuleMap,
  const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
    NumIn,
  const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    weight,
  const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    feature,
  int N,
  int in_channels,
  int out_channels,
  int kernel_volume,
  int KD, int KH, int KW,
  int sH, int sW,
  int padH, int padW,
  int dH, int dW,
  int oH, int oW,
  int H, int W)
{

  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= H || y >= W) return;
  for (int b = 0; b < N; b++) {
    for (int k = 0; k < kernel_volume; k++) {
      Index k_D = k / (KH * KW);
      Index k_H = (k / KW) % KH;
      Index k_W = k % KW;

      Index oX = OutSpatial(k_H, x, sH, dH, padH);
      Index oY = OutSpatial(k_W, y, sW, dW, padW);

      if (oX >= oH || oX < 0 || oY >= oW || oY < 0 ) continue;

      for (int ic = 0; ic < in_channels; ic++) {
        for (int i = 0; i < NumIn[b][k][x][y]; i++) {
          Index oc = threadIdx.z;
          while (oc < out_channels) {

            // input thickness
            int it = InRuleMap[b][k][x][y][i];
            // output thickness
            int ot = OutRuleMap[b][k][x][y][i];

            atomicAdd(&d_feature[b][ic][it][x][y], weight[oc][ic][k_D][k_H][k_W] * d_featureOut[b][oc][ot][oX][oY]);
            atomicAdd(&d_weight[oc][ic][k_D][k_H][k_W], feature[b][ic][it][x][y] * d_featureOut[b][oc][ot][oX][oY]);

            oc += blockDim.z;
          }// while
        } // for i
      } // for ic
    }
  }
}


// I * K = O

/// TileCoord is a structure derived from Coord<2> that specifies a location within the
/// coordinate space of a indice_conv_tile problem.
struct TileCoord : public cutlass::Coord<2, int> {
  /// Integer-valued index
  typedef int Index;

  /// Base type is a Coord of rank=2
  typedef Coord<2, Index> Base;

  /// GEMM H dimension - rows of the Input I matrix
  static int const kH = 0;

  /// GEMM W dimension - columns of the Input I matrix
  static int const kW = 1;

  CUTLASS_HOST_DEVICE
  TileCoord() { }

  CUTLASS_HOST_DEVICE
  TileCoord(Coord<2, Index> const &coord): Base(cutlass::make_Coord(coord[0], coord[1])) { }

  CUTLASS_HOST_DEVICE
  TileCoord(Index h, Index w): Base(cutlass::make_Coord(h, w )) { }

  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  CUTLASS_HOST_DEVICE
  Index & h() { return this->at(kH); }

  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  CUTLASS_HOST_DEVICE
  Index & w() { return this->at(kW); }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  TileCoord operator+(Base const& b) const {
    return TileCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  TileCoord operator-(Base const& b) const {
    return TileCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  TileCoord operator*(Base const& b) const {
    return TileCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  TileCoord operator/(Base const& b) const {
    return TileCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  TileCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  TileCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  TileCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  TileCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }

};


/// BatchedTileCoord is a structure derived from Coord<3> that specifies a location within the
/// coordinate space of a batched indice conv problem.
struct BatchedTileCoord : public cutlass::Coord<3, int> {

  /// Integer-valued index
  typedef int Index;

  /// Base type is a Coord of rank=3
  typedef Coord<3, Index> Base;

  /// GEMM H dimension - rows of the input I matrix
  static int const kH = 0;

  /// GEMM W dimension - columns of the input I matrix
  static int const kW = 1;

  /// GEMM batch dimension - inner dimension of the GEMM problem
  static int const kBatch = 2;

  //
  // Methods
  //

  /// Default ctor
  CUTLASS_HOST_DEVICE
  BatchedTileCoord() { }

  /// Constructs from Coord<4>
  CUTLASS_HOST_DEVICE
  BatchedTileCoord(Base const &coord): Base(coord) { }

  /// Helper to construct from a H, W, and batch variables
  CUTLASS_HOST_DEVICE
  BatchedTileCoord(Index h, Index w, Index b): Base(cutlass::make_Coord(h, w, b)) { }

  /// Returns the H coordinate
  CUTLASS_HOST_DEVICE
  Index const & h() const { return this->at(kH); }

  /// Returns reference to the H coordinate
  CUTLASS_HOST_DEVICE
  Index & h() { return this->at(kH); }

  /// Returns the W coordinate
  CUTLASS_HOST_DEVICE
  Index const & w() const { return this->at(kW); }

  /// Returns reference to the W coordinate
  CUTLASS_HOST_DEVICE
  Index & w() { return this->at(kW); }

  /// Returns the batch coordinate
  CUTLASS_HOST_DEVICE
  Index const & batch() const { return this->at(kBatch); }

  /// Returns reference to the batch coordinate
  CUTLASS_HOST_DEVICE
  Index & batch() { return this->at(kBatch); }

  /// Obtains a TileCoord from BatchedTileCoord
  CUTLASS_HOST_DEVICE
  TileCoord hw() const {
    return TileCoord(h(), w());
  }

  /// Obtains a Coord<4> from BatchedTileCoord
  CUTLASS_HOST_DEVICE
  Coord<3> hwb() const {
    return cutlass::make_Coord(h(), w(), batch());
  }

  //
  // Coord operators
  //

  /// Element-wise addition
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator+(Base const& b) const {
    return BatchedTileCoord(Base::operator+(b));
  }

  /// Element-wise subtraction
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator-(Base const& b) const {
    return BatchedTileCoord(Base::operator-(b));
  }

  /// Element-wise multiplication
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator*(Base const& b) const {
    return BatchedTileCoord(Base::operator*(b));
  }

  /// Element-wise division
  CUTLASS_HOST_DEVICE
  BatchedTileCoord operator/(Base const& b) const {
    return BatchedTileCoord(Base::operator/(b));
  }

  /// In-place addition
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator+=(Base const& b) {
    Base::operator+=(b);
    return *this;
  }

  /// In-place subtraction
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator-=(Base const& b) {
    Base::operator-=(b);
    return *this;
  }

  /// In-place multiplication
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator*=(Base const& b) {
    Base::operator*=(b);
    return *this;
  }

  /// In-place division
  CUTLASS_HOST_DEVICE
  BatchedTileCoord& operator/=(Base const& b) {
    Base::operator/=(b);
    return *this;
  }
};





namespace device {


  template <typename Index>
  class IndiceConvTiled {
    public:
    IndiceConvTiled () {}

    using Kernel = kernel::IndiceConvTiled<typename;
    using ThreadblockSwizzle = threadblock::BatchedIdentityThreadblockSwizzle;

    struct Arguments
    {
      // Indice problem_size[];

      const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits> feature;
      torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits> new_feature;
      const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits> InRuleMap;
      const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits> OutRuleMap;
      const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits> NumIn;
      const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits> weight;
      int N;
      int in_channels;
      int out_channels;
      int KD, KH, KW;
      int kernel_volume;
      int sH, sW;
      int padH, padW;
      int dH, dW;
      int oH, oW;
      int H, W;
      int oT;

      inline
      Arguments() {}

      inline
      Arguments(
        torch::Tensor feature_,
        torch::Tensor weight_,
        torch::Tensor InRuleMap_,
        torch::Tensor OutRuleMap_,
        torch::Tensor NumIn_,
        //  torch::Tensor bias,
        int oT_,
        int sD_, int sH_, int sW_,
        int padD_, int padH_, int padW_,
        int dD_, int dH_, int dW_,
        int groups_)
      ):
      feature(feature_.packed_accessor32<float, 5, RestrictPtrTraits>()),
      new_feature(new_feature_.packed_accessor32<float, 5, RestrictPtrTraits>()),
      InRuleMap(InRuleMap_.packed_accessor32<int32_t, 5, RestrictPtrTraits>()),
      OutRuleMap(OutRuleMap_.packed_accessor32<int32_t, 5, RestrictPtrTraits>()),
      NumIn(NumIn_.packed_accessor32<int32_t, 4, RestrictPtrTraits>()),
      weight(weight_.packed_accessor32<float, 5, RestrictPtrTraits>()),
      N(feature.size(0)),
      in_channels(weight.size(1)),
      out_channels(weight.size(0)),
      KD(weight.size(2)),
      KH(weight.size(3)),
      KW(weight.size(4)),
      kernel_volume(KD * KH * KW),
      oT(oT_), sH(sH_), sW(sW_), padH(padH_), padW(padW_), dH(dH_), dW(dW_),
      H(feature.size(3)), W(feature.size(4)) { }

    };

  private:

  typename Kernel::IndiceConvTiled::Params params_;

  public:

    void initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr)
    {
      ThreadblockSwizzle threadblock_swizzle;

      BatchedTileCoord grid_shape = threadblock_swizzle.get_tiled_shape(
          args.problem_size,
          {ThreadblockShape::kH, threadblockShape::kW, ThreadblockShape::kBatch},
          args.batch_count);

        params_ =

    }

    void run(cudaStream_t stream = nullptr)
    {
      ThreadblockSwizzle threadblock_swizzle;

      dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
      dim3 block(GemmKernel::kThreadCount, 1, 1);

      cudaError_t result;

      int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(
          Kernel<GemmKernel>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
        printf("erro\n");
        return;
      }
    }

    cutlass::Kernel<><<<grid, block, smem_size, stream>>>



    }
  };


} // namespace device









namespace kernel
{

template<
typename Mma_
>
class IndiceConvTiled {
  using Mma = Mma_;

  using ThreadblockSwizzle = threadblock::BatchedIdentityThreadblockSwizzle;
  using SharedStorage = threadblock::SharedStorage;
  struct Params {
    TileCoord problem_size; // the tile of Input feature's size (h, w)
    TileCoord grid_tiled_shape; // shape of tiles
    typename Mma::IteratorI::Params params_I;
    torch::PackedTensorAccessor32 ref_I;

    typename Mma::IteratorK::Params params_K;
    torch::PackedTensorAccessor32 ref_K;

    typename Mma::OutputTileIterator::Params params_O;
    torch::PackedTensorAccessor32 ref_O;

  };

  // struct SharedStorage {
  typename threadblock::SharedStorage main_loop;

  // };

  CUTLASS_HOST_DEVICE
  IndiceConvTiled() { }

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
      cutlass::MatrixCoord tb_offset_I{
        threadblock_tile_offset.h
        0
      };

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      typename Mma::IteratorI iterator_I(
        params.params_I,
        params.ref_I,
        params.problem_size
        thread_idx,
        tb_offset_I
      );
      iterator_I.add_pointer_offset();

      typename Mma::IteratorK iterator_K(
        params.params_K,
        params.ref_K,
      )

      // Main loop

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      Mma mma(shared_storage, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentO out;

      out.clear();

      // Compute threadblock-scoped indice-conv
      mma(out);

      typename OutputTileIterator iterator_O(
        params.params_O,
        params.ref_O,
      );

      iterator_O.add_pointer_offset( );


      }


  }

};





} // namespace kernel




namespace threadblock
{

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxX() {
  return threadIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxY() {
  return threadIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxZ() {
  return threadIdx.z;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxX() {
  return blockIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxY() {
  return blockIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxZ() {
  return blockIdx.z;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX() {
  return blockDim.x;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimY() {
  return blockDim.y;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimZ() {
  return blockDim.z;
}

struct BatchedIdentityThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  BatchedTileCoord get_tiled_Shape(
    BatchedTileCoord problem_size,
    BatchedTileCoord tile_size,
    int batch_count) const {
      return BatchedTileCoord(
        (problem_size.h() + tile_size.h() - 1) / tile_size.h(),
        (problem_size.w() + tile_size.w() - 1) / tile_size.w(),
        (problem_size.batch() + tile_size.batch() - 1) / tile_size.batch());
    }


  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(BatchedTileCoord tiled_shape) const {
    return dim3(tiled_shape.h(), tiled_shape.w(), tiled_shape.batch());
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  BatchedTileCoord get_tile_offset() const {
    return BatchedTileCoord{
      RematerializeBlockIdxX(),
      RematerializeBlockIdxY(),
      RematerializeBlockIdxZ()
    };
  }

  /// Gets the batch tile index
  CUTLASS_DEVICE
  int get_batch_tile_idx() const {
    return RematerializeBlockIdxZ();
  }

  /// Gets the absolute batch index
  CUTLASS_DEVICE
  int get_batch_idx() const {
    return RematerializeBlockDimZ()*RematerializeBlockIdxZ() + RematerializeThreadIdxZ();
  }

  CUTLASS_DEVICE
  int get_batch_tile_idx() const {
    return RematerializeBlockIdxY();
  }

  /// Gets the absolute batch index
  CUTLASS_DEVICE
  int get_batch_idx() const {
    return RematerializeBlockDimY()*RematerializeBlockIdxY() + RematerializeThreadIdxY();
  }



};

struct SharedStorage {

};

template<typename A>
struct PredicatedTileIterator{

};

template<typename A>
struct FilterIterator {

};

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
  typename Shape_,
  typename Policy_,
  int Stages >
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
  using WarpCount = GemmShape< >; // TODO

  /// Number of stages
  static int const kStages = Stages;

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
class MmaPipelined : public MmaBase<Shape_, Policy_, 2>
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

  // staticaly assert kStages for MmaPipelined is two (Double-buffered pipeline)
  static_assert((Base::kStages == 2), "MmaPipelined requires kStages set to value 2");

private:
  using WarpFragmentI = typename Operator::FragmentI;
  using WarpFragmentK = typename Operator::FragmentK;

protected:
  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

public:

  /// Contrcut from tensor referecnes
  CUTLASS_DEVICE
  MmaPipelined(
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
    ++iterator_B;

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

  using ThreadblockMma = threadblock::MmaPipelined<>;

};


} // namespace threadblock















} // namespace sphconv
