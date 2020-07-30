// #define DEBUG

#include "timer.h"
#include "debug_utils.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

using torch::RestrictPtrTraits;
// using namespace torch::indexing;

// input tile size = TILE + K
//
// __shared__ feature_tile[N][C][T][H_TILE][W_TILE];
// TODO: make T varadic length

// constexpr int T = 8;
// const int INPUT_TILE_H = 4;
// const int INPUT_TILE_W = 16;
// __shared__ scalar_t depth_tile[T][INPUT_TILE_H][INPUT_TILE_W];
// TODO
// __shared__ Index thickMap[INPUT_TILE_H][INPUT_TILE_W];
// __shared__ Index Num[3x3x3][INPUT_TILE_H][INPUT_TILE_W];

namespace sphconv {


///////////////////////////////////////////////////////////////////
// get indice pairs kernels
///////////////////////////////////////////////////////////////////

template <typename Index>
__global__ void get_indice_pairs_kernel_1(
    const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
        depth,
    const torch::PackedTensorAccessor32<Index, 3, RestrictPtrTraits>
        thick,
    torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      NumIn,
    torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
      InRuleMap,
    torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
      OutRuleMap,
    torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      CompactMap,
    int N,
    int H, int W,
    int KD, int KH, int KW,
    int sD, int sH, int sW,
    int padD, int padH, int padW,
    int dD, int dH, int dW,
    int oD, int oH, int oW
  )
{
  // input index
  // x y z ~ H W D
  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;
  Index k = threadIdx.z + blockDim.z * blockIdx.z;

  if (x >= H || y >= W ) return;

  Index k_D = k / (KH * KW);
  Index k_H = (k / KW) % KH;
  Index k_W = k % KW;

  Index oX = OutSpatial(k_H, x, sH, dH, padH);
  Index oY = OutSpatial(k_W, y, sW, dW, padW);
  for (int b = 0; b < N; b++) {
    for (int t = 0; t < thick[b][x][y]; t++)
    {
      Index z = depth[b][t][x][y];
      Index oZ = OutSpatial(k_D, z, sD, dD, padD);
      if (oX >= 0 && oX < oH && oY >= 0 && oY < oW  && oZ >= 0 && oZ < oD)
      {
        // the returned i seems tobe the old value
        // count number of active input
        Index i = atomicAdd(&NumIn[b][k][x][y], Index(1));

        // from which thick
        InRuleMap[b][k][x][y][i] = t;
        // to which z coordinate, this value is used to calculate the output thick
        OutRuleMap[b][k][x][y][i] = oZ;

        // fill nonempty place with 1
        CompactMap[b][oX][oY][oZ] = Index(1);

      } // if
    } // for t
  }// for b
}


template <typename Index>
__global__ void get_indice_pairs_kernel_2(
    torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      CompactMap,
    torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      new_depth,
    torch::PackedTensorAccessor32<Index, 3, RestrictPtrTraits>
      new_thick,
    int N,
    int kernel_volume,
    int oH, int oW, int oD)
{
  Index oX = threadIdx.x + blockDim.x * blockIdx.x;
  Index oY = threadIdx.y + blockDim.y * blockIdx.y;

  if (oX >= oH || oY >= oW) return;

  // scan (prefix sum) on CompactMap
  for (Index b = 0; b < N; b++) {
    int ot = 0;

    if (CompactMap[b][oX][oY][0] == 1) {
      new_depth[b][0][oX][oY] = 0;
      ot += 1;
    }

    Index *a = CompactMap[b][oX][oY].data();
    for (Index oZ = 1; oZ < oD; oZ++) {
      // if non empty
      int non_empty = a[oZ];

      // prefix sum
      a[oZ] += a[oZ - 1];

      if (non_empty) {
        new_depth[b][ot][oX][oY] = oZ;
        ot += 1;
      }
    } // for oZ

    new_thick[b][oX][oY] = CompactMap[b][oX][oY][oD - 1];

  } // for b
}

// re-assign OutRuleMap
template <typename Index>
__global__ void get_indice_pairs_kernel_3(
    const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      CompactMap,
    torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
      OutRuleMap,
    const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      NumIn,
    int N,
    int H, int W,
    int kernel_volume,
    int KD, int KH, int KW,
    int sH, int sW,
    int padH, int padW,
    int dH, int dW,
    int oH, int oW)
{

  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= H || y >= W) return;

  for (int b = 0; b < N; b++) {

    for (int k = 0; k < kernel_volume; k++) {

      // get oX and oY
      // int kD = k / (KH * KW);
      int k_H = (k / KW) % KH;
      int k_W = k % KW;
      Index oX = OutSpatial(k_H, x, sH, dH, padH);
      Index oY = OutSpatial(k_W, y, sW, dW, padW);

      if (oX < 0 || oX >= oH || oY < 0 || oY >= oW) continue;

      for (int i = 0; i < NumIn[b][k][x][y]; i++) {
        Index oZ = OutRuleMap[b][k][x][y][i];

        OutRuleMap[b][k][x][y][i] = CompactMap[b][oX][oY][oZ] - 1;
      }
    }
  }
}

// assign CompactMap for subm conv
template <typename Index>
__global__ void get_indice_pairs_subm_kernel_1(
    const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
        depth,
    const torch::PackedTensorAccessor32<Index, 3, RestrictPtrTraits>
        thick,
    torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      CompactMap,
    int N,
    int H, int W
)
{

  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= H || y >= W ) return;

  for (int b = 0; b < N; b++) {
    for (int t = 0; t < thick[b][x][y]; t++)
    {
      Index z = depth[b][t][x][y];
      CompactMap[b][x][y][z] = t + 1;
    }
  }
}

// init InRuleMap and OutRulemap, and NumIn
template <typename Index>
__global__ void get_indice_pairs_subm_kernel_2(
    const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
        depth,
    const torch::PackedTensorAccessor32<Index, 3, RestrictPtrTraits>
        thick,
    torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      NumIn,
    torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
      InRuleMap,
    torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
      OutRuleMap,
    const torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
      CompactMap,
    int N,
    int H, int W,
    int KD, int KH, int KW,
    int sD, int sH, int sW,
    int padD, int padH, int padW,
    int dD, int dH, int dW,
    int oD, int oH, int oW)
{

  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;
  Index k = threadIdx.z + blockDim.z * blockIdx.z;

  if (x >= H || y >= W) return;

  Index k_D = k / (KH * KW);
  Index k_H = (k / KW) % KH;
  Index k_W = k % KW;
  Index oX = OutSpatial(k_H, x, sH, dH, padH);
  Index oY = OutSpatial(k_W, y, sW, dW, padW);
  for (int b = 0; b < N; b++) {
    for (int t = 0; t < thick[b][x][y]; t++) {
      Index z = depth[b][t][x][y];
      Index oZ = OutSpatial(k_D, z, sD, dD, padD);
      if (oX >= 0 && oX < oH && oY >= 0 && oY < oW && oZ >= 0 && oZ < oD) {

        Index ot = CompactMap[b][oX][oY][oZ];
        if (ot == 0) continue;

        Index i = atomicAdd(&NumIn[b][k][x][y], Index(1));

        // from which thick
        InRuleMap[b][k][x][y][i] = t;
        // to which z coordinate, this value is used to calculate the output thick
        OutRuleMap[b][k][x][y][i] = ot - 1;
      }
    }
  }
}


///////////////////////////////////////////////////////////////////
// indice conv kernels
///////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////
// get indice pairs
///////////////////////////////////////////////////////////////////


///////
// return {new_depth, new_thick, InRuleMap, OutRuleMap, NumIn};
///////
std::vector<torch::Tensor>
get_indice_pairs(torch::Tensor depth,
                 torch::Tensor thick,
                 int D,
                 int KD, int KH, int KW,
                 int sD, int sH, int sW,
                 int padD, int padH, int padW,
                 int dD, int dH, int dW,
                 int groups)
{
  // assume we want to have each block to calcultate T output elements
  // (T + k - 1)^* inpuut elements are neededd ,

  // tile size
  // the output tlie
  // constexpr int H_TILE = 16, W_TILE = 16;
  int N = depth.size(0);
  int T = depth.size(1);
  int H = depth.size(2);
  int W = depth.size(3);

  int oD, oH, oW;
  oD = std::floor((D + 2 * padD - dD * (KD - 1) - 1) / sD + 1);
  oH = std::floor((H + 2 * padH - dH * (KH - 1) - 1) / sH + 1);
  oW = std::floor((W + 2 * padW - dW * (KW - 1) - 1) / sW + 1);
  int oT_MAX = T * 27;

  const int H_BLOCK = 4, W_BLOCK = 4;
  auto kernel_volume = KD * KH * KW;

  dim3 grid_size, block_size;
  at::Tensor new_depth, new_thick;

  // output tensor
  // int oT = T + 8 ; // This is bad
  // oT = T * 3 * 9; // This is even worse

  new_depth = torch::zeros(
      {N, oT_MAX, oH, oW}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  new_thick = torch::zeros(
      {N, oH, oW}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  // count number of valid input voxel at (b,k,x,y)
  auto NumIn = torch::zeros({N, kernel_volume, H, W},
                   torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  // the thickness of the valid input voxel
  auto InRuleMap = torch::full({N, kernel_volume, H, W, T},
    /*value=*/ -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  // the output thickness of the valid input voxel
  auto OutRuleMap = torch::full({N, kernel_volume, H, W, T},
    /*value=*/ -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  //// create <del>hash</del>map
  // the final value of CompactMap, means
  // the output thick + 1 at output coordinate, (b, oX, oY, oZ)
  auto CompactMap = torch::full({N, oH, oW, oD}, 0,
                  torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  const int oH_BLOCK = 8, oW_BLOCK = 32;

  printTensor_int(depth, "depth", 0, 0, H, 0, W);
  printTensor_int(thick, "thick", 0, 0, H, 0, W);

  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK));
  block_size = dim3(H_BLOCK, W_BLOCK, kernel_volume);
  get_indice_pairs_kernel_1<int32_t><<<grid_size, block_size>>>( // <scalar_t, int32_t, H_TILE, W_TILE>
      depth.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      thick.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
      NumIn.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      InRuleMap.packed_accessor32<int32_t, 5, torch::RestrictPtrTraits>(),
      OutRuleMap.packed_accessor32<int32_t, 5, torch::RestrictPtrTraits>(),
      CompactMap.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      N,
      H, W,
      KD, KH, KW,
      sD, sH, sW,
      padD, padH, padW,
      dD, dH, dW,
      oD, oH, oW);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printTensor_int(NumIn, "NumIn", 0, 0, H, 0, W);
  printTensor_k_int(InRuleMap, "InRuleMap", 0, 0, H, 0, W);
  printTensor_k_int(OutRuleMap, "OutRuleMap", 0, 0, H, 0, W);

  grid_size = dim3(divUp(oH, oH_BLOCK), divUp(oW, oW_BLOCK), 1);
  block_size = dim3(oH_BLOCK, oW_BLOCK, 1);

  get_indice_pairs_kernel_2<int32_t><<<grid_size, block_size>>>(
    CompactMap.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    new_depth.packed_accessor32<int32_t, 4, RestrictPtrTraits >(),
    new_thick.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
    N,
    kernel_volume,
    oH, oW, oD);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printTensor_int(new_depth, "new_depth", 0, 0, oH, 0, oW);
  printTensor_int(new_thick, "new_thick", 0, 0, oH, 0, oW);
  // std::cout << "CompactMap = " << CompactMap << std::endl;

  grid_size = dim3(divUp(H, H_BLOCK * 4), divUp(W, W_BLOCK * 4), 1);
  block_size = dim3(H_BLOCK * 4, W_BLOCK * 4, 1);

  get_indice_pairs_kernel_3<int32_t><<<grid_size, block_size>>>(
    CompactMap.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    N,
    H, W,
    kernel_volume,
    KD, KH, KW,
    sH, sW,
    padH, padW,
    dH, dW,
    oH, oW);


  int oT = torch::max(new_thick).item<int>();
  // only supported in pytorch 1.5
  // new_depth = new_depth.index({Ellipsis, Slice(0, oT), Ellipsis, Ellipsis}).contiguous();
  new_depth = new_depth.narrow(1, 0, oT).contiguous();

  // printf(" oT / oT_MAX = %d / %d\n", oT, oT_MAX );

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printTensor_k_int(OutRuleMap, "OutRuleMap after k3", 0, 0, H, 0, W);

  return {new_depth, new_thick, InRuleMap, OutRuleMap, NumIn};
}

///////
// return {depth, thick, InRuleMap, OutRuleMap, NumIn};
///////
std::vector<torch::Tensor>
get_indice_pairs_subm(torch::Tensor depth,
                      torch::Tensor thick,
                      int D,
                      int KD, int KH, int KW,
                      int sD, int sH, int sW,
                      int padD, int padH, int padW,
                      int dD, int dH, int dW,
                      int groups)
{
  int N = depth.size(0);
  int T = depth.size(1);
  int H = depth.size(2);
  int W = depth.size(3);
  // output sizes
  int oD, oH, oW;
  oD = D;
  oH = H;
  oW = W;

  const int H_BLOCK = 4, W_BLOCK = 4;
  auto kernel_volume = KD * KH * KW;

  dim3 grid_size, block_size;
  at::Tensor new_depth, new_thick;

  // count number of valid input voxel at (b,k,x,y)
  auto NumIn = torch::zeros({N, kernel_volume, H, W},
                   torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  // the thickness of the valid input voxel
  auto InRuleMap = torch::full({N, kernel_volume, H, W, T},
    /*value=*/ -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  // the output thickness of the valid input voxel
  auto OutRuleMap = torch::full({N, kernel_volume, H, W, T},
    /*value=*/ -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  //// create <del>hash</del>map
  // the final value of CompactMap, means
  // the output thick + 1 at output coordinate, (b, oX, oY, oZ)
  auto CompactMap = torch::full({N, oH, oW, oD}, 0,
                  torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  grid_size = dim3(divUp(H, H_BLOCK * 4), divUp(W, W_BLOCK * 4), 1);
  block_size = dim3(H_BLOCK * 4, W_BLOCK * 4, 1);
  get_indice_pairs_subm_kernel_1<int32_t><<<grid_size, block_size>>>(
      depth.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      thick.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
      CompactMap.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      N, H, W);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK));
  block_size = dim3(H_BLOCK, W_BLOCK, kernel_volume);
  get_indice_pairs_subm_kernel_2<int32_t><<<grid_size, block_size>>>(
      depth.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      thick.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
      NumIn.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      InRuleMap.packed_accessor32<int32_t, 5, torch::RestrictPtrTraits>(),
      OutRuleMap.packed_accessor32<int32_t, 5, torch::RestrictPtrTraits>(),
      CompactMap.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
      N,
      H, W,
      KD, KH, KW,
      sD, sH, sW,
      padD, padH, padW,
      dD, dH, dW,
      oD, oH, oW);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return {depth, thick, InRuleMap, OutRuleMap, NumIn};
}

template <typename Index>
__global__ void gather_kernel(
    const torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
      feature,
    const torch::PackedTensorAccessor32<Index , 5, RestrictPtrTraits>
      InRuleMap,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
      NumIn,
    torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
      inputBuffer,
      int N, int k, int iC,
      int H, int W)
{
  for (int x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    for (int y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {

      for (int b = 0; b < N; b++)
      {
        for (int i = 0; i < NumIn[b][k][x][y]; i++)
        {
          Index it = InRuleMap[b][k][x][y][i];
          for (int ic = threadIdx.z; ic < iC; ic += blockDim.z)
          {
            inputBuffer[b][i][x][y][ic] = feature[b][it][x][y][ic];
          }
        }
      }
    }
  }
}


// TODO: obviously can be optimized, by sharing oX, oY, offset
// TODO: output buffer underutilized when stride > 1
template <typename Index>
__global__ void gather_kernel_k(
    const torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
      featureOut,
    const torch::PackedTensorAccessor32<Index , 5, RestrictPtrTraits>
      OutRuleMap,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
      NumIn,
    torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
      outputBuffer,
      int N, int k, int oC,
      int H, int W,
      int k_H, int k_W,
      int sH, int sW,
      int padH, int padW,
      int dH, int dW,
      int oH, int oW
      )
{
  for (Index x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    Index oX = OutSpatial(k_H, x, sH, dH, padH);

    if (oX < 0 || oX >= oH) continue;
    for (Index y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {
      Index oY = OutSpatial(k_W, y, sW, dW, padW);
      if (oY < 0 || oY >= oW) continue;

      for (Index b = 0; b < N; b++)
      {
        for (Index i = 0; i < NumIn[b][k][x][y]; i++)
        {
          // if ( k == 9 && threadIdx.z == 0) {
          //   printf("outrulemap[0][9][%d][%d][%d] = %d\n ", x, y, i, OutRuleMap[b][k][x][y][i] );
          //   printf("NumIn[0][9][%d][%d] = %d\n ", x, y, NumIn[b][k][x][y]);
          //   // printf("featureOut[0][0][")
          // }
          Index ot = OutRuleMap[b][k][x][y][i];
          for (Index oc = threadIdx.z; oc < oC; oc += blockDim.z)
          {
            outputBuffer[b][i][x][y][oc] = featureOut[b][ot][oX][oY][oc];
          }
        }
      }
    }
  }
}


template <typename Index>
__global__ void scatter_add_kernel(
    const torch::PackedTensorAccessor32<Index , 5, RestrictPtrTraits>
      OutRuleMap,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
      NumIn,
    const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
      outputBuffer,
    torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
      output,
    int N, int oC, int k,
    int KD, int KH, int KW,
    int sD, int sH, int sW,
    int padD, int padH, int padW,
    int dD, int dH, int dW,
    int oH, int oW,
    int H, int W)
{

// TODO: move out these two
  Index k_H = (k / KW) % KH;
  Index k_W = k % KW;

  for (int x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    Index oX = OutSpatial(k_H, x, sH, dH, padH);
    if (oX < 0 || oX >= oH) continue;
    for (int y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {
      Index oY = OutSpatial(k_W, y, sW, dW, padW);
      if (oY < 0 || oY >= oW) continue;
      for (int b = 0; b < N; b++)
      {
        for (int i = 0; i < NumIn[b][k][x][y]; i++)
        {
          Index ot = OutRuleMap[b][k][x][y][i];
          for (int oc = threadIdx.z; oc < oC; oc += blockDim.z)
          {
            atomicAdd(&output[b][ot][oX][oY][oc], outputBuffer[b][i][x][y][oc]);
          }
        }
      }
    }
  }
}

template <typename Index>
__global__ void scatter_add_kernel_backward(
    const torch::PackedTensorAccessor32<Index , 5, RestrictPtrTraits>
      InRuleMap,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
      NumIn,
    const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
      inputBuffer,
    torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
      d_feature,
    int N, int iC, int k,
    int H, int W)
{

  for (int x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    for (int y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {
      for (int b = 0; b < N; b++)
      {
        for (int i = 0; i < NumIn[b][k][x][y]; i++)
        {
          Index it = InRuleMap[b][k][x][y][i];
          for (int ic = threadIdx.z; ic < iC; ic += blockDim.z)
          {
            atomicAdd(&d_feature[b][it][x][y][ic], inputBuffer[b][i][x][y][ic]);
          }
        }
      }
    }
  }
}




///////////////////////////////////////////////////////////////////
// indice conv
///////////////////////////////////////////////////////////////////

std::vector<torch::Tensor>
indice_conv_gemm(torch::Tensor feature,
                 torch::Tensor weight,
                 torch::Tensor InRuleMap,
                 torch::Tensor OutRuleMap,
                 torch::Tensor NumIn,
                 //  torch::Tensor bias,
                 int oT,
                 int sD, int sH, int sW,
                 int padD, int padH, int padW,
                 int dD, int dH, int dW,
                 int groups)
{
  int N = feature.size(0);
  int C = feature.size(1);
  int T = feature.size(2);
  int H = feature.size(3);
  int W = feature.size(4);
  int oC = weight.size(0);
  int iC = weight.size(1);
  int KD = weight.size(2);
  int KH = weight.size(3);
  int KW = weight.size(4);

  int oH = std::floor((H + 2 * padH - dH * (KH - 1) - 1) / sH + 1);
  int oW = std::floor((W + 2 * padW - dW * (KW - 1) - 1) / sW + 1);

  const int H_BLOCK = 4, W_BLOCK = 4;

  int kernel_volume = KD * KH * KW;

  // the output RangeVoxel
  auto new_feature = torch::zeros({N, oC, oT, oH, oW},
                                  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  auto filters = weight.permute({2, 3, 4, 1, 0}).contiguous().view({-1, iC, oC});

  auto options = torch::TensorOptions().dtype(feature.dtype()).device(feature.device());

  torch::Tensor output = torch::zeros({N, oT, oH, oW, oC}, options);
  torch::Tensor inputBuffer = torch::zeros({N, T, H, W, iC}, options);
  torch::Tensor outputBuffer = torch::zeros({N, T, H, W, oC}, options);
  torch::Tensor inputBufferGemm = inputBuffer.view({N * T * H * W, iC});
  torch::Tensor outputBufferGemm = outputBuffer.view({N * T * H * W, oC});

  torch::Tensor feature_ = feature.permute({0, 2, 3, 4, 1}).contiguous();

  for (int k = 0; k < kernel_volume; ++k)
  {
    inputBufferGemm.fill_(0);
    outputBufferGemm.fill_(0);

    gather_kernel<int32_t><<<grid_size, block_size>>>(
        feature_.packed_accessor32<float, 5, RestrictPtrTraits>(),
        InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        inputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, k, iC, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // std::cout << "inputBuffer = " << inputBuffer << std::endl;
    // std::cout << "inputbuffergemm = " << inputbuffergemm << std::endl;
    // std::cout << "filters[k] = " << filters[k] << std::endl;

    torch::mm_out(outputBufferGemm, inputBufferGemm, filters[k]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "outputBuffer = " << outputBuffer << std::endl;
    // std::cout << "outputBufferGemm = " << outputBufferGemm << std::endl;

    scatter_add_kernel<int32_t><<<grid_size, block_size>>>(
        OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        outputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        output.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, oC, k,
        KD, KH, KW,
        sD, sH, sW,
        padD, padH, padW,
        dD, dH, dW,
        oH, oW, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // std::cout << "output = " << output << std::endl;

  }
  new_feature = output.permute({0, 4, 1, 2, 3}).contiguous();

  return {new_feature};
}


std::vector<torch::Tensor>
indice_conv_backward_gemm(torch::Tensor feature,
                      torch::Tensor d_featureOut,
                      torch::Tensor weight,
                      // torch::Tensor bias,
                      torch::Tensor InRuleMap,
                      torch::Tensor OutRuleMap,
                      torch::Tensor NumIn,
                      int sD, int sH, int sW,
                      int padD, int padH, int padW,
                      int dD, int dH, int dW,
                      int groups, int subm)
{
  // auto d_bias = torch::zeros_like(bias);

  // input size
  int N = feature.size(0);
  int iT = feature.size(2);
  int H = feature.size(3);
  int W = feature.size(4);
  int oC = weight.size(0);
  int iC = weight.size(1);
  int KD = weight.size(2);
  int KH = weight.size(3);
  int KW = weight.size(4);

  const int H_BLOCK = 4, W_BLOCK = 4;
  auto kernel_volume = KD * KH * KW;

  dim3 grid_size, block_size;
  torch::Tensor new_depth, new_thick;

  // d_featureOut's shape = N, oC, oT, oH, oW
  int oT = d_featureOut.size(2);
  int oH = d_featureOut.size(3);
  int oW = d_featureOut.size(4);

  d_featureOut = d_featureOut.permute({0, 2, 3, 4, 1}).contiguous();
  feature = feature.permute({0, 2, 3, 4, 1}).contiguous();
  weight = weight.permute({2, 3, 4, 1, 0}).contiguous();
  weight = weight.view({-1, iC, oC});

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (oC > 4 && oC <= 8) {
    C_BLOCK = 8;
  } else if (oC > 8 && oC <= 16) {
     C_BLOCK = 16;
  } else if (oC > 16) {
    C_BLOCK = 32;
  }

  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  auto options = torch::TensorOptions().dtype(feature.dtype()).device(feature.device());
  auto d_feature = torch::zeros_like(feature);
  auto d_weight = torch::zeros_like(weight);

  torch::Tensor inputBuffer = torch::zeros({N, iT, H, W, iC}, options);
  torch::Tensor outputBuffer = torch::zeros({N, iT, H, W, oC}, options);

  torch::Tensor inputBufferGemm = inputBuffer.view({N * iT * H * W, iC});
  torch::Tensor outputBufferGemm = outputBuffer.view({N * iT * H * W, oC});

  d_weight = d_weight.view({-1, iC, oC});

  for (int k = 0; k < kernel_volume; ++k)
  {

    inputBufferGemm.fill_(0);
    outputBufferGemm.fill_(0);

    gather_kernel<int32_t><<<grid_size, block_size>>>(
        feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
        InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        inputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, k, iC, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int k_H = (k / KW) % KH;
    int k_W = k % KW;
    gather_kernel_k<int32_t><<<grid_size, block_size>>>(
        d_featureOut.packed_accessor32<float, 5, RestrictPtrTraits>(),
        OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        outputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, k, oC,
        H, W,
        k_H, k_W,
        sH, sW,
        padH, padW,
        dH, dW,
        oH, oW);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // iC, oC
    auto filterGradSub = d_weight[k];

    torch::mm_out(filterGradSub, inputBufferGemm.t(), outputBufferGemm);
    torch::mm_out(inputBufferGemm, outputBufferGemm, weight[k].t());

    scatter_add_kernel_backward<int32_t><<<grid_size, block_size>>>(
        InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        inputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        d_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, iC, k, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  d_weight = d_weight.view({KD, KH, KW, iC, oC}).permute({4, 3, 0, 1, 2}).contiguous();
  d_feature = d_feature.permute({0, 4, 1, 2, 3}).contiguous();

  return {d_feature, d_weight};
}


std::vector<torch::Tensor>
indice_conv(torch::Tensor feature,
            torch::Tensor weight,
            torch::Tensor InRuleMap,
            torch::Tensor OutRuleMap,
            torch::Tensor NumIn,
            //  torch::Tensor bias,
            int oT,
            int sD, int sH, int sW,
            int padD, int padH, int padW,
            int dD, int dH, int dW,
            int groups)
{
  int N, C, H, W, oC, KD, KH, KW;
  N = feature.size(0);
  C = feature.size(1);
  H = feature.size(3);
  W = feature.size(4);
  oC = weight.size(0);
  KD = weight.size(2);
  KH = weight.size(3);
  KW = weight.size(4);

  int oH = std::floor((H + 2 * padH - dH * (KH - 1) - 1) / sH + 1);
  int oW = std::floor((W + 2 * padW - dW * (KW - 1) - 1) / sW + 1);

  const int H_BLOCK = 4, W_BLOCK = 4;

  int kernel_volume = KD * KH * KW;

  // the output RangeVoxel
  auto new_feature = torch::zeros({N, oC, oT, oH, oW},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  indice_conv_kernel<int32_t><<<grid_size, block_size>>>(
    feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    new_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    weight.packed_accessor32<float, 5, RestrictPtrTraits>(),
    N, C, oC,
    kernel_volume,
    KD, KH, KW,
    sH, sW,
    padH, padW,
    dH, dW,
    oH, oW,
    H, W
  );

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  printTensor_int(NumIn, "NumIn final", 0, 0, H, 0, W);

  return {new_feature};
}



std::vector<torch::Tensor>
indice_conv_backward(torch::Tensor feature,
                      torch::Tensor d_featureOut,
                      torch::Tensor weight,
                      // torch::Tensor bias,
                      torch::Tensor InRuleMap,
                      torch::Tensor OutRuleMap,
                      torch::Tensor NumIn,
                      int sD, int sH, int sW,
                      int padD, int padH, int padW,
                      int dD, int dH, int dW,
                      int groups, int subm)
{
  auto d_feature = torch::zeros_like(feature);
  auto d_weight = torch::zeros_like(weight);
  // auto d_bias = torch::zeros_like(bias);

  // input size
  int N, H, W, oC, iC, KD, KH, KW;
  N = feature.size(0);
  H = feature.size(3);
  W = feature.size(4);
  oC = weight.size(0);
  iC = weight.size(1);
  KD = weight.size(2);
  KH = weight.size(3);
  KW = weight.size(4);


  const int H_BLOCK = 4, W_BLOCK = 4;
  auto kernel_volume = KD * KH * KW;

  dim3 grid_size, block_size;
  torch::Tensor new_depth, new_thick;

  // d_featureOut's shape = N, oC, oT, oH, oW
  int oT, oH, oW;
  oT = d_featureOut.size(2);
  oH = d_featureOut.size(3);
  oW = d_featureOut.size(4);

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (oC > 4 && oC <= 8) {
    C_BLOCK = 8;
  } else if (oC > 8 && oC <= 16) {
     C_BLOCK = 16;
  } else if (oC > 16) {
    C_BLOCK = 32;
  }

  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  // the output RangeVoxel
  auto new_feature = torch::zeros({N, oC, oT, oH, oW},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  indice_conv_backward_kernel<int32_t><<<grid_size, block_size>>>(
    d_featureOut.packed_accessor32<float, 5, RestrictPtrTraits>(),
    d_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    d_weight.packed_accessor32<float, 5, RestrictPtrTraits>(),
    InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    weight.packed_accessor32<float, 5, RestrictPtrTraits>(),
    feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    N, iC, oC,
    kernel_volume,
    KD, KH, KW,
    sH, sW,
    padH, padW,
    dH, dW,
    oH, oW,
    H, W);

  return {d_feature, d_weight};
}


template<typename Index>
__global__ void to_dense_kernel(
    const torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  feature,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
  depth,
    const torch::PackedTensorAccessor32<Index , 3, RestrictPtrTraits>
  thick,
    torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  buffer,
  int N, int C,
  int H, int W
)
{
  for (int x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    for (int y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {

      for (int b = 0; b < N; b++)
      {
        for (int t = 0; t < thick[b][x][y]; t++)
        {
          Index z = depth[b][t][x][y];

          for (int c = threadIdx.z; c < C; c += blockDim.z)
          {
            buffer[b][c][z][x][y] = feature[b][c][t][x][y];
          }
        }
      }
    }
  }
}


template<typename Index>
__global__ void to_dense_backward_kernel(
    const torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  d_featureOut,
    const torch::PackedTensorAccessor32<Index , 4, RestrictPtrTraits>
  depth,
    const torch::PackedTensorAccessor32<Index , 3, RestrictPtrTraits>
  thick,
    torch::PackedTensorAccessor32<float , 5, RestrictPtrTraits>
  d_feature,
  int N, int C,
  int H, int W
)
{
  for (int x = threadIdx.x + blockDim.x * blockIdx.x;
       x < H; x += blockDim.x * gridDim.x)
  {
    for (int y = threadIdx.y + blockDim.y * blockIdx.y;
         y < W; y += blockDim.y * gridDim.y)
    {

      for (int b = 0; b < N; b++)
      {
        for (int t = 0; t < thick[b][x][y]; t++)
        {

          Index z = depth[b][t][x][y];

          for (int c = threadIdx.z; c < C; c += blockDim.z)
          {
            d_feature[b][c][t][x][y] = d_featureOut[b][c][z][x][y];
          }

        }
      }
    }
  }
}


torch::Tensor
to_dense(torch::Tensor feature,
         torch::Tensor depth,
         torch::Tensor thick,
         int D)
{
  int N = feature.size(0);
  int C = feature.size(1);
  int H = feature.size(3);
  int W = feature.size(4);

  auto options = torch::TensorOptions().dtype(feature.dtype()).device(feature.device());
  torch::Tensor buffer = torch::zeros({N, C, D, H, W}, options);

  const int H_BLOCK = 4, W_BLOCK = 4;
  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  to_dense_kernel<int32_t><<<grid_size, block_size>>>(
      feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
      depth.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
      thick.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
      buffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
      N, C, H, W);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return buffer;
}

torch::Tensor
to_dense_backward(torch::Tensor d_featureOut,
                  torch::Tensor depth,
                  torch::Tensor thick,
                  int T)
{

  int N = d_featureOut.size(0);
  int C = d_featureOut.size(1);
  int H = d_featureOut.size(3);
  int W = d_featureOut.size(4);

  const int H_BLOCK = 4, W_BLOCK = 4;
  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  auto options = torch::TensorOptions().dtype(d_featureOut.dtype()).device(d_featureOut.device());
  auto d_feature = torch::zeros({N,C,T,H,W} , options);

  to_dense_backward_kernel<int32_t><<<grid_size, block_size>>>(
      d_featureOut.packed_accessor32<float, 5, RestrictPtrTraits>(),
      depth.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
      thick.packed_accessor32<int32_t, 3, RestrictPtrTraits>(),
      d_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
      N, C, H, W);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return d_feature;
}

} // namespace sphconv
