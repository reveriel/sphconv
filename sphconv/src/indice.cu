#include <debug_utils.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "indice.cu.h"

namespace sphconv {

const int H_BLOCK = 4, W_BLOCK = 8;

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

}
