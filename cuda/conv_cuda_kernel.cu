
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

using Accessor = torch::GenericPackedTensorAccessor;
using torch::RestrictPtrTraits;

template <typename ElementType, typename Index, int H_TILE, int W_TILE,
          int NUM_KERNELS_PER_BLOCK, int BLOCK_SIZE>
__global__ void sphconv_cuda_forward_kernel(
    const Accessor<scalar_t, 5, RestrictPtrTraits, size_t> feature,
    const Accessor<scalar_t, 4, RestrictPtrTraits, size_t> depth,
    const Accessor<scalar_t, 3, RestrictPtrTraits, size_t> thick,
    const Accessor<scalar_t, 5, RestrictPtrTraits, size_t> weights,
    const Accessor<scalar_t, 1, RestrictPtrTraits, size_t> bias,
    int64_t N int64_t K, int64_t INPUT_TILE_H, int64_t INPUT_TILE_W) {
  // load tile to shared mem
  // input tile size = TILE + K
  //
  // __shared__ feature_tile[N][C][T][H_TILE][W_TILE];
  __shared__ scalar_t depth_tile[T][INPUT_TILE_H][INPUT_TILE_W];
  __shared__ scalar_t thick_tile[INPUT_TILE_H][INPUT_TILE_W];

  // index in shared mem

  // output index
  int oX = threadIdx.x + blockDim.x * W_TILE;
  int oY = threadIdx.y + blockDim.y * H_TILE;

  // left corner
  int iX = oX + blockIdx.x * H_TILE - K / 2 - padH;
  int iY = oY + blockIdx.y * W_TILE - K / 2 - padW;
  int dest = threadIdx.x + threadIdx.y * H_TILE;
  int destX = threadIdx.x;
  int dextY = threadIdx.y;
  while (destX < INPUT_TILE_H) {
    while (destY < INPUT_TILE_W) {
      int t = threadidx.z;
      while (t < T) {
        depth_tile[t][destX][destY] = depth[t][iX][iY];
        t += blockDim.z;
      }
      iY += blockDim.y;
      destY += blockDim.y;
    }
    iX += blockDim.x;
    dextX += blockDim.x;
  }

  __shared__ scalar_t depth_tile[T][INPUT_TILE_H][INPUT_TILE_W];

  // valid compute indices
  // general convolution
  // write out to
  for (int i = 0; i < K; i++){
      for (int j = 0; j < K; j++) {
          for (int k = 0; k < K; k++) {
            int i_out = blockDim.x;
            int j_out = blockDim.y;
            // TODO:
            int i_in = InLocal(i_out, sH, padH, K);
            int j_in = InLocal(j_out, sW, padD, K);
            // compare depth[:][i_out, j_out] and depth[:][i_in][j_in]
            // build cocking hash


          }
      }
  }

}

/***
* feature : N C T H W
* depth : N T H W
* thick : N H W
* weight : D C K K K
* bias : D
**/
std::vector<torch::Tensor> sphconv_cuda_forward(
    torch::Tensor feature,
    torch::Tensor depth,
    torch::Tensor thick,
    torch::Tensor weights,
    torch::Tensor bias,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dD, int64_t dH, int64_t dW)
{
    // input size
    int64_t N, C, T, H, W;
    N = feature.size(0);
    C = feature.size(1);
    T = feature.size(2);
    H = feature.size(3);
    W = feature.size(4);
    K = weights.size(2)

        // assume we want to have each block to calcultate T output elements
        // (T + k - 1)^* inpuut elements are neededd ,

        // tile size
        // the output tlie
        constexpr int H_TILE = 16,
    W_TILE = 16,

    oH = std::floor((H + 2 * padH - dH * (K - 1) - 1) / sH + 1);
    oW = std::floor((W + 2 * padW - dW * (K - 1) - 1) / sW + 1);

    auto divUp = [](int x, int y) { return (x + y - 1) / y; };
    // number of tiles

    dim3 grid_size, block_size;
    grid_size.x = divUp(oH, H_TILE);
    grid_size.y = divUp(oW, W_TILE);

    block_size.x = H_TILE;
    block_size.y = W_TILE;
    block_size.z = 128;

    INPUT_TILE_H = (H_TILE - 1) sH + dH * (K - 1) + 1;
    INPUT_TILE_W = (W_TILE - 1) sW + dW * (K - 1) + 1;

    feature.AT_DISPATCH_FLOATING_TYPES(
        feature.type(), "sphconv_forward_cuda", ([&] {
          sphconv_cuda_forward_kernel<scalar_t, H_TILE, W_TILE,
                                      TILIES_PER_BLOCK,
                                      BLOCK_SIZE><<<grid_size, block_size>>>(
              feature.generic_packed_accessor<
                  scallar_t, 5, torch::RestrictPtrTraits, size_t>(),
              depth.generic_packed_accessor<Index, 4, torch::RestrictPtrTraits,
                                            size_t>(),
              thick.generic_packed_accessor<Index, 3, torch::RestrictPtrTraits,
                                            size_t>(),
              weights.generic_packed_accessor<
                  scallar_t, 5, torch::RestrictPtrTraits, size_t>(),
              bias.generic_packed_accessor<scallar_t, 1,
                                           torch::RestrictPtrTraits, size_t>(),
              N, K, INPUT_TILE_H, INPUT_TILE_W, )
        }))
  }

  std::vector<torch::Tensor> sphconv_cuda_backward(
      torch::Tensor grad_feature, torch::Tensor depth, torch::Tensor thick,
      torch::Tensor grad_weight, torch::Tensor bias, torch::Tensor stride,
      torch::Tensor padding, torch::Tensor dilation, torch::Tensor groups);