
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#define Accesor torch::GenericPackedTensorAccessor
using torch::RestrictPtrTraits;

template<typename Index>
__device__ Index OutSpatial(int k, int x, int s, int d, int pad)
{
  // forgive me. do nothing with the dillation
  // TODO
  return (x + pad - k)/ s;
}


template <typename scalar_t, typename Index, int H_TILE, int W_TILE>
__global__ void sphconv_cuda_forward_kernel(
    const torch::GenericPackedTensorAccessor<scalar_t, 5, RestrictPtrTraits, size_t>
        feature,
    const torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
        depth,
    const torch::GenericPackedTensorAccessor<Index, 3, RestrictPtrTraits, size_t>
        thick,
    const torch::GenericPackedTensorAccessor<scalar_t, 5, RestrictPtrTraits, size_t>
        weight,
    // const torch::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t>
    //     bias,
    torch::GenericPackedTensorAccessor<scalar_t, 5, RestrictPtrTraits, size_t>
        new_feature,
    torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
        new_depth,
    torch::GenericPackedTensorAccessor<Index, 3, RestrictPtrTraits, size_t>
        new_thick,
    int N, int DK, int HK, int WK,
    // int INPUT_TILE_H, int INPUT_TILE_W,
    int sD, int sH, int sW,
    int padD, int padH, int padW,
    int dD, int dH, int dW
  )
{
  // input tile size = TILE + K
  //
  // __shared__ feature_tile[N][C][T][H_TILE][W_TILE];
  // TODO: make T varadic length

  constexpr int T = 8;
  // const int INPUT_TILE_H = 4;
  // const int INPUT_TILE_W = 16;
  // __shared__ scalar_t depth_tile[T][INPUT_TILE_H][INPUT_TILE_W];
  // TODO
  // __shared__ Index thickMap[INPUT_TILE_H][INPUT_TILE_W];
  // __shared__ Index Num[3x3x3][INPUT_TILE_H][INPUT_TILE_W];

  ///// load tile to shared mem

  // input index
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  int k0 = k /( kH * KW);
  int k1 = (k / KW) % kH;
  int k2 = k % KW;


  for (int b = 0; b < N; b++) {
    for (int t = 0; t < ThickMap[x][y]; t++) {
      z = depth[t][x][y];
      oX = OutSpatial(k0, x, sD, dD, padD);
      oY = OutSpatial(k1, y, sH, dH, padH);
      oZ = OutSpatial(k2, z, sW, dW, padW);
      printf("oX, oY, oZ =  %d, %d, %d", oX, oY, oZ);
      i = AtomicAdd(Num[ k, x, y]);
      InRuleMap[k, x, y, i] = t;
      OutRuleMap[k, oX, oY, i] = oZ;
    }
  }


  // output index
  // int oX = threadIdx.x + blockDim.x * W_TILE;
  // int oY = threadIdx.y + blockDim.y * H_TILE;

  // left corner
  // int iX = oX + blockIdx.x * H_TILE - K / 2 - padH;
  // int iY = oY + blockIdx.y * W_TILE - K / 2 - padW;
  // int dest = threadIdx.x + threadIdx.y * H_TILE;
  // int destX = threadIdx.x;
  // int destY = threadIdx.y;
  // while (destX < INPUT_TILE_H) {
  //   while (destY < INPUT_TILE_W) {
  //     int t = threadIdx.z;
  //     while (t < T) {
  //       // depth_tile[t][destX][destY] = depth[t][iX][iY];
  //       t += blockDim.z;
  //     }
  //     iY += blockDim.y;
  //     destY += blockDim.y;
  //   }
  //   iX += blockDim.x;
  //   destX += blockDim.x;
  // }


  // valid compute indices
  // general convolution
  // write out to
  // for (int i = 0; i < K; i++) {
  //   for (int j = 0; j < K; j++) {
  //     for (int k = 0; k < K; k++) {
  //       int i_out = blockDim.x;
  //       int j_out = blockDim.y;
  //       // TODO:
  //       // int i_in = InLocal(i_out, sH, padH, K);
  //       // int j_in = InLocal(j_out, sW, padD, K);
  //       // compare depth[:][i_out, j_out] and depth[:][i_in][j_in]
  //       // build cocking hash
  //     }
  //   }
  // }
}

/***
 * feature : N C T H W
 * depth : N T H W
 * thick : N H W
 * weight : D C K K K
 * bias : D
 **/
std::vector<torch::Tensor>
sphconv_cuda_forward(torch::Tensor feature,
                     torch::Tensor depth,
                     torch::Tensor thick,
                     torch::Tensor weight,
                    //  torch::Tensor bias,
                     int sD, int sH, int sW,
                     int padD, int padH, int padW, int dD,
                     int dH, int dW,
                     int groups
                    )
{
  // input size
  int N, C, T, H, W, oC, KD, KH, KW;
  N = feature.size(0);
  C = feature.size(1);
  T = feature.size(2);
  H = feature.size(3);
  W = feature.size(4);
  oC = weight.size(0);
  KD = weight.size(2);
  kH = weight.size(3);
  kW = weight.size(4);

  // assume we want to have each block to calcultate T output elements
  // (T + k - 1)^* inpuut elements are neededd ,

  // tile size
  // the output tlie
  constexpr int H_TILE = 16, W_TILE = 16;

  int oH = std::floor((H + 2 * padH - dH * (K - 1) - 1) / sH + 1);
  int oW = std::floor((W + 2 * padW - dW * (K - 1) - 1) / sW + 1);

  auto divUp = [](int x, int y) { return (x + y - 1) / y; };
  // number of tiles

  const int H_BLOCK = 4, W_TILE = 8;
  dim3 grid_size, block_size;
  // grid_size.x = divUp(oH, H_TILE);
  // grid_size.y = divUp(oW, W_TILE);
  grid_size.x = divUp(H, H_BLOCK);
  grid_size.y = divUp(W, W_BLOCK);
  gird_size.z = 1

  block_size.x = H_BLOCK;
  block_size.y = W_BLOCK;
  block_size.z = KD * KH * KW;

  // int INPUT_TILE_H = (H_TILE - 1) * sH + dH * (K - 1) + 1; // TODO: check
  // int INPUT_TILE_W = (W_TILE - 1) * sW + dW * (K - 1) + 1;

  // output tensor
  // including new_feature
  auto new_feature = torch::empty({N, oC, T + 2, oH, oW},
                                torch::dtype(torch::kFloat32));
  auto new_depth = torch::empty({N, T + 2, oH, oW},
                                 torch::dtype(torch::kInt32));
  auto new_thick = torch::empty({N, oH, oW},
                                torch::dtype(torch::kInt32));

  AT_DISPATCH_FLOATING_TYPES(
      feature.type(), "sphconv_forward_cuda", ([&] {
        sphconv_cuda_forward_kernel
        <scalar_t, int32_t, H_TILE, W_TILE>
        <<<grid_size, block_size>>>(
            feature.generic_packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
            depth.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
            thick.generic_packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
            weight.generic_packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
            // bias.generic_packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            new_feature.generic_packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
            new_depth.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
            new_thick.generic_packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
            N, DK, HK, WK
            // INPUT_TILE_H, INPUT_TILE_W,
            sD, sH, sW,
            padD, padH, padW,
            dD, dH, dW
          );
      }));

  return {new_feature, new_depth, new_thick};
}

std::vector<torch::Tensor>
sphconv_cuda_backward(torch::Tensor feature,
                      torch::Tensor depth,
                      torch::Tensor thick,
                      torch::Tensor gradOutput,
                      torch::Tensor weight,
                      // torch::Tensor bias,
                      int sD, int sH, int sW,
                      int padD, int padH, int padW,
                      int dD, int dH, int dW,
                      int groups)
{
  auto d_feature = torch::zeros_like(feature);
  auto d_weight = torch::zeros_like(weight);
  // auto d_bias = torch::zeros_like(bias);

  return {d_feature, d_weight};
}