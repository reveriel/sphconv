#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#define Accesor torch::GenericPackedTensorAccessor
using torch::RestrictPtrTraits;

template <typename Index>
__device__ Index OutSpatial(Index k, Index x, Index s, Index d, Index pad)
{
  // forgive me. do nothing with the dillation
  // TODO
  return (x + pad - k)/ s;
}

__global__ void helloCUDA(float f)
{
  printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}



// template <typename scalar_t, typename Index>
#define scalar_t float
#define Index int32_t

__global__ void sphconv_cuda_forward_kernel(
    // const torch::GenericPackedTensorAccessor<scalar_t, 5, RestrictPtrTraits, size_t>
    //     feature,
    const torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
        depth,
    const torch::GenericPackedTensorAccessor<Index, 3, RestrictPtrTraits, size_t>
        thick,
    // const torch::GenericPackedTensorAccessor<scalar_t, 5, RestrictPtrTraits, size_t>
        // weight,
    // const torch::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t>
    //     bias,
    // torch::GenericPackedTensorAccessor<scalar_t, 5, RestrictPtrTraits, size_t>
    //     new_feature,
    // torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
    //     new_depth,
    // torch::GenericPackedTensorAccessor<Index, 3, RestrictPtrTraits, size_t>
    //     new_thick,
        // i = AtomicAdd(NumIn[b, k, x, y]);/
    torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
      NumIn,
    torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
      NumOut,
    torch::GenericPackedTensorAccessor<Index, 5, RestrictPtrTraits, size_t>
      InRuleMap,
    torch::GenericPackedTensorAccessor<Index, 5, RestrictPtrTraits, size_t>
      OutRuleMap,
    int N,
    int KD, int KH, int KW,
    // int INPUT_TILE_H, int INPUT_TILE_W,
    int sD, int sH, int sW,
    int padD, int padH, int padW,
    int dD, int dH, int dW,
    int oD, int oH, int oW
  )
{
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

  ///// load tile to shared mem

  // input index
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  printf("x, y, k =  %d, %d, %d\n", x, y, k);
  int k0 = k /( KH * KW);
  int k1 = (k / KW) % KH;
  int k2 = k % KW;

  for (int b = 0; b < N; b++) {
    for (int t = 0; t < thick[b][x][y]; t++)
    {
      Index z = depth[b][t][x][y];
      Index oX = OutSpatial(k0, x, sD, dD, padD);
      Index oY = OutSpatial(k1, y, sH, dH, padH);
      Index oZ = OutSpatial(k2, z, sW, dW, padW);
      printf("oX, oY, oZ =  %d, %d, %d\n", oX, oY, oZ);
      if (oX >= 0 && oX < oH && oY >= 0 && oY < oW  && oZ >= 0 && oZ < oD)
      {
        Index i = atomicAdd(&NumIn[b][k][x][y], Index(1));
        InRuleMap[b][k][x][y][i] = t;
        Index j = atomicAdd(&NumOut[b][k][oX][oY], Index(1));
        OutRuleMap[b][k][oX][oY][j] = oZ;
      } // if
    } // for t
  }// for b
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
                     int64_t sD, int64_t sH, int64_t sW,
                     int64_t padD, int64_t padH, int64_t padW,
                     int64_t dD, int64_t dH, int64_t dW,
                     int64_t groups,
                     int64_t D
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
  KH = weight.size(3);
  KW = weight.size(4);

  // assume we want to have each block to calcultate T output elements
  // (T + k - 1)^* inpuut elements are neededd ,

  // tile size
  // the output tlie
  // constexpr int H_TILE = 16, W_TILE = 16;

  int oD = std::floor((D + 2 * padD - dD * (KD - 1) - 1) / sD + 1);
  int oH = std::floor((H + 2 * padH - dH * (KH - 1) - 1) / sH + 1);
  int oW = std::floor((W + 2 * padW - dW * (KW - 1) - 1) / sW + 1);

  auto divUp = [](int x, int y) { return (x + y - 1) / y; };
  // number of tiles

  const int H_BLOCK = 4, W_BLOCK = 4;
  dim3 grid_size, block_size;
  // grid_size.x = divUp(oH, H_TILE);
  // grid_size.y = divUp(oW, W_TILE);
  grid_size.x = divUp(H, H_BLOCK);
  grid_size.y = divUp(W, W_BLOCK);
  grid_size.z = 1;

  block_size.x = H_BLOCK;
  block_size.y = W_BLOCK;
  auto kernel_volume = KD * KH * KW;
  block_size.z = kernel_volume;

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
  auto numIn = torch::full({N, kernel_volume, H, W},
     /*value=*/ -1, torch::dtype(torch::kInt32));
  auto numOut = torch::full({N, kernel_volume, oH, oW},
     /*value=*/ -1, torch::dtype(torch::kInt32));

  auto InRuleMap = torch::full({N, kernel_volume, H, W, 2},
    /*value=*/ -1, torch::dtype(torch::kInt32));
  auto OutRuleMap = torch::full({N, kernel_volume, oH, oW, 8},
    /*value=*/ -1, torch::dtype(torch::kInt32));

    printf("launch <<< %dx%dx%d, %dx%dx%d>>>\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z );

  // AT_DISPATCH_FLOATING_TYPES(
      // feature.type(), "sphconv_forward_cuda", ([&] {
        sphconv_cuda_forward_kernel<<<grid_size, block_size>>>( // <scalar_t, int32_t, H_TILE, W_TILE>
            // feature.generic_packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
            depth.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
            thick.generic_packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
            // weight.generic_packed_accessor<float, 5, torch::RestrictPtrTraits, size_t>(),
            // bias.generic_packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
            // new_feature.generic_packed_accessor<float, 5, torch::RestrictPtrTraits, size_t>(),
            // new_depth.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
            // new_thick.generic_packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
            numIn.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
            numOut.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
            InRuleMap.generic_packed_accessor<int32_t, 5, torch::RestrictPtrTraits, size_t>(),
            OutRuleMap.generic_packed_accessor<int32_t, 5, torch::RestrictPtrTraits, size_t>(),
            N,
            KD, KH, KW,
            // INPUT_TILE_H, INPUT_TILE_W,
            sD, sH, sW,
            padD, padH, padW,
            dD, dH, dW,
            oD, oH, oW
          );
      // }));

    // helloCUDA<<<10,10>>>(123.4f);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

  return {new_feature, new_depth, new_thick};
}

std::vector<torch::Tensor>
sphconv_cuda_backward(torch::Tensor feature,
                      torch::Tensor depth,
                      torch::Tensor thick,
                      torch::Tensor gradOutput,
                      torch::Tensor weight,
                      // torch::Tensor bias,
                      int64_t sD, int64_t sH, int64_t sW,
                      int64_t padD, int64_t padH, int64_t padW,
                      int64_t dD, int64_t dH, int64_t dW,
                      int64_t groups)
{
  auto d_feature = torch::zeros_like(feature);
  auto d_weight = torch::zeros_like(weight);
  // auto d_bias = torch::zeros_like(bias);

  return {d_feature, d_weight};
}