// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
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


// template <typename scalar_t, typename Index>
template <typename Index>
__global__ void sphconv_cuda_forward_kernel_1(
    // const torch::GenericPackedTensorAccessor<scalar_t, 5, RestrictPtrTraits, size_t>
    //     feature,
    const torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
        depth,
    const torch::GenericPackedTensorAccessor<Index, 3, RestrictPtrTraits, size_t>
        thick,
    torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
      NumIn,
    // torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
    //   NumOut,
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
  // input index
  // x y z ~ H W D
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  // printf("threadIdx.x(%d) + blockDim.x(%d) * blockIdx.x(%d) = x(%d)\n", threadIdx.x, blockDim.x, blockIdx.x, x);
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;
  // printf("x, y, k =  %d, %d, %d\n", x, y, k);
  int kD = k / (KH * KW);
  int kH = (k / KW) % KH;
  int kW = k % KW;
  // printf("k0, k1, k2 (k) = %d, %d, %d, (%d)\n", k0, k1, k2, k);

  for (int b = 0; b < N; b++) {
    for (int t = 0; t < thick[b][x][y]; t++)
    {
      Index z = depth[b][t][x][y];
      Index oX = OutSpatial(kH, x, sH, dH, padH);
      Index oY = OutSpatial(kW, y, sW, dW, padW);
      Index oZ = OutSpatial(kD, z, sD, dD, padD);
      printf("oX, oY, oZ =  %d, %d, %d\n", oX, oY, oZ);
      if (oX >= 0 && oX < oH && oY >= 0 && oY < oW  && oZ >= 0 && oZ < oD)
      {
        Index i = atomicAdd(&NumIn[b][k][x][y], Index(1));
        // printf("InRuleMap i = %d\n", i);
        // if (i >= 32) continue;
        InRuleMap[b][k][x][y][i] = t;
        // Index j = atomicAdd(&NumOut[b][k][oX][oY], Index(1));
        // printf("OutRuleMap j = %d\n", j);
        // if (j >= 32) continue;
        // OutRuleMap[b][k][oX][oY][j] = oZ;
        OutRuleMap[b][k][x][y][i] = oZ;
      } // if
    } // for t
  }// for b
}


template <typename Index>
__global__ void sphconv_cuda_forward_kernel_2(
    torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
      CompactMap,
    torch::GenericPackedTensorAccessor<Index, 5, RestrictPtrTraits, size_t>
      OutRuleMap,
    torch::GenericPackedTensorAccessor<Index, 5, RestrictPtrTraits, size_t>
      thick,
    torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
      NumIn,
      int N,
      int kernel_volume,
      )
{
  int oX = threadIdx.x + blockDim.x * blockIdx.x;
  int oY = threadIdx.y + blockDim.y * blockIdx.y;

  // fill compact map with 1
  for (int b = 0; b < N; b++) {
    for (int i = 0; i < NumIn[b][k][x][y]; i++) {
      for (int k = 0; k < kernel_volume; k++) {
        int oZ = OutRuleMap[b][k][oX][oY][i];
        CompactMap[b][oX][oY][oZ] = 1;
      }
    }
  }

  // scan (prefix sum) on CompactMap
  for (int b = 0; b < N; b++) {
    for (int i = 0; i < NumIn[b][k][x][y]; i++) {
      for (int z = 1; z < oD; z++) {

        CompactMap[b][oX][oY][z+1] += CompactMap[b][oX][oY][z];
      }
    }
  }

}

template <typename Index>
__global__ void sphconv_cuda_forward_kernel_3(
    const torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
      CompactMap,
    torch::GenericPackedTensorAccessor<Index, 5, RestrictPtrTraits, size_t>
      OutRuleMap,
    const torch::GenericPackedTensorAccessor<Index, 4, RestrictPtrTraits, size_t>
      NumIn,
    int kernel_volume,
    int KD, int kH, int kW,
    int sH, int sW,
    int padH, int padW,
    int dH, int dW
)
{

  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;

  // re assign OutRuleMap
  for (int b = 0; b < N; b++) {

    for (int k = 0; k < kernel_volume; k++) {

      // get oX and oY
      int kD = k / (KH * KW);
      int kH = (k / KW) % KH;
      int kW = k % KW;
      Index oX = OutSpatial(kH, x, sH, dH, padH);
      Index oY = OutSpatial(kW, y, sW, dW, padW);

      for (int i = 0; i < NumIn[b][k][x][y]; i++) {
        Index oZ = OutRuleMap[b][k][x][y][i];

        OutRuleMap[b][k][x][y][i] = CompactMap[b][oX][oY][oZ];
      }
    }
  }
}



template <typename Index>
__global__ void sphconv_cuda_forward_kernel_4(
  torch::GenericPackedTensorAccessor<float, 5, RestrictPtrTraits, size_t>
    feature,
  torch::GenericPackedTensorAccessor<float, 5, RestrictPtrTraits, size_t>
    new_feature,
  torch::GenericPackedTensorAccessor<Index, 5, RestrictPtrTraits, size_t>
    InRuleMap,
  torch::GenericPackedTensorAccessor<Index, 5, RestrictPtrTraits, size_t>
    OutRuleMap,
  int in_channels,
  int out_channels,

) {
  // gather
  // InRuleMap
  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;
  // int k = threadIdx.z + blockDim.z * blockIdx.z;
  // out put channel
  int oc = threadIdx.z;


  for (int b = 0; b < N; b++) {
    for (int k = 0; k < kernel_volume ; k++) {

      int kD = k / (KH * KW);
      int kH = (k / KW) % KH;
      int kW = k % KW;

      Index oX = OutSpatial(kH, x, sH, dH, padH);
      Index oY = OutSpatial(kW, y, sW, dW, padW);

      for (int ic = 0; ic < in_channels; ic++) {
        for (int i =0; i < NumIn[b][k][x][y]; i++) {
          while (oc < out_channels) {

            // input thick
            it = InRuleMap[b][k][x][y][i];
            // output thick
            ot = OutRuleMap[b][k][x][y][i];

            new_feature[b][oc][ot][oX][oY] = weight[oc][ic][kD][kH][kW] * feature[b][ic][it][x][y];
            oc += blockDim.z;
          }
        }
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

  printf("input spatial shape (D,H,W) = %d, %d, %d\n", D, H, W);
  printf("output spatial shape (oD,oH,oW) = %d, %d, %d\n", oD, oH, oW);

  auto divUp = [](int x, int y) { return (x + y - 1) / y; };
  // number of tiles

  const int H_BLOCK = 4, W_BLOCK = 4;
  dim3 grid_size, block_size;
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
                                torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  auto new_depth = torch::empty({N, T + 2, oH, oW},
                                 torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  auto new_thick = torch::empty({N, oH, oW},
                                torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  auto NumIn = torch::full({N, kernel_volume, H, W},
     /*value=*/ 0, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  auto NumOut = torch::full({N, kernel_volume, oH, oW},
     /*value=*/ 0, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  auto InRuleMap = torch::full({N, kernel_volume, H, W, T},
    /*value=*/ -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  auto OutRuleMap = torch::full({N, kernel_volume, oH, oW, (T * sD * sH * sW)},
    /*value=*/ -1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  printf("launch <<< %dx%dx%d, %dx%dx%d>>>\n", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z );

  sphconv_cuda_forward_kernel_1<int32_t><<<grid_size, block_size>>>( // <scalar_t, int32_t, H_TILE, W_TILE>
      // feature.generic_packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
      depth.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
      thick.generic_packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      // weight.generic_packed_accessor<float, 5, torch::RestrictPtrTraits, size_t>(),
      // bias.generic_packed_accessor<float, 1, torch::RestrictPtrTraits, size_t>(),
      // new_feature.generic_packed_accessor<float, 5, torch::RestrictPtrTraits, size_t>(),
      // new_depth.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
      // new_thick.generic_packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),
      NumIn.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
      NumOut.generic_packed_accessor<int32_t, 4, torch::RestrictPtrTraits, size_t>(),
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


  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());


  //// create <del>hash</del>map
  auto CompactMap = torch::full({N, H, W, D},
     /*value=*/-1, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

  const int  oH_BLOCK = 8, oW_BLOCK = 32;

  grid_size.x = divUp(oH, oH_BLOCK);
  grid_size.y = divUp(oW, oW_BLOCK);
  grid_size.z = 1;

  block_size.x = oH_BLOCK;
  block_size.y = oW_BLOCK;
  block_size.z = 1;

  sphconv_cuda_forward_kernel_2<int32_t><<<grid_size, block_size>>>(
    CompactMap.generic_packed_accessor<int32_t, 4, RestrictPtrTraits, size_t>(),
    OutRuleMap.generic_packed_accessor<int32_t, 5, RestrictPtrTraits, size_t>(),
    thick.generic_packed_accessor<int32_t, 3, RestrictPtrTraits, size_t(),
    NumOut.generic_packed_accessor<int32_t, 4, RestrictPtrTraits, size_t(),
    N,
    kernel_volume
  );



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