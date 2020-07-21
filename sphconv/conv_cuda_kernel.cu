// #define DEBUG

#ifdef DEBUG
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
#else
#define gpuErrchk(ans)
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define isThread(thread_x, thread_y, thread_z, block_x, block_y, block_z) \
  (threadIdx.x  == thread_x && threadIdx.y == thread_y && threadIdx.z == thread_z \
  && blockIdx.x == block_x && blockIdx.y == block_y && blockIdx.z == block_z)




#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

using torch::RestrictPtrTraits;

template <typename Index>
__device__ __inline__ Index OutSpatial(Index k, Index x, Index s, Index d, Index pad)
{
  // forgive me. do nothing with the dillation
  // TODO
  if ((x + pad - k) % s == 0)
    return (x + pad - k)/ s;
  return -1;
}

// Print a some tile, for debugging.
template <typename T>
void printTensor(at::Tensor A, std::string name, int batch_idx, int H_start,
                 int H_end, int W_start, int W_end) {

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== BEGIN" << std::endl;

  int dim = A.dim();
  if (dim == 3) {

    for (int x = H_start; x < H_end; x++) {
      std::cout << "  [ ";
      for (int y = W_start; y < W_end - 1; y++) {
        std::cout << A[batch_idx][x][y].item<T>() << ", ";
      }
      std::cout << A[batch_idx][x][W_end - 1].item<T>() << " ]" << std::endl;
    }
  } else if (dim == 4) {
    int size1 = A.size(1);
    for (int i = 0; i < size1; i++) {
      std::cout << " [ == " << i << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end - 1; y++) {
          std::cout << A[batch_idx][i][x][y].item<T>() << ", ";
        }
        std::cout << A[batch_idx][i][x][W_end - 1].item<T>() << " ]" << std::endl;
      }
      std::cout << " ] ==" << i << std::endl;
    }
  } else if (dim == 5) {
    // we only print dim == 4 tensor, print channel 0
    int size1 = A.size(1);
    for (int i = 0; i < size1; i++) {
      std::cout << " [ ==" << i << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end - 1; y++) {
          std::cout << A[batch_idx][0][i][x][y].item<T>() << ", ";
        }
        std::cout << A[batch_idx][0][i][x][W_end - 1].item<T>() << " ]";
      }
      std::cout << " ] ==" << i << std::endl;
    }
  }

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== END" << std::endl;
}

// Print a some tile, for debugging.
// the second dimension is kernel
template <typename T>
void printTensor_k(at::Tensor A, std::string name, int batch_idx, int H_start,
                 int H_end, int W_start, int W_end) {

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== BEGIN" << std::endl;

  int dim = A.dim();
  if (dim == 4) {
    int kernel_volume = A.size(1);
    for (int k = 0; k < kernel_volume; k++) {
      std::cout << " [ == " << k << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end - 1; y++) {
          std::cout << A[batch_idx][k][x][y].item<T>() << ", ";
        }
        std::cout << A[batch_idx][k][x][W_end - 1].item<T>() << " ]" << std::endl;
      }
      std::cout << " ] ==" << k << std::endl;
    }
  } else if (dim == 5) {
    // we only print dim == 4 tensor, print channel 0
    int kernel_volume = A.size(1);
    int len = A.size(4);
    for (int k = 0; k < kernel_volume; k++) {
      std::cout << " [ ==" << k << std::endl;
      for (int x = H_start; x < H_end; x++) {
        std::cout << "   [ ";
        for (int y = W_start; y < W_end ; y++) {
          std::cout << "(";
          for (int i = 0; i < A.size(4); i++ ) {
            std::cout << A[batch_idx][k][x][y][i].item<T>() << " ";
          }
          std::cout << "),";
        }
        std::cout <<  " ]";
      }
      std::cout << " ] ==" << k << std::endl;
    }
  }

  std::cout << "======= Tensor \"" << name << "\" at batch " << batch_idx
            << " ======== END" << std::endl;
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
  // printf("threadIdx.x(%d) + blockDim.x(%d) * blockIdx.x(%d) = x(%d)\n", threadIdx.x, blockDim.x, blockIdx.x, x);
  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;
  Index k = threadIdx.z + blockDim.z * blockIdx.z;

  if (x >= H || y >= W ) return;

  // printf("x, y, k =  %d, %d, %d\n", x, y, k);
  Index k_D = k / (KH * KW);
  Index k_H = (k / KW) % KH;
  Index k_W = k % KW;
  // printf("k0, k1, k2 (k) = %d, %d, %d, (%d)\n", k0, k1, k2, k);

  for (int b = 0; b < N; b++) {
    for (int t = 0; t < thick[b][x][y]; t++)
    {
      Index z = depth[b][t][x][y];
      Index oX = OutSpatial(k_H, x, sH, dH, padH);
      Index oY = OutSpatial(k_W, y, sW, dW, padW);
      Index oZ = OutSpatial(k_D, z, sD, dD, padD);
      // printf("oX, oY, oZ =  %d, %d, %d\n", oX, oY, oZ);
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
__global__ void sphconv_cuda_forward_kernel_2(
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

template <typename Index>
__global__ void sphconv_cuda_forward_kernel_3(
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
    int oH, int oW
)
{

  Index x = threadIdx.x + blockDim.x * blockIdx.x;
  Index y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= H || y >= W) return;

  // re-assign OutRuleMap
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


template <typename Index>
__global__ void sphconv_cuda_forward_kernel_4(
  const torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    feature,
  torch::PackedTensorAccessor32<float, 5, RestrictPtrTraits>
    new_feature,
  const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
    InRuleMap,
  const torch::PackedTensorAccessor32<Index, 5, RestrictPtrTraits>
    OutRuleMap,
  torch::PackedTensorAccessor32<Index, 4, RestrictPtrTraits>
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

          // if (oX == 0 && oY == 2) {
          //   printf("x, y = %d, %d, k_D, k_H, k_W, k =%d,%d,%d,%d\n",
          //   x,y, k_D, k_H, k_W, k);
          //   printf("NumIn[%d][%d][%d][%d] = %d\n",
          //   b, k ,x, y, NumIn[b][k][x][y]);
          // }
          Index oc = threadIdx.z;
          while (oc < out_channels) {

            // input thickness
            int it = InRuleMap[b][k][x][y][i];
            // output thickness
            int ot = OutRuleMap[b][k][x][y][i];

            // printf("new_feature[%d][%d][%d][%d][%d] += "
            //        "weight[%d][%d][%d][%d][%d] * feature[%d][%d][%d][%d][%d]\n",
            //        b, oc, ot, oX, oY, oc, ic, k_D, k_H, k_W, b, ic, it, x, y);

            atomicAdd(&new_feature[b][oc][ot][oX][oY], weight[oc][ic][k_D][k_H][k_W] * feature[b][ic][it][x][y]);
            // new_feature[b][oc][ot][oX][oY] +=
            //     weight[oc][ic][k_D][k_H][k_W] * feature[b][ic][it][x][y];

            oc += blockDim.z;
          }// while
        } // for i
      } // for ic
      // NumIn[b][k][x][y] = 0;

      // __syncthreads();
    }
  }
}


/***
 * feature : N C T H W
 * depth : N T H W
 * thick : N H W
 * weight : oC iC K K K
 * bias : oC
 **/
std::vector<torch::Tensor>
conv_cuda_forward(torch::Tensor feature,
                     torch::Tensor depth,
                     torch::Tensor thick,
                     torch::Tensor weight,
                    //  torch::Tensor bias,
                     int64_t sD, int64_t sH, int64_t sW,
                     int64_t padD, int64_t padH, int64_t padW,
                     int64_t dD, int64_t dH, int64_t dW,
                     int64_t groups,
                     int64_t D, int64_t subm
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

#ifdef DEBUG
  printf("input spatial shape (D,H,W) = %d, %d, %d\n", D, H, W);
  printf("output spatial shape (oD,oH,oW) = %d, %d, %d\n", oD, oH, oW);
#endif

  // auto divUp = [](int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); };

  auto divUp = [](int x, int y) { return (x + y - 1) / y; };

  // number of tiles

  const int H_BLOCK = 4, W_BLOCK = 4;
  dim3 grid_size, block_size;
  grid_size.x = divUp(H, H_BLOCK);
  grid_size.y = divUp(W, W_BLOCK);
  grid_size.z = 1;

  auto kernel_volume = KD * KH * KW;

  block_size.x = H_BLOCK;
  block_size.y = W_BLOCK;
  block_size.z = kernel_volume;

  // int INPUT_TILE_H = (H_TILE - 1) * sH + dH * (K - 1) + 1; // TODO: check
  // int INPUT_TILE_W = (W_TILE - 1) * sW + dW * (K - 1) + 1;

  // output tensor
  // including new_feature

  // int oT = T + 8 ; // This is bad
  int oT = T * 3 * 9;  // This is even worse

  // the output RangeVoxel
  auto new_feature = torch::zeros({N, oC, oT, oH, oW},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  auto new_depth = torch::zeros({N, oT, oH, oW},
                                 torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
  auto new_thick = torch::zeros({N, oH, oW},
                                torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

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
  auto CompactMap = torch::full({N, oH, oW, oD}, 0,
                  torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

#ifdef DEBUG
  printTensor<int>(depth, "depth", 0, 0, H, 0, W);
  printTensor<int>(thick, "thick", 0, 0, H, 0, W);
#endif

#ifdef DEBUG
  printf("launch <<< %dx%dx%d, %dx%dx%d>>> kernel_1\n", grid_size.x,
         grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);
#endif
  sphconv_cuda_forward_kernel_1<int32_t><<<grid_size, block_size>>>( // <scalar_t, int32_t, H_TILE, W_TILE>
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

#ifdef DEBUG
  printTensor<int>(NumIn, "NumIn", 0, 0, H, 0, W);
  printTensor_k<int>(InRuleMap, "InRuleMap", 0, 0, H, 0, W);
  printTensor_k<int>(OutRuleMap, "OutRuleMap", 0, 0, H, 0, W);
#endif

  const int  oH_BLOCK = 8, oW_BLOCK = 32;

  grid_size.x = divUp(oH, oH_BLOCK);
  grid_size.y = divUp(oW, oW_BLOCK);
  grid_size.z = 1;

  block_size.x = oH_BLOCK;
  block_size.y = oW_BLOCK;
  block_size.z = 1;

#ifdef DEBUG
  printf("launch <<< %dx%dx%d, %dx%dx%d>>> kernel_2\n",
    grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z );
#endif
  sphconv_cuda_forward_kernel_2<int32_t><<<grid_size, block_size>>>(
    CompactMap.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    new_depth.packed_accessor32<int32_t, 4, RestrictPtrTraits >(),
    new_thick.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
    N,
    kernel_volume,
    oH, oW, oD);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

#ifdef DEBUG
  printTensor<int>(new_depth, "new_depth", 0, 0, oH, 0, oW);
  printTensor<int>(new_thick, "new_thick", 0, 0, oH, 0, oW);
  std::cout << "CompactMap = " << CompactMap << std::endl;
#endif

  grid_size.x = divUp(H, H_BLOCK * 4);
  grid_size.y = divUp(W, W_BLOCK * 4);
  grid_size.z = 1;

  block_size.x = H_BLOCK * 4;
  block_size.y = W_BLOCK * 4;
  block_size.z = 1;

#ifdef DEBUG
  printf("launch <<< %dx%dx%d, %dx%dx%d>>> kernel_3\n",
    grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z );
#endif
  sphconv_cuda_forward_kernel_3<int32_t><<<grid_size, block_size>>>(
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

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

#ifdef DEBUG
  printTensor_k<int>(OutRuleMap, "OutRuleMap after k3", 0, 0, H, 0, W);
#endif

  grid_size.x = divUp(H, H_BLOCK);
  grid_size.y = divUp(W, W_BLOCK);
  grid_size.z = 1;

  block_size.x = H_BLOCK;
  block_size.y = W_BLOCK;

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (oC > 4 && oC <= 8) {
    C_BLOCK = 8;
  } else if (oC <= 16) {
    C_BLOCK = 16;
  } else {
    C_BLOCK = 32;
  }

  block_size.z = C_BLOCK;

#ifdef DEBUG
  std::cout << "feature = " << feature << std::endl;
  std::cout << "weight = " << weight << std::endl;
#endif

#ifdef DEBUG
  printf("launch <<< %dx%dx%d, %dx%dx%d>>> kernel_4\n",
    grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z );
#endif
  sphconv_cuda_forward_kernel_4<int32_t><<<grid_size, block_size>>>(
    feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    new_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    weight.packed_accessor32<float, 5, RestrictPtrTraits>(),
    N,
    C,
    oC,
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

#ifdef DEBUG
  printTensor<int>(NumIn, "NumIn final", 0, 0, H, 0, W);
#endif

  return {new_feature, new_depth, new_thick};
}

std::vector<torch::Tensor>
conv_cuda_backward(torch::Tensor feature,
                      torch::Tensor depth,
                      torch::Tensor thick,
                      torch::Tensor gradOutput,
                      torch::Tensor weight,
                      // torch::Tensor bias,
                      int64_t sD, int64_t sH, int64_t sW,
                      int64_t padD, int64_t padH, int64_t padW,
                      int64_t dD, int64_t dH, int64_t dW,
                      int64_t groups, int64_t subm)
{
  auto d_feature = torch::zeros_like(feature);
  auto d_weight = torch::zeros_like(weight);
  // auto d_bias = torch::zeros_like(bias);

  return {d_feature, d_weight};
}