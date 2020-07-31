///////////////////////////////////////////////////////////////////
// gather and scatter kernels
///////////////////////////////////////////////////////////////////

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
using torch::RestrictPtrTraits;

namespace sphconv
{

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

} // namespace sphconv
