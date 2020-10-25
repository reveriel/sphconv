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























} // namespace sphconv
