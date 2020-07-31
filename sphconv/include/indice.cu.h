#pragma once
#include "debug_utils.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

using torch::RestrictPtrTraits;

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

  if (x >= H || y >= W) return;

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
