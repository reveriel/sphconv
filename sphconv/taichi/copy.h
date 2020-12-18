#pragma once

#include "torch/extension.h"
#include "conv.hpp"
#include <cassert>
#include <taichi/lang.h>

using namespace taichi::Tlang;

/**
 *  fill weights with value
 */
template<typename ConvolutionConfig>
void fill_weights(Expr &weights, float value)
{
    using Conv = ConvolutionConfig;
    for (int c_out = 0; c_out < Conv::Co; c_out++) {
        for (int c_in = 0; c_in < Conv::Ci; c_in++) {
            // float inc = 0.1f;
            for (int i = 0; i < Conv::K0; i++) {
                for (int j = 0; j < Conv::K1; j++) {
                    for (int k = 0; k < Conv::K2; k++) {
                        weights.val<taichi::float32>(
                            i, j, k, c_in * Conv::Co + c_out) = value;
                    }
                }
            }
        }
    }
}


/**
 *  fill weights with torch tensor
 *
 * weight in torch::Tensor are of shape, [Co,Ci,K,K,K]
 * weights in taichi, K,K,K, ci * co
 */
template<typename ConvolutionConfig>
void copy_weights_cpu(Expr &weights, torch::Tensor &src)
{
    assert(src.device().type() == torch::kCPU);
    // auto src_a = src.generic_packed_accessor<float, 5>();
    auto src_a = src.accessor<float, 5>();
    // number of float should ?
    using Conv = ConvolutionConfig;
    for (int c_out = 0; c_out < Conv::Co; c_out++) {
        for (int c_in = 0; c_in < Conv::Ci; c_in++) {
            // float inc = 0.1f;
            for (int i = 0; i < Conv::K0; i++) {
                for (int j = 0; j < Conv::K1; j++) {
                    for (int k = 0; k < Conv::K2; k++) {
                        weights.val<taichi::float32>(
                            i, j, k, c_in * Conv::Co + c_out) = src_a[c_out][ c_in][ i] [j][ k];
                    }
                }
            }
        }
    }
}

// points is on cpu
void init_layer0(Expr &layer0, torch::Tensor &points, const VoxelizationConfig &vcfg);


/**
 * src feature : HWDC
 *  dst feature shape:  W, D, H, C ?
 *
 *  1
 *   |   3
 *   | /
 *   *------ 2
 *
 *
 *  TODO: triple check here
 */
template<typename FeatureShape>
void copy_feature_cpu(Expr &layer, torch::Tensor &dst)
{
    assert(dst.device().type() == torch::kCPU);
    // auto dst_a = dst.generic_packed_accessor<float, 4>();
    auto dst_a = dst.accessor<float, 4>();
    using Shape = FeatureShape;
    for (int i = 0; i < Shape::H; i++) {
        for (int j = 0; j < Shape::W; j++ ) {
            for (int k = 0; k < Shape::D; k++) {
                for (int c = 0; c < Shape::C; c++) {
                    dst_a[j][k][i][c] = layer.val<taichi::float32>(i,j,k,c);
                }
            }
        }
    }
}

// // === GPU ===
// //

// /**
//  *  fill weights with torch tensor
//  *
//  * weight in torch::Tensor are of shape, [Co,Ci,K,K,K]
//  * weights in taichi, K,K,K, ci * co
//  */

// template<typename ConvolutionConfig>
// void copy_weights_gpu(Expr &weights, torch::Tensor &src)
// {
//     assert(src.device().type() == torch::kCUDA);
//     auto src_a = src.generic_packed_accessor<float, 5>();
//     // auto src_a = src.accessor<float, 5>();
//     // number of float should ?
//     using Conv = ConvolutionConfig;
//     dim3 dimGrid(Conv::K0, Conv::K1, Conv::K2);
//     dim3 dimBlock(Conv::Co, Conv::Ci);
//     auto snode = weights->cat<GlobalVariableExpression>()->snode;


//     copy_weights_gpu_kernel<Conv><<<dimGrid, dimBlock>>>(
//         get_current_program().data_structre,
//         snode->access_func,
//         src_a,
//         physical_index_position);
// }


// template <typename ConvolutionConfig>
// __global__ void copy_weights_gpu_kernel(
//     void *ds,
//     SNode::AccessorFunction access_func,
//     torch::GenericPackedTensorAccessor<float, 5> &src_a
//     )
// {
//     using Conv = ConvolutionConfig;
//     int c_out = threadIdx.x;
//     int c_in = threadIdx.y;
//     int i = blockIdx.x;
//     int j = blockIdx.y;
//     int k = blockIdx.z;

//     *((float*)access_func(ds, i, j, k, c_in * Conv::Co + c_out)) =
//         src_a[c_out][c_in][i][j][k];
// }







