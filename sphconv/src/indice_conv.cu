///////////////////////////////////////////////////////////////////
// indice conv
///////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include "debug_utils.h"
#include "indice.cu.h"
#include "indice_conv.cu.h"
#include "reorder.cu.h"
#include "timer.h"

using torch::RestrictPtrTraits;

namespace sphconv {
const int H_BLOCK = 4, W_BLOCK = 8;

std::vector<torch::Tensor>
indice_conv_gemm(torch::Tensor feature,
                 torch::Tensor weight,
                 torch::Tensor InRuleMap,
                 torch::Tensor OutRuleMap,
                 torch::Tensor NumIn,
                 //  torch::Tensor bias,
                 int oT,
                 int sD, int sH, int sW,
                 int padD, int padH, int padW,
                 int dD, int dH, int dW,
                 int groups)
{
  double totalInitTime = 0;
  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;

  int N = feature.size(0);
  int C = feature.size(1);
  int T = feature.size(2);
  int H = feature.size(3);
  int W = feature.size(4);
  int oC = weight.size(0);
  int iC = weight.size(1);
  int KD = weight.size(2);
  int KH = weight.size(3);
  int KW = weight.size(4);

  int oH = std::floor((H + 2 * padH - dH * (KH - 1) - 1) / sH + 1);
  int oW = std::floor((W + 2 * padW - dW * (KW - 1) - 1) / sW + 1);


  int kernel_volume = KD * KH * KW;

  // the output RangeVoxel
  auto new_feature = torch::zeros({N, oC, oT, oH, oW},
                                  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  auto timer = CudaContextTimer<>();

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  auto filters = weight.permute({2, 3, 4, 1, 0}).contiguous().view({-1, iC, oC});

  auto options = torch::TensorOptions().dtype(feature.dtype()).device(feature.device());

  torch::Tensor output = torch::zeros({N, oT, oH, oW, oC}, options);
  torch::Tensor inputBuffer = torch::zeros({N, T, H, W, iC}, options);
  torch::Tensor outputBuffer = torch::zeros({N, T, H, W, oC}, options);
  torch::Tensor inputBufferGemm = inputBuffer.view({N * T * H * W, iC});
  torch::Tensor outputBufferGemm = outputBuffer.view({N * T * H * W, oC});

  torch::Tensor feature_ = feature.permute({0, 2, 3, 4, 1}).contiguous();

  for (int k = 0; k < kernel_volume; ++k)
  {
    inputBufferGemm.fill_(0);
    outputBufferGemm.fill_(0);

    totalInitTime += timer.report() / 1000.;

    gather_kernel<int32_t><<<grid_size, block_size>>>(
        feature_.packed_accessor32<float, 5, RestrictPtrTraits>(),
        InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        inputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, k, iC, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    totalGatherTime += timer.report() / 1000.;

    // std::cout << "inputBuffer = " << inputBuffer << std::endl;
    // std::cout << "inputbuffergemm = " << inputbuffergemm << std::endl;
    // std::cout << "filters[k] = " << filters[k] << std::endl;

    torch::mm_out(outputBufferGemm, inputBufferGemm, filters[k]);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    totalGEMMTime += timer.report() / 1000.;
    // std::cout << "outputBuffer = " << outputBuffer << std::endl;
    // std::cout << "outputBufferGemm = " << outputBufferGemm << std::endl;

    scatter_add_kernel<int32_t><<<grid_size, block_size>>>(
        OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        outputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        output.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, oC, k,
        KD, KH, KW,
        sD, sH, sW,
        padD, padH, padW,
        dD, dH, dW,
        oH, oW, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    totalSAddTime += timer.report() / 1000.;

  }
  new_feature = output.permute({0, 4, 1, 2, 3}).contiguous();
  double permuteTime = timer.report() / 1000.;

  printf("%s:%s  %.3f\n", __FUNCTION__, "init", totalInitTime);
  printf("%s:%s  %.3f\n", __FUNCTION__, "gather", totalGatherTime);
  printf("%s:%s  %.3f\n", __FUNCTION__, "GEMM", totalGEMMTime);
  printf("%s:%s  %.3f\n", __FUNCTION__, "sadd", totalSAddTime);
  printf("%s:%s  %.3f\n", __FUNCTION__, "permute", permuteTime);

  return {new_feature};
}


std::vector<torch::Tensor>
indice_conv_backward_gemm(torch::Tensor feature,
                      torch::Tensor d_featureOut,
                      torch::Tensor weight,
                      // torch::Tensor bias,
                      torch::Tensor InRuleMap,
                      torch::Tensor OutRuleMap,
                      torch::Tensor NumIn,
                      int sD, int sH, int sW,
                      int padD, int padH, int padW,
                      int dD, int dH, int dW,
                      int groups, int subm)
{
  // auto d_bias = torch::zeros_like(bias);

  // input size
  int N = feature.size(0);
  int iT = feature.size(2);
  int H = feature.size(3);
  int W = feature.size(4);
  int oC = weight.size(0);
  int iC = weight.size(1);
  int KD = weight.size(2);
  int KH = weight.size(3);
  int KW = weight.size(4);

  auto kernel_volume = KD * KH * KW;

  dim3 grid_size, block_size;
  torch::Tensor new_depth, new_thick;

  // d_featureOut's shape = N, oC, oT, oH, oW
  int oT = d_featureOut.size(2);
  int oH = d_featureOut.size(3);
  int oW = d_featureOut.size(4);

  d_featureOut = d_featureOut.permute({0, 2, 3, 4, 1}).contiguous();
  feature = feature.permute({0, 2, 3, 4, 1}).contiguous();
  weight = weight.permute({2, 3, 4, 1, 0}).contiguous();
  weight = weight.view({-1, iC, oC});

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (oC > 4 && oC <= 8) {
    C_BLOCK = 8;
  } else if (oC > 8 && oC <= 16) {
     C_BLOCK = 16;
  } else if (oC > 16) {
    C_BLOCK = 32;
  }

  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  auto options = torch::TensorOptions().dtype(feature.dtype()).device(feature.device());
  auto d_feature = torch::zeros_like(feature);
  auto d_weight = torch::zeros_like(weight);

  torch::Tensor inputBuffer = torch::zeros({N, iT, H, W, iC}, options);
  torch::Tensor outputBuffer = torch::zeros({N, iT, H, W, oC}, options);

  torch::Tensor inputBufferGemm = inputBuffer.view({N * iT * H * W, iC});
  torch::Tensor outputBufferGemm = outputBuffer.view({N * iT * H * W, oC});

  d_weight = d_weight.view({-1, iC, oC});

  for (int k = 0; k < kernel_volume; ++k)
  {

    inputBufferGemm.fill_(0);
    outputBufferGemm.fill_(0);

    gather_kernel<int32_t><<<grid_size, block_size>>>(
        feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
        InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        inputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, k, iC, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int k_H = (k / KW) % KH;
    int k_W = k % KW;
    gather_kernel_k<int32_t><<<grid_size, block_size>>>(
        d_featureOut.packed_accessor32<float, 5, RestrictPtrTraits>(),
        OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        outputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, k, oC,
        H, W,
        k_H, k_W,
        sH, sW,
        padH, padW,
        dH, dW,
        oH, oW);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // iC, oC
    auto filterGradSub = d_weight[k];

    torch::mm_out(filterGradSub, inputBufferGemm.t(), outputBufferGemm);
    torch::mm_out(inputBufferGemm, outputBufferGemm, weight[k].t());

    scatter_add_kernel_backward<int32_t><<<grid_size, block_size>>>(
        InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
        NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
        inputBuffer.packed_accessor32<float, 5, RestrictPtrTraits>(),
        d_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
        N, iC, k, H, W);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  d_weight = d_weight.view({KD, KH, KW, iC, oC}).permute({4, 3, 0, 1, 2}).contiguous();
  d_feature = d_feature.permute({0, 4, 1, 2, 3}).contiguous();

  return {d_feature, d_weight};
}


// slow when channel size gets bigger
std::vector<torch::Tensor>
indice_conv(torch::Tensor feature,
            torch::Tensor weight,
            torch::Tensor InRuleMap,
            torch::Tensor OutRuleMap,
            torch::Tensor NumIn,
            //  torch::Tensor bias,
            int oT,
            int sD, int sH, int sW,
            int padD, int padH, int padW,
            int dD, int dH, int dW,
            int groups)
{
  int N, C, H, W, oC, KD, KH, KW;
  N = feature.size(0);
  C = feature.size(1);
  H = feature.size(3);
  W = feature.size(4);
  oC = weight.size(0);
  KD = weight.size(2);
  KH = weight.size(3);
  KW = weight.size(4);

  int oH = std::floor((H + 2 * padH - dH * (KH - 1) - 1) / sH + 1);
  int oW = std::floor((W + 2 * padW - dW * (KW - 1) - 1) / sW + 1);

  int kernel_volume = KD * KH * KW;

  // the output RangeVoxel
  auto new_feature = torch::zeros({N, oC, oT, oH, oW},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (C > 4 && C <= 8) {
    C_BLOCK = 8;
  } else if (C > 8 && C <= 16) {
     C_BLOCK = 16;
  } else if (C > 16) {
    C_BLOCK = 32;
  }

  dim3 grid_size, block_size;
  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  indice_conv_kernel<int32_t><<<grid_size, block_size>>>(
    feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    new_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    weight.packed_accessor32<float, 5, RestrictPtrTraits>(),
    N, C, oC,
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

  printTensor_int(NumIn, "NumIn final", 0, 0, H, 0, W);

  return {new_feature};
}


// slow when channel size gets bigger
std::vector<torch::Tensor>
indice_conv_backward(torch::Tensor feature,
                      torch::Tensor d_featureOut,
                      torch::Tensor weight,
                      // torch::Tensor bias,
                      torch::Tensor InRuleMap,
                      torch::Tensor OutRuleMap,
                      torch::Tensor NumIn,
                      int sD, int sH, int sW,
                      int padD, int padH, int padW,
                      int dD, int dH, int dW,
                      int groups, int subm)
{
  auto d_feature = torch::zeros_like(feature);
  auto d_weight = torch::zeros_like(weight);
  // auto d_bias = torch::zeros_like(bias);

  // input size
  int N, H, W, oC, iC, KD, KH, KW;
  N = feature.size(0);
  H = feature.size(3);
  W = feature.size(4);
  oC = weight.size(0);
  iC = weight.size(1);
  KD = weight.size(2);
  KH = weight.size(3);
  KW = weight.size(4);

  auto kernel_volume = KD * KH * KW;

  dim3 grid_size, block_size;
  torch::Tensor new_depth, new_thick;

  // d_featureOut's shape = N, oC, oT, oH, oW
  int oT, oH, oW;
  oT = d_featureOut.size(2);
  oH = d_featureOut.size(3);
  oW = d_featureOut.size(4);

  // choose C_BLOCK
  int C_BLOCK = 4;
  if (oC > 4 && oC <= 8) {
    C_BLOCK = 8;
  } else if (oC > 8 && oC <= 16) {
     C_BLOCK = 16;
  } else if (oC > 16) {
    C_BLOCK = 32;
  }

  grid_size = dim3(divUp(H, H_BLOCK), divUp(W, W_BLOCK), 1);
  block_size = dim3(H_BLOCK, W_BLOCK, C_BLOCK);

  // the output RangeVoxel
  auto new_feature = torch::zeros({N, oC, oT, oH, oW},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  indice_conv_backward_kernel<int32_t><<<grid_size, block_size>>>(
    d_featureOut.packed_accessor32<float, 5, RestrictPtrTraits>(),
    d_feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    d_weight.packed_accessor32<float, 5, RestrictPtrTraits>(),
    InRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    OutRuleMap.packed_accessor32<int32_t, 5, RestrictPtrTraits>(),
    NumIn.packed_accessor32<int32_t, 4, RestrictPtrTraits>(),
    weight.packed_accessor32<float, 5, RestrictPtrTraits>(),
    feature.packed_accessor32<float, 5, RestrictPtrTraits>(),
    N, iC, oC,
    kernel_volume,
    KD, KH, KW,
    sH, sW,
    padH, padW,
    dH, dW,
    oH, oW,
    H, W);

  return {d_feature, d_weight};
}

} // namespace sphconv
