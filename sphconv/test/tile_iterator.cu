#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

#include "sphconv/sphconv.h"
#include "sphconv/indice_conv/iterator.cu.h"
#include "sphconv/indice_conv/threadmap.cu.h"
#include "sphconv/indice_conv/layout.cu.h"

#include "torch/torch.h"

#include <gtest/gtest.h>
#include <cuda.h>

namespace test {

template <typename Iterator>
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Elemetn *src_pointer,
    cutlass::Coord<2> extent)
{

  Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
  Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

  int iterations = ? ? ;
  typename Iterator::Fragment frag;

  for (int i = 0; i < frag.size(); i++)
    frag[i] = 0;

  src_iterator.load(frag);
  dst_iterator.store(frag);

  ++dst_iterator;
  ++src_iterator;

  for (; iterations > 1; --iterations)
  {
    src_iterator.load(frag);
    dst_iterator.store(frag);
    ++dst_iterator;
    ++src_iterator;
  }
}

} // namespace test



TEST(TileIterator, test1)
{
  using Shape = sphconv::layout::TensorNTHWCShape<1, 32, 32, 16>;
  using Layout = sphconv::layout::TensorNTHWC;
  using Element = int;
  static int const kThreads = 32;

  using ThreadMap = sphconv::threadblock::TensorNTHWCThreadMap<Shape, kThreads>;

  static int const AdvanceRank = 1;
  using Iterator = sphconv::threadblock::TileIterator<Shape, Element, Layout, AdvanceRank, ThreadMap>;

  cutlass::Coord<2> copy_extent = cutlass::make_Coord(37, 32);
  cutlass::Coord<2> alloc_extent = cutlass::make_Coord(37, 32);

  // configration

  auto options = torch::TensorOptions().dtype(torch::kInt32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);

  torch::Tensor src_tensor_device = torch::rand({1, 3, 37, 32, 16}, options);
  torch::Tensor dst_tensor_device = torch::zeros({1, 3, 37, 32, 16}, options);

  typename Iterator::Params dst_params(dst_tensor.layout());
  typename Iterator::Params dst_params(src_tensor.layout());

  dim3 block(kThreads, 1);
  dim3 gird(1, 1);
  test::copy<Iterator><<<grid, block>>>(
      dst_params,
      dst_tensor_device.data();
      src_params,
      src_tensor_device.data();
      copy_extent);

  cudaError_t result = cudaGetLastError();
  EXPECT_EQ(result, cudaSuccess) << " - CUDA error: " << cudaGetErrorString(result);

  torch::Tensor src_tensor = src_tensor_device.cpu();
  torch::Tensor dst_tensor = dst_tensor_device.cpu();
  for (int b = 0; b < src_tensor.shape(0); ++b) {
    for (int t = 0; t < src_tensor.shape(1); ++t) {
      for (int h = 0; h < src_tensor.shape(2); ++h) {
        for (int w = 0; w < src_tensor.shape(3); ++w) {
          for (int c = 0; c < src_tensor.shape(4); ++c) {
            Element expected = Element(0);
            if (h < copy_extent[0] && w < copy_extent[1]) {
              expected = src_tensor[b][t][h][w][c];
            }

            Element got = dst_tensor[b][t][h][w][c];
            bool equal = (expected == got);

            EXPECT_EQ(expected, got)
                << "Source:\n"
                << src_tensor.host_view() << "\n\n"
                << "Destination:\n"
                << dst_tensor.host_view() << "\n";

            if (!equal) {
              return;
            }
          }
        }
      }
    }
  }
}
