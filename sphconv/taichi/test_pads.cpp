
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

#include "kernel.h"
#include "utils.h"

#include <taichi/lang.h>
#include <taichi/testing.h>
#include <numeric>
// #include <taichi/visual/gui.h>
// #include <torch/extension.h>
#include <string>
#include <vector>
#include <cmath>

// test if convolution get a correct result

using namespace taichi::Tlang;

Program *prog;

// case1:
// input  layer of 3, 3
// compare with pytorch ?
// data


struct Layer {
    //  3D layer
    int  h_, w_, d_, c_;
    float *data;
    Layer(int h, int w, int d, int c) : h_(h), w_(w), d_(d), c_(c)
    {
        data = (float*)malloc(sizeof(float) *   h * w  * d * c);
    }
    ~Layer() {
        free(data);
    }
};

struct Weight {
    int h_, w_, d_, cin_, cout_;
    float *data;
    Weight(int h, int w, int d, int cin, int cout)
    : h_(h), w_(w), d_(d), cin_(cin), cout_(cout) {
        data = (float *)malloc(sizeof(float) * h * w * d * cin * cout);
    }
    ~Weight() {
        free(data);
    }
};

// Layer  conv(Layer l, Weight w) {
//     for (int i = 0; i <)
// }


// test a simple pplication
// try pooling


int main_v() {

    bool gpu = true;
    auto prog = new Program(gpu ? Arch::gpu : Arch::x86_64);
    prog->config.lower_access = false;
    init_taichi_program();

    Global(layer1, f32);
    Global(layer2, f32);
    Global(weights1, f32);

    int block_size = 4;
    constexpr int num_ch1 = 4;
    constexpr int num_ch2 = 4;
    constexpr int res = 64;

    layout([&]() {
        auto ijkl = Indices(0, 1, 2, 3);
        root.dense(ijkl, {res / block_size, res / block_size, res / block_size, 1})
            .pointer()
            .dense(ijkl, {block_size, block_size, block_size, num_ch1})
            .place(layer1);
        root.dense(ijkl, {res / block_size, res / block_size, res / block_size, 1})
            // .pointer()
            .dense(ijkl, {block_size, block_size, block_size, num_ch2}).place(layer2);
        root.dense(ijkl, {4, 4, 4, num_ch1 * num_ch2}).place(weights1);
    });

    // init layer1 data, fill with ones
    for (int i = 0; i < res; i++) {
        for (int j = 0; j < res; j++) {
            for (int k = 0; k < res; k++) {
                for (int l = 0; l < num_ch1; l++)
                    layer1.val<taichi::float32>(i, j, k, l) = 1.0f;
            }
        }
    }

    // fill weights1, with ones
    for (int c_out = 0; c_out < num_ch2; c_out++) {
        for (int c_in = 0; c_in < num_ch1; c_in++) {
            // float inc = 0.1f;
            for (int dx = -1; dx < 2; dx++) {
                for (int dy = -1; dy < 2; dy++) {
                    for (int dz = -1; dz < 2; dz++) {
                        // if (dx == 0 && dy == 0 && dz == 0)
                            // weights1.val<taichi::float32>(dx + 1, dy + 1, dz + 1, c_in * num_ch2 + c_out) = inc;
                            weights1.val<taichi::float32>(dx + 1, dy + 1, dz + 1, c_in * num_ch2 + c_out) = 1.0f;
                        // inc += 0.1f;
                    }
                }
            }
        }
    }



    Kernel(forward_conv1).def(
        convolution<3, 3, 3, 1, 1, 1, 1, 1, 1, num_ch1, num_ch2, res, res, res>(layer1, layer2,  weights1)
    );

    // channel wise 3x3x3  sum pooling
    Kernel(pool).def([&]() {
        BlockDim(64);
        Cache(0, layer1);
        // Cache(0, layer2);
        // ?? For(layer2,...), or For(layer1, ...)
        For(layer2, [&](Expr i, Expr j, Expr k, Expr c_in) {
            auto sum = Var(0.0f);
            for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++)
                    for (int r = 0; r < 3; r++)
                    {
                        auto coord_i = AssumeInRange(i + p, i, 0, 3);
                        auto coord_j = AssumeInRange(j + q, j, 0, 3);
                        auto coord_k = AssumeInRange(k + r, k, 0, 3);
                        auto coord_c = AssumeInRange(c_in, c_in, 0, 1);

                        // auto coord_i = i + p;
                        // auto coord_j = j + q;
                        // auto coord_k = k + r;
                        // auto coord_c = c_in;

                        // auto v  =  layer1[coord_i ,
                        //               coord_j ,
                        //               coord_k ,
                        //               coord_c];
                        // Print(v);
                        sum += layer1[coord_i , coord_j , coord_k , coord_c];
                    }

            // #define Print(x) Print_(x, #x);
            // Print(sum);
             layer2[i, j, k, c_in] = sum;
        });
    });

    // activate layer2
    kernel([&] {
        if (!gpu) {
            Parallelize(8);
        } else {
            BlockDim(64);
        }
        kernel_name("dilate");
        For(layer1, [&](Expr i, Expr j, Expr k, Expr c) {
            If(i % block_size == 0 && j % block_size == 0 && k % block_size == 0 && c == 0)
                .Then([&] {
                    Activate(layer2, (i, j, k, 0));
                    // layer2[i , j , k, 0] = 0.0f; // activate the block
                                // Activate(layer2.parent(), (i + x * block_size, j + y * block_size, k + z * block_size, 0));
                });
        });
    })();



    for (int i = 0 ; i < 1; i ++) {
        // init_input();
        // init_weight();
        // activate();
        pool();
        // forward_conv1();
        // check();
    }

    prog->profiler_print();

    // print each element
    print_3d<taichi::float32>(layer2, res, res, res, 0);

    delete prog;

    return 0;
}

TLANG_NAMESPACE_BEGIN

int main() {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 10000000;
  Program prog(Arch::gpu);
  prog.config.lower_access = false;

  Global(x, f32);
  Global(y, f32);

  int domain_size = 256;
  int block_size = 8;

  auto ijk = Indices(0, 1, 2);
  layout([&]() {
    root.dense(ijk, domain_size / block_size)
        // .pointer()
        .dense(ijk, block_size)
        .place(x, y);
  });

  auto dist_imm = [&](int i, int j, int k) {
    auto dx = (1.0_f / domain_size) * (i + 0.5f) - 0.5f;
    auto dy = (1.0_f / domain_size) * (j + 0.5f) - 0.5f;
    auto dz = (1.0_f / domain_size) * (k + 0.5f) - 0.5f;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };

  auto x_val = [&](int i, int j, int k) {
    auto d = dist_imm(i, j, k);
    if (0.43f < d && d < 0.47f) {
      return d * d * d;
    } else {
      return 0.0f;
    }
  };

  auto dist = [&](Expr i, Expr j, Expr k) {
    auto dx = (1.0_f / domain_size) * (cast<float32>(i) + 0.5f) - 0.5f;
    auto dy = (1.0_f / domain_size) * (cast<float32>(j) + 0.5f) - 0.5f;
    auto dz = (1.0_f / domain_size) * (cast<float32>(k) + 0.5f) - 0.5f;
    return sqrt(dx * dx + dy * dy + dz * dz);
  };

  // Initialize
  kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    BlockDim(1);
    For(i, 0, domain_size, [&] {
      For(j, 0, domain_size, [&] {
        For(k, 0, domain_size, [&] {
          auto d = Var(dist(i, j, k));
          If(0.43f < d && d < 0.47f, [&] {
            Activate(x, (i, j, k));
            x[i, j, k] = d * d * d;
          });
        });
      });
    });
  })();

  auto &laplacian = kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    Cache(0, x);
    Cache(0, y);
    For((i, j, k), x, [&]() {
      y[i, j, k] = 6.0f * x[i, j, k] - x[i, j, k - 1] - x[i, j, k + 1] -
                   x[i, j - 1, k] - x[i, j + 1, k] - x[i - 1, j, k] -
                   x[i + 1, j, k];
    });
  });

  laplacian();

  for (int i = 0; i < domain_size; i++) {
    for (int j = 0; j < domain_size; j++) {
      for (int k = 0; k < domain_size; k++) {
        auto d = dist_imm(i, j, k);
        if (0.44f < d && d < 0.46f) {
          auto gt = 6.0f * x_val(i, j, k) - x_val(i, j, k - 1) -
                    x_val(i, j, k + 1) - x_val(i, j - 1, k) -
                    x_val(i, j + 1, k) - x_val(i - 1, j, k) -
                    x_val(i + 1, j, k);
          if (std::abs(gt - y.val<float32>(i, j, k)) > 1) {
            TC_P(d);
            TC_P(gt);
            TC_P(y.val<float32>(i, j, k));
            TC_P(i);
            TC_P(j);
            TC_P(k);
          }
          TC_CHECK_EQUAL(gt, y.val<float32>(i, j, k),
                         1e-1f / domain_size / domain_size);
        }
      }
    }
  }
};

TLANG_NAMESPACE_END
int main() {
    main_v();
    // taichi::Tlang::main();
    return 0;
}