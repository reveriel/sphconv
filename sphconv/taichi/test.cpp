
#include <taichi/lang.h>
#include <numeric>
// #include <taichi/visual/gui.h>
// #include <torch/extension.h>
#include <string>
#include <vector>
#include "npy.hpp"
#include <cmath>

// test if convolution get a correct result

using namespace taichi::Tlang;

Program *prog;

void init_taichi_program() {
    taichi::CoreState::set_trigger_gdb_when_crash(true);
}

// case1:
// input  layer of 3, 3
// compare with pytorch ?
// data


template <int k0, int k1, int k2,
    int s0, int s1, int s2,
    int p0, int p1, int p2,
    int channel_in,
    int channel_out>
std::function<void()> convolution(Expr layer_in, Expr layer_out, Expr weights) {
    return [&]() {
        bool use_cache = true;
        CacheL1(weights);
        BlockDim(256);

        For(layer_in, [&](Expr i, Expr j, Expr k, Expr c_out) {
            auto sum = Var(0.0f);
            for (int c_in = 0; c_in < channel_in; c_in++) {
                for (int dx = -1; dx < 2; dx++) {
                    for (int dy = -1; dy < 2; dy++) {
                        for (int dz = -1; dz < 2; dz++) {

                            auto weight = weights[Expr(dx + 1), Expr(dy + 1), Expr(dz + 1), c_in * channel_out + c_out];

                            auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;

                            sum += weight * layer_in[i + dx, j + dy, k + dz, c_in2];
                        }
                    }
                }
            }
            layer_out[i, j, k, c_out] = sum;
        });
    };
}




int main() {

    bool gpu = true;

    auto prog = new Program(gpu ? Arch::gpu : Arch::x86_64);

    init_taichi_program();

    // int n = 128;

    Global(layer1, f32);
    Global(layer2, f32);

    Global(weights1, f32);

    int block_size = 4;
    int num_ch1 = 16;
    int num_ch2 = 16;

    int res = 8;

    layout([&]() {
        auto ijkl = Indices(0, 1, 2, 3);
        root.dense(ijkl, {res / block_size, res / block_size, res / block_size, 1}).bitmasked()
            .dense(ijkl, {block_size, block_size, block_size, num_ch1}).place(layer1);
        root.dense(ijkl, {res / block_size, res / block_size, res / block_size, 1}).bitmasked()
            .dense(ijkl, {block_size, block_size, block_size, num_ch2}).place(layer2);

        root.dense(ijkl, {3, 3, 3, num_ch1 * num_ch2}).place(weights1);
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

// // ? I have to print the value, layer!
//     for (int i = 0; i < res; i++) {
//         for (int j = 0; j < res; j++) {
//             for (int k = 0; k < res; k++) {
//                 for (int l = 0; l < 1; l++)
//                 printf("layer1.val1(%d,%d,%d,%d) = %f\n", i, j, k, l, layer2.val<taichi::float32>(i, j, k, l));
//             }
//         }
//     }


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
        convolution<3, 3, 3, 1, 1, 1, 0, 0, 0, 16, 16>(layer1, layer2,  weights1)
    );



    kernel([&] {
        if (!gpu) {
            Parallelize(8);
        } else {
            BlockDim(256);
        }
        kernel_name("dilate");
        For(layer1, [&](Expr i, Expr j, Expr k) {
            If(i % block_size == 0 && j % block_size == 0 && k % block_size == 0)
                .Then([&] {
                    for (int x = -1; x < 2; x++) {
                        for (int y = -1; y < 2; y++) {
                            for (int z = -1; z < 2; z++) {
                                layer2[i + x * block_size, j + y * block_size, k + z * block_size, 0] = 0.0f; // activate the block
                                Activate(layer2.parent(), (i + x * block_size, j + y * block_size, k + z * block_size, 0));
                            }
                        }
                    }
                });
        });
    })();


    // for (int i = 0; i < 50; i++) {
    // }

    forward_conv1();


    prog->profiler_print();

    // print each element



//  // 这厮是 不带参数的！
//     Kernel &kernel_double =
//         kernel([&]() {
//             kernel_name("double");
//             For(0, n, [&](Expr i) {
//                 auto ret = Var(0);
//                 If(i % 2 == 0).Then([&] { ret = dou(i); }).Else([&] { ret = i; });
//                 a[i] = ret;
//             });
//         });

//     kernel_double();

//     for (int i = 0; i < n; i ++) {
//         if (a.val<taichi::int32>(i) == (2 - i % 2) * i) {
//             std::cout << "correct " << i << std::endl;

//             std::cout << "error " << std::endl;
//         }
//     }

    for (int i = 0; i < res; i++) {
        for (int j = 0; j < res; j++) {
            for (int k = 0; k < res; k++) {
                for (int l = 0; l < 1; l++)
                printf("layer2.val(%d,%d,%d,%d) = %f\n", i, j, k, l, layer2.val<taichi::float32>(i, j, k, l));
            }
        }
    }

    delete prog;

    return 0;
}



