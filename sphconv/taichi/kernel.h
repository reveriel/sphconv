#pragma once
#include <taichi/lang.h>
#include <type_traits>



using namespace taichi::Tlang;

auto relu = [](Expr a) { return  max(a, Var(0.0f)); };

/**
 *  convolution kernel body
 */
template <
    int K0, int K1, int K2,
    int S0, int S1, int S2,
    int P0, int P1, int P2,
    int channel_in, int channel_out,
    int N0, int N1, int N2> // input featuremap shape
std::function<void()> convolution(Expr layer_in, Expr layer_out, Expr weights)
{
    return [&]() {
        bool use_cache = true;
        CacheL1(weights);
        BlockDim(256);

        For(layer_out, [&](Expr i, Expr j, Expr k, Expr c_out) {
            auto sum = Var(0.0f);
            for (int c_in = 0; c_in < channel_in; c_in++) {
                for (int dx = 0; dx < K0; dx++) {
                    for (int dy = 0; dy < K1; dy++) {
                        for (int dz = 0; dz < K2; dz++) {
                            // i * s0 + dx = p0 + x_in
                            // x_out * s0 + dx = po + x_in;
                            // x_in = i * s0 + dx - p0

                            auto weight = weights[Expr(dx), Expr(dy), Expr(dz), c_in * channel_out + c_out];

                            auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;
                            // auto c_in2 = c_in;
                            auto x_in = S0 * i + dx - P0;
                            auto y_in = S1 * j + dy - P1;
                            auto z_in = S2 * k + dz - P2;

                            auto feature_in =
                                select(x_in >= 0 && y_in >= 0 && z_in >= 0 &&
                                           x_in < N0 && y_in < N1 && z_in < N2,
                                       layer_in[x_in, y_in, z_in, c_in2],
                                       Var(0.0f));

                            sum += weight * feature_in;
                        }
                    }
                }
            }
            layer_out[i, j, k, c_out] = sum;
        });
    };
}



/**
 * activate convolution output layer  (dilated version)
 *  in Taichi, sparse data block (like bitmasked) must be activated before writing
 */
template <int block_size0, int block_size1, int block_size2,
          int K0, int K1, int K2,
          int S0, int S1, int S2,
          int P0, int P1, int P2,
          int N0, int N1, int N2> // output shape
std::function<void()> conv_activate_dilate(Expr layer_in, Expr layer_out)
{
    return [&]() {
        BlockDim(256);
        kernel_name("dilate");
        For(layer_in, [&](Expr i, Expr j, Expr k) {
            for (int dx = 0; dx < K0; dx++) {
                auto x_out = (i + P0 - dx) / S0;
                for (int dy = 0; dy < K1; dy++) {
                    auto y_out = (j + P1 - dy) / S1;
                    for (int dz = 0; dz < K2; dz++) {
                        auto z_out = (k + P2 - dz) / S2;
                        If(x_out >= 0 && y_out >= 0 && z_out >= 0 &&
                           x_out < N0 && y_out < N1 && z_out < N2)
                            .Then([&] {
                                layer_out[x_out, y_out, z_out, 0] = 0.0f;
                            });
                        // layer_out[i + x * block_size0, j + y * block_size1, k + z * block_size2, 0] = 0.0f; // activate the block
                        // Activate(layer_out, (x_out, y_out, z_out));
                        // Activate(layer_out, (x_out, y_out, z_out, 0));
                    }
                }
            }
            // If(i % block_size0 == 0 && j % block_size1 == 0 &&  k % block_size2 == 0)
            //     .Then([&] {
            //     });
        });
    };
}

/**
 * activate convolution output layer  (submanifold version)
 *  in Taichi, sparse data block (like bitmasked) must be activated before writing
 */
template <int block_size0, int block_size1, int block_size2>
std::function<void()> conv_activate_subm(Expr layer_in, Expr layer_out)
{
    return [&]() {
        BlockDim(256);
        kernel_name("submanifold");
        For(layer_in, [&](Expr i, Expr j, Expr k) {
            If(i % block_size0 == 0 && j % block_size1 == 0 && k % block_size2 == 0)
                .Then([&] {
                    layer_out[i, j, k, 0] = 0.0f; // activate the block
                });
        });
    };
}


template <int block_size0, int block_size1, int block_size2,
          int K0, int K1, int K2,
          int S0, int S1, int S2,
          int P0, int P1, int P2,
          int N0, int N1, int N2,
          bool subm>
static inline std::function<void()> conv_activate(Expr layer_in, Expr layer_out) {
    if (subm) {
        printf(" errrrrrr---------- \n");
        return conv_activate_subm<block_size0, block_size1, block_size2>(layer_in, layer_out);
    } else {
        return conv_activate_dilate<block_size0, block_size1, block_size2,
                                    K0, K1, K2, S0, S1, S2, P0, P1, P2,
                                    N0, N1, N2>(layer_in, layer_out);
    }
}