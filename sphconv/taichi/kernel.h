#pragma once
#include <taichi/lang.h>


using namespace taichi::Tlang;


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