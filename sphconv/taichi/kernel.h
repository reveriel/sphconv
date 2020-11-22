#pragma once
#include <taichi/lang.h>


using namespace taichi::Tlang;


/**
 *  convolution kernel body
 */
template <int k0, int k1, int k2, int s0, int s1, int s2, int p0, int p1, int p2, int channel_in, int channel_out>
std::function<void()> convolution(Expr layer_in, Expr layer_out, Expr weights)
{
    return [&]() {
        bool use_cache = true;
        CacheL1(weights);
        BlockDim(256);

        For(layer_out, [&](Expr i, Expr j, Expr k, Expr c_out) {
            auto sum = Var(0.0f);
            for (int c_in = 0; c_in < channel_in; c_in++) {
                for (int dx = 0; dx < k0; dx++) {
                    for (int dy = 0; dy < k1; dy++) {
                        for (int dz = 0; dz < k2; dz++) {
                            // i * s0 + dx = p0 + x_in
                            // x_out * s0 + dx = po + x_in;
                            // x_in = i * s0 + dx - p0

                            auto weight = weights[Expr(dx), Expr(dy), Expr(dz), c_in * channel_out + c_out];

                            auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;
                            // auto c_in2 = c_in;
                            auto x_in = i * s0 + dx - p0;
                            auto y_in = j * s1 + dy - p1;
                            auto z_in = j * s2 + dz - p2;

                            auto feature_in =
                            select(x_in >= 0 && y_in >= 0 && z_in >= 0,
                                layer_in[x_in, y_in, z_in, c_in2],
                                Var(0.0f)
                             );


                            sum += weight * feature_in;
                        }
                    }
                }
            }
            layer_out[i, j, k, c_out] = sum;

        });
    };
}