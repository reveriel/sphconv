#pragma once

#include "feature.h"
#include <torch/extension.h>
#include <cassert>

template <int K0_, int K1_, int K2_,
          int S0_, int S1_, int S2_,
          int P0_, int P1_, int P2_>
struct Conv3DConfig
{
    static const int K0 = K0_;
    static const int K1 = K1_;
    static const int K2 = K2_;
    static const int S0 = S0_;
    static const int S1 = S1_;
    static const int S2 = S2_;
    static const int P0 = P0_;
    static const int P1 = P1_;
    static const int P2 = P2_;
};

/**
 * @brief compute
 *
 * assume out.z_ptr out.z_ind are ready, space for out.val has been allocated.
 *
 * @tparam T
 * @tparam Index
 * @tparam Config : Conv3DConfig
 * @param in
 * @param out
 * @param weights_
 * @return int
 */
template <typename T,
          typename Index,
          typename Config>
int conv(Feature<T, Index> &in, Feature<T, Index> &out, torch::Tensor weights_)
{
    auto weights = weights_.accessor<T, 5>();
    assert(in.B == out.B);
    // for each output coordinate
    for (int b = 0; b < out.B; b++) {
        for (int xo = 0; xo < out.H; xo++) {
            for (int yo = 0; yo < out.W; yo++) {
                Index zo_start = out.z_ptr_at(b, xo, yo);
                Index zo_end = out.z_ptr_at(b, xo, yo + 1);

                for (Index o_pos = zo_start; o_pos <zo_end; o_pos++) {
                    Index zo = out.z_ind[o_pos];
                    // now xo yo zo is ready
                    // compute x y z

                    // for each kernel pos
                    for (int i = 0; i < Config::K0; i++) {
                        int xi = (Config::S0 * xo + i - Config::P0);
                        if (xi >= in.H || xi < 0)
                            continue;

                        for (int j = 0; j < Config::K1; j++) {
                            int yi = (Config::S1 * yo + j - Config::P1);
                            if (yi >= in.W || yi < 0)
                                continue;

                            // find matching ptr_  based on zi
                            Index zi_start = in.z_ptr_at(b, xi, yi);
                            Index zi_end = in.z_ptr_at(b, xi, yi + 1);

                            Index i_pos = zi_start;
                            for (int k = 0; k < Config::K2; k++) {
                                int zi = (Config::S2 * zo + k - Config::P2);
                                if (zi >= in.D || zi < 0)
                                    continue;
                                // found zi's pos
                                // this search is a problem
                                while (i_pos < zi_end && in.z_ind[i_pos] < zi)
                                    i_pos++;

                                if (i_pos == zi_end || in.z_ind[i_pos] > zi)
                                    break; //

                                // a tensor vector multiplication
                                for (int co = 0; co < out.C; co++) {
                                    T sum = (T)0;
                                    for (int ci = 0; ci < in.C; ci++) {
                                        sum += in.val[i_pos * in.C + ci] *
                                               weights[i, j, k, ci, co];
                                    }
                                    out.val[o_pos * out.C + co] = sum;
                                }
                            } // for k
                        } // for j
                    } // for i
                } // for zo
            } // for yo
        } // for xo
    } // for b
}



// template <typename Index>
// __device__ __inline__ Index OutSpatial(Index k, Index x, Index s, Index d, Index pad)
// {
//   // forgive me. do nothing with the dillation
//   // TODO
//   if ((x + pad - k) % s == 0)
//     return (x + pad - k)/ s;
//   return -1;
// }


//
// compute out z_ptr and out z_ind from in.z_ptr and in.z_ind
// assume out.H out.W out.D out.C are ready
//
// allocate a  'B H W D' array to do the counting
//  1. fill with 1                   0 1 1 0 0
//  2. scan on each D,               0 1 2 2 2
//  3. scan on D's last element, now get z_ptr
//  4. allocate z_ind since its size is known
//  5. based on z_ptr, fill z_ind.
/*
template <typename T,
          typename Index,
          typename Config>
void prepare(Feature<T,Index> &in, Feature<T,Index> &out) {
    // for each output coordinate
    for (int b = 0; b < out.B; b++) {
        for (int xo = 0; xo < out.H; xo++) {
            for (int yo = 0; yo < out.W; yo++) {
                Index zo_start = out.z_ptr_at(b, xo, yo);
                Index zo_end = out.z_ptr_at(b, xo, yo + 1);

                for (Index o_pos = zo_start; o_pos <zo_end; o_pos++) {
                    Index zo = out.z_ind[o_pos];

                    for (int i = 0; i < Config::K0; i++) {
                        int xi = (Config::S0 * xo + i - Config::P0);
                        if (xi >= in.H || xi < 0)
                            continue;

                        for (int j = 0; j < Config::K1; j++) {
                            int yi = (Config::S1 * yo + j - Config::P1);
                            if (yi >= in.W || yi < 0)
                                continue;

                            // find matching ptr_  based on zi
                            Index zi_start = in.z_ptr_at(b, xi, yi);
                            Index zi_end = in.z_ptr_at(b, xi, yi + 1);

                            Index i_pos = zi_start;
                            for (int k = 0; k < Config::K2; k++) {
                                int zi = (Config::S2 * zo + k - Config::P2);
                                //

                                if (zi >= in.D || zi < 0)
                                    continue;
                                // found zi's pos
                                // this search is a problem
                                while (i_pos < zi_end && in.z_ind[i_pos] < zi)
                                    i_pos++;

                                if (i_pos == zi_end || in.z_ind[i_pos] > zi)
                                    break; //

                                // a tensor vector multiplication
                                for (int co = 0; co < out.C; co++) {
                                    T sum = (T)0;
                                    for (int ci = 0; ci < in.C; ci++) {
                                        sum += in.val[i_pos * in.C + ci] *
                                               weights[i, j, k, ci, co];
                                    }
                                    out.val[o_pos * out.C + co] = sum;
                                }
                            } // for k
                        } // for j
                    } // for i

                }
            } // for yo
        } // for xo
    } // for b
}
*/