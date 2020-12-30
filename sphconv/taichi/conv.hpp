#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <cassert>
// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

#include <boost/mp11.hpp>
using namespace boost::mp11;


struct VoxelizationConfig {
    // vertical resolution
    const int v_res = 64;
    // horizontal resolution
    const int h_res = 512;
    // depth resolution
    const int d_res = 512;
    const float v_range[2] = {87.5, 103.4};
    const float h_range[2] = {-45, 45};
    const float d_range[2] = {6, 70.4};
    const bool log = true;

    const float delta_phi ;
    const float delta_theta ;
    const float delta_r ;

    static inline constexpr float radians(float degree) { return degree / 180 * PI; }

    constexpr static const float PI = 3.14159265;
    constexpr VoxelizationConfig()
        : delta_phi(radians(h_range[1] - h_range[0]) / h_res),
          delta_theta(radians(v_range[1] - v_range[0]) / v_res),
          delta_r(log ? (std::log(d_range[1]) - std::log(d_range[0])) / d_res
                      : (d_range[1] - d_range[0]) / d_res)
    {
    }

    constexpr int theta_idx(float theta) const {
        return (theta - radians(v_range[0])) / delta_theta;
    }
    constexpr int phi_idx(float phi) const {
        return (phi - radians(h_range[0])) / delta_phi;
    }

    constexpr int depth_idx(float r) const {
        return log
                   ? ((std::log(r) - std::log(d_range[0])) / delta_r)
                   : ((r - d_range[0]) / delta_r);
    }

    constexpr int H() const { return v_res; }
    constexpr int W() const { return h_res; }
    constexpr int D() const { return d_res; }

};

static inline bool in_range(int x, int low, int high) {
    return x >= low && x < high;
}

/**
 * describe the parameters of a convolution
 */
// TODO: change order to K S P
template <
    int K0_,
    int K1_,
    int K2_,  // kernel size
    int P0_,
    int P1_,
    int P2_, // padding
    int S0_,
    int S1_,
    int S2_, // stride
    int Ci_, // channel_in
    int Co_, // channel_out
    bool subm_ // submanifold
    >
struct ConvolutionConfig
{
    static const int K0 = K0_;
    static const int K1 = K1_;
    static const int K2 = K2_;
    static const int P0 = P0_;
    static const int P1 = P1_;
    static const int P2 = P2_;
    static const int S0 = S0_;
    static const int S1 = S1_;
    static const int S2 = S2_;
    static const int Ci = Ci_;
    static const int Co = Co_;
    static const int subm = subm_;
};

template <int H_,
          int W_,
          int D_,
          int C_>
struct FeatureShape
{
    static int const H = H_;
    static int const W = W_;
    static int const D = D_;
    static int const C = C_;
};

template <typename ShapeList, // mp_list< FeatureShape<...>>
          typename ConvConfig> // ConvConfig<...>
struct conv_apply_concate_impl
{
    using Shape = mp_front<ShapeList>;
    // using Shape = ShapeList;
    using Conv = ConvConfig;
    static constexpr int H =
        Conv::subm
            ? Shape::H
            : (Shape::H + 2 * Conv::P0 - Conv::K0) / Conv::S0 + 1;
    static constexpr int W =
        Conv::subm
        ? Shape::W
        : (Shape::W + 2 * Conv::P1 - Conv::K1) / Conv::S1 + 1;
    static constexpr int D =
        Conv::subm
        ? Shape::D
        : (Shape::D + 2 * Conv::P2 - Conv::K2) / Conv::S2 + 1;
    static_assert(Shape::C == Conv::Ci, "input channel not match");
    static constexpr int C = Conv::Co;
public:
    using type = mp_push_front<ShapeList, FeatureShape<H, W, D, C>>;
    // using type = FeatureShape<H,W,D,C>;
};

template <typename ShapeList, // mp_list< FeatureShape<...>>
          typename ConvConfig> // ConvConfig<...>
using conv_apply_concate = typename::conv_apply_concate_impl<ShapeList, ConvConfig>::type;

template<int H, int W, int D, int C>
std::ostream & operator<<(std::ostream & os, const FeatureShape<H, W, D, C> &shape)
{
    return os << "<" << shape.H << " " << shape.W << " " << shape.D << " " << shape.C << ">";
}

template<class... T>
void print(mp_list<T...> const &l) {
    std::size_t const N = sizeof...(T);
    mp_for_each<mp_iota_c<N>>( [&] (auto I) {
        // I is mp_size_t<0> ...
        // std::cout << std::get<I>(l) << std::endl;
        std::cout << mp_at_c<mp_list<T...>, I>()  << std::endl;
    });
}


