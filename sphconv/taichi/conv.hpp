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
          delta_theta(radians(v_range[1] - v_range[1]) / v_res),
          delta_r(log ? (std::log(d_range[1]) - std::log(d_range[0])) / d_res
                      : (d_range[1] - d_range[0]) / d_res)
    {
    }

    constexpr int H() const { return v_res; }
    constexpr int W() const { return h_res; }
    constexpr int D() const { return d_res; }

};

static inline bool in_range(int x, int low, int high) {
    return x >= low && x < high;
}

// data read from numpy file
struct Points {
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> data;
};


/**
 *  base class for configrations
 */
struct ConvolutionConfigBase {
    std::array<int, 3> kernel_size;
    std::array<int, 3> padding;
    std::array<int, 3> stride;
    int channel_in;
    int channel_out;
    bool subm; // is submanifold convolution

    ConvolutionConfigBase(const std::array<int, 3> kernel_size_,
                            const std::array<int, 3> padding_,
                            const std::array<int, 3> stride_,
                            int channel_in_, int channel_out_, bool subm_)
        : kernel_size(kernel_size_), padding(padding_), stride(stride_),
          channel_in(channel_in_), channel_out(channel_out_), subm(subm_) {
          }
};

/**
 * describe the parameters of a convolution
 */
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
struct ConvolutionConfig : public ConvolutionConfigBase
{
    using Base = ConvolutionConfigBase;
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

    ConvolutionConfig()
        : Base({K0, K1, K2}, {P0, P1, P2}, {S0, S1, S2}, Ci, Co, subm_) { }
};

/***
 * descripte the feature map size
 * also do the convolution shape inference
 */
struct FeatureShapeBase {
    // HWDC
    const int s0;
    const int s1;
    const int s2;
    const int s3;
    std::array<int, 4> shape;

    constexpr FeatureShapeBase(const std::array<int, 4> &shape_) :
    s0(shape_[0]), s1(shape_[1]), s2(shape_[2]), s3(shape_[3]),  shape(shape_) {};

    constexpr FeatureShapeBase(int s0_, int s1_, int s2_, int s3_) :
    s0(s0_), s1(s1_), s2(s2_), s3(s3_), shape({s0_, s1_, s2_, s3_}) {}

    constexpr FeatureShapeBase(const FeatureShapeBase &o) :
    s0(o.s0), s1(o.s1), s2(o.s2), s3(o.s3), shape(o.shape) {}


    int const &h() const { return shape[0]; }
    int const &w() const { return shape[1]; }
    int const &d() const { return shape[2]; }
    int const &c() const { return shape[3]; }

    constexpr int hc() const { return s0; };
    constexpr int wc() const { return s1; };
    constexpr int dc() const { return s2; };
    constexpr int cc() const { return s3; };

    /**
     * compute the output FeatureShape after applying a convlution
     */
    constexpr FeatureShapeBase conv(const ConvolutionConfigBase &cfg) const {
        // assertm((this->cc() == cfg.channel_in), "channel size not match");
        if (cfg.subm)
            return *this;

        return FeatureShapeBase(
            (this->hc() + 2 * cfg.padding[0] - cfg.kernel_size[0]) / cfg.stride[0] + 1,
            (this->wc() + 2 * cfg.padding[1] - cfg.kernel_size[1]) / cfg.stride[1] + 1,
            (this->dc() + 2 * cfg.padding[2] - cfg.kernel_size[2]) / cfg.stride[2] + 1,
            cfg.channel_out
        );
    }
    template<typename ConvConfig>
    constexpr FeatureShapeBase conv() const {
        using Conv = ConvConfig;
        assertm((this->cc() == Conv::Co), "channel size not match");
        if (Conv::subm)
            return *this;
        return FeatureShapeBase(
            (this->hc() + 2 * Conv::P0 - Conv::K0) / Conv::S0 + 1,
            (this->wc() + 2 * Conv::P1 - Conv::K1) / Conv::S1 + 1,
            (this->dc() + 2 * Conv::P2 - Conv::K2) / Conv::S2 + 1,
            Conv::Co
        );
    }
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

/**
 * ConfigList:
 */
template <typename ConfigList,
          int LEN = mp_size<ConfigList>::value + 1>
struct BackBoneShape
{
    std::array<FeatureShapeBase, LEN> shapes;

    BackBoneShape(const FeatureShapeBase &input_shape)
    {
        shapes[0] = (input_shape);
        int i = 0;
        mp_for_each<ConfigList>([&](auto ConvConfig) {
            using Conv = decltype(ConvConfig);
            Conv conv;
            shapes[i+1] = (shapes[i].conv(conv));
            i++;
        });
    }

    size_t size() {
        return shapes.size();
    }

    FeatureShapeBase operator[](size_t i) {
        return shapes[i];
    }

    template<typename T>
    friend std::ostream& operator<<(std::ostream& os, const BackBoneShape<T> & bbshape);
};

std::ostream &operator<<(std::ostream &os, const FeatureShapeBase &fshape) {
    return os << "Tensor[" << fshape.h() << ", " << fshape.w() << ", " << fshape.d() << ", " << fshape.c() << "]";
}

// template<typename T>
// std::ostream & operator<<(std::ostream & os, const BackBoneShape<T> &bbshape)
// {
//     for (auto shape : bbshape.shapes) {
//         os << shape << std::endl;
//     }
//     return os;
// }

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


template<typename BackBoneConfigType_>
struct BackBoneConfig {
    using BackBoneConfigType = BackBoneConfigType_;

    std::vector<ConvolutionConfigBase> convs;
    BackBoneConfig () {
        mp_for_each<BackBoneConfigType>(
        [&](auto Config){
            using Conv = decltype(Config);
            Conv conv;
            convs.push_back(conv);
        });
    }
    ConvolutionConfigBase operator[](size_t i) const { return convs[i]; };
    size_t size() const { return convs.size(); }
};

