#pragma once

#include "mp_helper.h"

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <cassert>
// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))



struct VoxelizationConfig {
    // vertical resolution
    int v_res = 64;
    // horizontal resolution
    int h_res = 512;
    // depth resolution
    int d_res = 512;
    float v_range[2] = {87.5, 103.4};
    float h_range[2] = {-45, 45};
    float d_range[2] = {6, 70.4};
    bool log = true;

    float delta_phi = 0;
    float delta_theta = 0;
    float delta_r = 0;

    constexpr static const float PI = 3.14159265;
    VoxelizationConfig()
    {
        delta_phi = radians(h_range[1] - h_range[0]) / h_res;
        delta_theta = radians(v_range[1] - v_range[1]) / v_res;
        if (log) {
            delta_r = (std::log(d_range[1]) - std::log(d_range[0])) / d_res;
        } else {
            delta_r = (d_range[1] - d_range[0]) / d_res;
        }
    }
    static inline float radians(float degree) { return degree / 180 * PI; }

    const int H() const { return v_res; }
    const int W() const { return h_res; }
    const int D() const { return d_res; }

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

    ConvolutionConfig()
        : Base({K0, K1, K2}, {P0, P1, P2}, {S0, S1, S2}, Ci, Co, subm_) { }
};

/***
 * descripte the feature map size
 * also do the convolution shape inference
 */
struct FeatureShape {
    // HWDC
    std::array<int, 4> shape;
    FeatureShape(const std::array<int, 4> &shape_) : shape(shape_) {}


    int const &h() const { return shape[0]; }
    int const &w() const { return shape[1]; }
    int const &d() const { return shape[2]; }
    int const &c() const { return shape[3]; }

    /**
     * compute the output FeatureShape after applying a convlution
     */
    FeatureShape conv(const ConvolutionConfigBase &cfg) {
        assertm((this->c() == cfg.channel_in), "channel size not match");
        if (cfg.subm)
            return *this;
        return FeatureShape({
            (this->h() + 2 * cfg.padding[0] - cfg.kernel_size[0]) / cfg.stride[0] + 1,
            (this->w() + 2 * cfg.padding[1] - cfg.kernel_size[1]) / cfg.stride[1] + 1,
            (this->d() + 2 * cfg.padding[2] - cfg.kernel_size[2]) / cfg.stride[2] + 1,
            cfg.channel_out
        });
    }

};

/**
 * ConfigList:
 */
template <typename ConfigList>
struct BackBoneShape {
    std::vector<FeatureShape> shapes;

    BackBoneShape(const FeatureShape &input_shape)
    {
        shapes.push_back(input_shape);

        mp_for_each<ConfigList>([&](auto ConvConfig) {
            using Conv = decltype(ConvConfig);
            Conv conv;
            shapes.push_back(shapes.back().conv(conv));
        });
    }

    size_t size() {
        return shapes.size();
    }

    FeatureShape operator[](size_t i) {
        return shapes[i];
    }

    template<typename T>
    friend std::ostream& operator<<(std::ostream& os, const BackBoneShape<T> & bbshape);
};

std::ostream &operator<<(std::ostream &os, const FeatureShape &fshape) {
    return os << "Tensor[" << fshape.h() << ", " << fshape.w() << ", " << fshape.d() << ", " << fshape.c() << "]";
}

template<typename T>
std::ostream & operator<<(std::ostream & os, const BackBoneShape<T> &bbshape)
{
    for (auto shape : bbshape.shapes) {
        os << shape << std::endl;
    }
    return os;
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

