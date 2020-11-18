#pragma once

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
 * describe the parameters of a convolution
 */
struct ConvolutionConfig {
    std::array<int, 3> kernel_size;
    std::array<int, 3> padding;
    std::array<int, 3> stride;
    int channel_in;
    int channel_out;
    bool subm; // is submanifold convolution
    ConvolutionConfig(const std::array<int, 3> &kernel_size_,
                      const std::array<int, 3> &padding_,
                      const std::array<int, 3> &stride_,
                      int channel_in_, int channel_out_,
                      bool subm_)
        : kernel_size(kernel_size_), padding(padding_), stride(stride_),
          channel_in(channel_in_), channel_out(channel_out_), subm(subm_) {}
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
    FeatureShape conv(const ConvolutionConfig &cfg) {
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

struct BackBoneShape {
    std::vector<FeatureShape> shapes;
    BackBoneShape(const std::vector<ConvolutionConfig> &backbone_cfg,
        const FeatureShape &input_shape)
    {
        shapes.push_back(input_shape);
        for (auto conv_cfg : backbone_cfg) {
            shapes.push_back(shapes.back().conv(conv_cfg));
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const BackBoneShape & bbshape);
};

std::ostream &operator<<(std::ostream &os, const FeatureShape &fshape) {
    return os << "Tensor[" << fshape.h() << ", " << fshape.w() << ", " << fshape.d() << ", " << fshape.c() << "]";
}

std::ostream & operator<<(std::ostream & os, const BackBoneShape &bbshape)
{
    for (auto shape : bbshape.shapes) {
        os << shape << std::endl;
    }
    return os;
}
