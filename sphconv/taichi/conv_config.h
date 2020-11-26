#pragma once

#include "conv.hpp"

#include <boost/mp11.hpp>
using namespace boost::mp11;

constexpr VoxelizationConfig vcfg;

constexpr int block_size = 4;

using BackBoneConvConfigs =
mp_list<
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 16, 16, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 16, 16, true>
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 2, 2, 2, 16, 32, false>
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 32, 32, true>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 32, 32, true>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 2, 2, 2, 32, 64, false>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 2, 1, 1, 64, 64, false>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    // ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    // ConvolutionConfig<3, 1, 1, 0, 0, 0, 2, 1, 1, 64, 64, false>
>;

// the number of convs
constexpr int N_layer = mp_size<BackBoneConvConfigs>::value;

using BackBoneShape_mp = mp_reverse<mp_fold<
    BackBoneConvConfigs,
    mp_list<FeatureShape<vcfg.H(),vcfg.W(), vcfg.D(), 16>>,
    conv_apply_concate
    >>;
