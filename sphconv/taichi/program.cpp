#include "conv.hpp"
#include "kernel.h"
#include "points.h"
#include "utils.h"

#include <taichi/lang.h>
#include <numeric>
// #include <torch/extension.h>

#include <boost/mp11.hpp>
using namespace boost::mp11;
using namespace taichi::Tlang;

//
// global variables
//
Program *prog;
constexpr VoxelizationConfig vcfg;

constexpr int block_size = 4;

using BackBoneConvConfigs =
mp_list<
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 16, 16, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 16, 16, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 2, 2, 2, 16, 32, false>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 32, 32, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 32, 32, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 2, 2, 2, 32, 64, false>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 2, 1, 1, 64, 64, false>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    ConvolutionConfig<3, 3, 3, 1, 1, 1, 1, 1, 1, 64, 64, true>,
    ConvolutionConfig<3, 1, 1, 0, 0, 0, 2, 1, 1, 64, 64, false>
>;

// the number of convs
constexpr int N_layer = mp_size<BackBoneConvConfigs>::value;

using BackBoneShape_mp = mp_reverse<mp_fold<
    BackBoneConvConfigs,
    mp_list<FeatureShape<vcfg.H(),vcfg.W(), vcfg.D(), 16>>,
    conv_apply_concate
    >>;

/**
 *  fill weights with value
 */
template<typename ConvolutionConfig>
void fill_weights(Expr weights, float value)
{
    using Conv = ConvolutionConfig;
    for (int c_out = 0; c_out < Conv::Co; c_out++) {
        for (int c_in = 0; c_in < Conv::Ci; c_in++) {
            // float inc = 0.1f;
            for (int i = 0; i < Conv::K0; i++) {
                for (int j = 0; j < Conv::K1; j++) {
                    for (int k = 0; k < Conv::K2; k++) {
                        weights.val<taichi::float32>(
                            i, j, k, c_in * Conv::Co + c_out) = value;
                    }
                }
            }
        }
    }
}

/**
 * activate convolution output layer  (dilated version)
 *  in Taichi, sparse data block (like bitmasked) must be activated before writing
 */
template<int block_size0, int block_size1, int block_size2>
std::function<void()> conv_activate_dilate(Expr layer_in, Expr layer_out) {
    return [&]() {
        BlockDim(256);
        kernel_name("dilate");
        For(layer_in, [&](Expr i, Expr j, Expr k) {
            If(i % block_size0 == 0 && j % block_size1 == 0 &&  k % block_size2 == 0)
                .Then([&] {
                    for (int x = -1; x < 2; x++) {
                        for (int y = -1; y < 2; y++) {
                            for (int z = -1; z < 2; z++) {
                                layer_out[i + x * block_size0, j + y * block_size1, k + z * block_size2, 0] = 0.0f; // activate the block
                            }
                        }
                    }
                });
        });
    };
}


/**
 * activate convolution output layer  (submanifold version)
 *  in Taichi, sparse data block (like bitmasked) must be activated before writing
 */
template<int block_size0, int block_size1, int block_size2>
std::function<void()> conv_activate_subm(Expr layer_in, Expr layer_out) {
    return [&]() {
        BlockDim(256);
        kernel_name("submanifold");
        For(layer_in, [&](Expr i, Expr j, Expr k) {
            If(i % block_size0 == 0 && j % block_size1 == 0 &&  k % block_size2 == 0)
                .Then([&] {
                    layer_out[i , j , k , 0] = 0.0f; // activate the block
                });
        });
    };
}


/**
* init layer0 from points data,  voxelization is done at here
*/
void init_layer0(Expr &layer0, const Points &points, const VoxelizationConfig &vcfg,  int num_ch1) {

    for (size_t i = 0; i < points.shape[0]; i++) {
        float x = points.data[i * 4 + 0];
        float y = points.data[i * 4 + 1];
        float z = points.data[i * 4 + 2];
        float refl = points.data[i * 4 + 3];

        float x2y2 = x * x + y * y;
        float r = std::sqrt(x2y2 + z * z);
        float theta = std::acos(z / r);
        float phi = std::asin(y / std::sqrt(x2y2));

        int theta_idx = theta / vcfg.delta_theta;
        int phi_idx = phi / vcfg.delta_phi;

        int depth_idx = vcfg.log ? (std::log(r) / vcfg.delta_r) : ( r / vcfg.delta_r);

        if (in_range(theta_idx, 0, vcfg.v_res)
            && in_range(phi_idx, 0, vcfg.h_res)
            && in_range(depth_idx, 0, vcfg.d_res))
        {
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 0) = x;
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 1) = y;
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 2) = z;
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 3) = refl;
            // for testing
            for (int j = 0; j < num_ch1; j++) {
                layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, j) = refl;
            }
        }
    }
}

auto relu = [](Expr a) { return  max(a, Var(0.0f)); };

inline
Expr declare_global(std::string name, DataType t) {
    auto var_global = Expr(std::make_shared<IdExpression>(name));
    return global_new(var_global, t);
}

int main() {
    //
    // Debug
    //
    print( BackBoneShape_mp() );

    // read points data from npy file, for test.
    Points points = gen_points_data(0, 0);

    auto prog = new Program(Arch::gpu);
    init_taichi_program();

    // declare all variables
    std::vector<Expr> layers; // of size N_layer + 1
    std::vector<Expr> weights; // of size N_layer
    layers.push_back(declare_global("layer0", DataType::f32));
    for (size_t i = 1; i <= N_layer; i++) {
        layers.push_back(declare_global("layer" + i, DataType::f32));
        weights.push_back(declare_global("weights" + i, DataType::f32));
    }

    // define data layout
    layout([&]() {
        auto ijkl = Indices(0, 1, 2, 3);

        mp_for_each<mp_iota_c<N_layer + 1>> ([&] (auto I) {
            using Shape = mp_at<BackBoneShape_mp, decltype(I)>;
            root.dense(ijkl,
                       {Shape::H / block_size, Shape::W / block_size, Shape::D / block_size, 1})
                .bitmasked()
                .dense(ijkl, {block_size, block_size, block_size, Shape::C})
                .place(layers[I]);

        });

        mp_for_each<mp_iota_c<N_layer>>([&](auto I) {
            using Conv = mp_at<BackBoneConvConfigs, decltype(I)>;
            root.dense(ijkl,
                       {Conv::K0, Conv::K1, Conv::K2, Conv::Ci * Conv::Co})
                .place(weights[I]);
        });
    });

    // init input layer's data
    init_layer0(layers[0], points, vcfg, mp_at_c<BackBoneConvConfigs, 0>::Ci);

    //
    // define kernels
    std::vector<Program::Kernel> activate_convs;
    std::vector<Program::Kernel> forward_convs;

    mp_for_each<mp_iota_c<N_layer>>([&](auto I) {
        using Conv = mp_at<BackBoneConvConfigs, decltype(I)>;
        using Shape = mp_at<BackBoneShape_mp, decltype(I)>;
        activate_convs.push_back(
            std::move(
                get_current_program()
                    .kernel("activate_conv" + I)
                    .def(conv_activate_subm<block_size, block_size, block_size>(
                        layers[I], layers[I + 1]))));

        forward_convs.push_back(
            std::move(
                get_current_program()
                    .kernel("forward_conv" + I)
                    .def(convolution<Conv::K0, Conv::K1, Conv::K2,
                                     Conv::P0, Conv::P1, Conv::P2,
                                     Conv::S0, Conv::S1, Conv::S2,
                                     Conv::Ci, Conv::Co,
                                     Shape::H, Shape::W, Shape::D>(
                        layers[I], layers[I + 1], weights[I]))));
    });

    // fill weights1, 串行的？
    for (auto weight : weights) {
    }

    //
    // running
    for (int i = 0; i < 50; i++) {
        for (size_t j = 0; j < N_layer; j++) {
            activate_convs[j]();
            forward_convs[j]();
        }
    }

    prog->profiler_print();

    delete prog;

    return 0;
}


