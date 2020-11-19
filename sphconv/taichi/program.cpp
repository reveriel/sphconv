#include "conv.hpp"
#include "npy.hpp"
#include "mp_helper.h"

#include <taichi/lang.h>
#include <numeric>
// #include <taichi/visual/gui.h>
// #include <torch/extension.h>

using namespace taichi::Tlang;

Program *prog;

VoxelizationConfig vcfg;

void init_taichi_program() {
    taichi::CoreState::set_trigger_gdb_when_crash(false);
}

// create points data,  on GPU, of shape (N, 4)
Points gen_points_data(int num_points, int num_channel ) {

    std::string points_file_name = "points0.npy";
    // numpy array

    Points points;
    npy::LoadArrayFromNumpy(points_file_name, points.shape, points.fortran_order, points.data);
    std::cout << "shape: ";
    for (size_t i = 0; i < points.shape.size(); i++)
        std::cout << points.shape[i] << ", ";
    std::cout << std::endl;
    // shape: shape: 20285, 4,

    return points;
}


using BackBoneConfigType =
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
    ConvolutionConfig<3, 1, 1, 0, 0, 1, 2, 1, 1, 64, 64, false>
>;

// BackBoneConfig backbone_cfg;


BackBoneConfig<BackBoneConfigType> backbone_cfg;
auto input_shape = FeatureShape({vcfg.H(), vcfg.W(), vcfg.D(), 16});
auto backbone_shape = BackBoneShape<BackBoneConfigType>(input_shape);

/**
 *  fill weights with value
 */
void fill_weights(Expr weights, ConvolutionConfigBase conv, float value) {
    for (int c_out = 0; c_out < conv.channel_out; c_out++) {
        for (int c_in = 0; c_in < conv.channel_in; c_in++) {
            // float inc = 0.1f;
            for (int i = 0; i < conv.kernel_size[0]; i++) {
                for (int j = 0; j < conv.kernel_size[1]; j++) {
                    for (int k = 0; k < conv.kernel_size[2]; k++) {
                        weights.val<taichi::float32>(i, j, k, c_in * conv.channel_out + c_out ) = value;
                    }
                }
            }
            // for (int dx = -1; dx < 2; dx++) {
            //     for (int dy = -1; dy < 2; dy++) {
            //         for (int dz = -1; dz < 2; dz++) {
            //             // if (dx == 0 && dy == 0 && dz == 0)
            //             weights.val<taichi::float32>(dx + 1, dy + 1, dz + 1, c_in * conv.channel_out + c_out) = value;
            //             // inc += 0.1f;
            //         }
            //     }
            // }
        }
    }
}


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

        For(layer_in, [&](Expr i, Expr j, Expr k, Expr c_out) {
            auto sum = Var(0.0f);
            for (int c_in = 0; c_in < channel_in; c_in++) {
                for (int dx = -1; dx < 2; dx++) {
                    for (int dy = -1; dy < 2; dy++) {
                        for (int dz = -1; dz < 2; dz++) {

                            auto weight = weights[Expr(dx + 1), Expr(dy + 1), Expr(dz + 1), c_in * channel_out + c_out];

                            auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;

                            sum += weight * layer_in[i + dx, j + dy, k + dz, c_in2];
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

    Points points = gen_points_data(0, 0);

    auto prog = new Program(Arch::gpu);
    init_taichi_program();

    // declare all varialbes
    std::vector<Expr> layers;
    std::vector<Expr> weights;
    layers.push_back(declare_global("layer0", DataType::f32));
    for (size_t i = 1; i <= backbone_cfg.size(); i++) {
        // auto conv_cfg = backbone_cfg[i];
        layers.push_back(declare_global("layer" + i, DataType::f32));
        weights.push_back(declare_global("weights" + i, DataType::f32));
    }

    std::cout << backbone_shape;

    constexpr int block_size = 4;

    assert((backbone_shape.size() == layers.size()));
    assert((backbone_cfg.size() == weights.size()));

    layout([&]() {
        auto ijkl = Indices(0, 1, 2, 3);
        for (size_t i = 0; i < backbone_shape.size(); i++) {
            auto shape = backbone_shape[i];
            root.dense(ijkl, {shape.h()/block_size, shape.w()/block_size, shape.d()/block_size, 1 })
            .dense(ijkl, {block_size, block_size, block_size, shape.c()})
            .place(layers[i]);
        }
        for (size_t i = 0; i < backbone_cfg.size(); i++) {
            auto conv = backbone_cfg[i];
            root.dense(ijkl, {conv.kernel_size[0], conv.kernel_size[1], conv.kernel_size[2],
            conv.channel_in * conv.channel_out}).place(weights[i]);
        }
    });

    // init layer1 data

    init_layer0(layers[0], points, vcfg, backbone_cfg[0].channel_in);


    std::vector<Program::Kernel> activate_convs;
    std::vector<Program::Kernel> forward_convs;
    mp_for_each<decltype(backbone_cfg)::BackBoneConfigType>([&](auto ConvConfig) {
        using Conv = decltype(ConvConfig);
        size_t i = activate_convs.size();
        activate_convs.push_back(
            std::move(
            get_current_program()
                .kernel("activate_conv" + i)
                .def(conv_activate_subm<block_size, block_size, block_size>(layers[i], layers[i + 1]))));

        forward_convs.push_back(
            std::move(
            get_current_program()
            .kernel("forward_conv" + i)
            .def(convolution<Conv::K0, Conv::K1, Conv::K2,
                            Conv::P0, Conv::P1, Conv::P2,
                            Conv::S0, Conv::S1, Conv::S2,
                            Conv::Ci, Conv::Co>(layers[i], layers[i+1], weights[i]))));
    });

    // fill weights1, 串行的？
    for (auto weight : weights) {
    }

    for (int i = 0; i < 50; i++) {
        for (size_t j = 0; j < backbone_cfg.size(); j++) {
            activate_convs[j]();
            forward_convs[j]();
        }
    }

    prog->profiler_print();



    delete prog;

    return 0;
}


