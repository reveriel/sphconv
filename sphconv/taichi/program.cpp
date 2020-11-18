#include "conv.hpp"
#include "npy.hpp"

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

std::vector<ConvolutionConfig> backbone_cfg = {
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 16, 16, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 16, 16, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {2, 2, 2}, 16, 32, false),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 32, 32, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 32, 32, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {2, 2, 2}, 32, 64, false),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 64, 64, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 64, 64, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 64, 64, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {2, 1, 1}, 64, 64, false),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 64, 64, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 64, 64, true),
    ConvolutionConfig({3, 3, 3}, {1, 1, 1}, {1, 1, 1}, 64, 64, true),
    ConvolutionConfig({3, 1, 1}, {1, 0, 0}, {2, 1, 1}, 64, 64, false),
};

auto input_shape = FeatureShape({vcfg.H(), vcfg.W(), vcfg.D(), 16});
auto backbone_shape = BackBoneShape(backbone_cfg, input_shape);


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
* init layer1 from points data,  voxelization is done at here
*/
void init_layer1(Expr &layer1, const Points &points, const VoxelizationConfig &vcfg,  int num_ch1) {

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
            layer1.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 0) = x;
            layer1.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 1) = y;
            layer1.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 2) = z;
            layer1.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 3) = refl;
            // for testing
            for (int j = 0; j < num_ch1; j++) {
                layer1.val<taichi::float32>(theta_idx, phi_idx, depth_idx, j) = refl;
            }
        }
    }
}

int main() {

    Points points = gen_points_data(0, 0);

    auto prog = new Program(Arch::gpu);


    init_taichi_program();

    int n = 128;

    // Global(layer1, f32);
    //
    Declare(layer1_global); auto layer1 = global_new(layer1_global, DataType::f32);

    Global(layer2, f32);
    Global(layer3, f32);
    Global(layer4, f32);

    Global(weights1, f32);
    Global(weights2, f32);
    Global(weights3, f32);

    auto relu = [](Expr a) { return  max(a, Var(0.0f)); };



    std::cout << backbone_shape;

    int block_size = 4;
    int num_ch1 = 16;
    int num_ch2 = 16;
    int num_ch3 = 32;
    int num_ch4 = 16;


    layout([&]() {
        auto ijkl = Indices(0, 1, 2, 3);
        root.dense(ijkl, {vcfg.v_res / block_size, vcfg.h_res / block_size, vcfg.d_res/ block_size, 1}).bitmasked()
            .dense(ijkl, {block_size, block_size, block_size, num_ch1}).place(layer1);

        root.dense(ijkl, {vcfg.v_res / block_size, vcfg.h_res / block_size, vcfg.d_res / block_size, 1}).bitmasked()
            .dense(ijkl, {block_size, block_size, block_size, num_ch2}).place(layer2);

        root.dense(ijkl, {vcfg.v_res / block_size, vcfg.h_res / block_size, vcfg.d_res / block_size, 1}).bitmasked()
            .dense(ijkl, {block_size, block_size, block_size, num_ch2}).place(layer3);

        root.dense(ijkl, {vcfg.v_res / block_size, vcfg.h_res / block_size, vcfg.d_res / block_size, 1}).bitmasked()
            .dense(ijkl, {block_size, block_size, block_size, num_ch2}).place(layer4);


        root.dense(ijkl, {3, 3, 3, num_ch1 * num_ch2}).place(weights1);
        root.dense(ijkl, {3, 3, 3, num_ch2 * num_ch3}).place(weights2);
        root.dense(ijkl, {3, 3, 3, num_ch3 * num_ch4}).place(weights3);
    });

    // init layer1 data

    init_layer1(layer1, points, vcfg, num_ch1);

    Kernel(activate_conv1).def(conv_activate_subm<4, 4, 4>(layer1, layer2));
    Kernel(forward_conv1).def(convolution<3, 3, 3, 1, 1, 1, 0, 0, 0, 16, 16>(layer1, layer2, weights1));
    Kernel(activate_conv2).def(conv_activate_subm<4, 4, 4>(layer2, layer3));
    Kernel(forward_conv2).def(convolution<3, 3, 3, 1, 1, 1, 0, 0, 0, 16, 16>(layer2, layer3, weights2));
    Kernel(activate_conv3).def(conv_activate_subm<4, 4, 4>(layer3, layer4));
    Kernel(forward_conv3).def(convolution<3, 3, 3, 1, 1, 1, 0, 0, 0, 16, 16>(layer3, layer4, weights3));



    // fill weights1, 串行的？
    for (int c_out = 0; c_out < num_ch2; c_out++) {
        for (int c_in = 0; c_in < num_ch1; c_in++) {
            float inc = 0.1f;
            for (int dx = -1; dx < 2; dx++) {
                for (int dy = -1; dy < 2; dy++) {
                    for (int dz = -1; dz < 2; dz++) {
                        if (dx == 0 && dy == 0 && dz == 0)
                            weights1.val<taichi::float32>(dx + 1, dy + 1, dz + 1, c_in * num_ch2 + c_out) = inc;
                        inc += 0.1f;
                    }
                }
            }
        }
    }

    for (int i = 0; i < 0; i++) {
        activate_conv1();
        forward_conv1();
        activate_conv2();
        forward_conv2();
        activate_conv3();
        forward_conv3();
    }

    prog->profiler_print();



//  // 这厮是 不带参数的！
//     Kernel &kernel_double =
//         kernel([&]() {
//             kernel_name("double");
//             For(0, n, [&](Expr i) {
//                 auto ret = Var(0);
//                 If(i % 2 == 0).Then([&] { ret = dou(i); }).Else([&] { ret = i; });
//                 a[i] = ret;
//             });
//         });

//     kernel_double();

//     for (int i = 0; i < n; i ++) {
//         if (a.val<taichi::int32>(i) == (2 - i % 2) * i) {
//             std::cout << "correct " << i << std::endl;

//             std::cout << "error " << std::endl;
//         }
//     }



    delete prog;

    return 0;
}


