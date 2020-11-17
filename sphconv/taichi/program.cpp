#include <taichi/lang.h>
#include <numeric>
// #include <taichi/visual/gui.h>
// #include <torch/extension.h>
#include <string>
#include <vector>
#include "npy.hpp"
#include <cmath>

using namespace taichi::Tlang;

Program *prog;

void init_taichi_program() {
    taichi::CoreState::set_trigger_gdb_when_crash(false);
}


struct Points {
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> data;
};

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

// convert  pytorch points data to taichi data

struct VoxelizationConfig {
    int v_res = 64;
    int h_res = 512;
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
};


static inline bool in_range(int x, int low, int high) {
    return x >= low && x < high;
}

int main() {

    Points points = gen_points_data(0, 0);

    auto prog = new Program(Arch::gpu);


    init_taichi_program();

    int n = 128;

    Global(layer1, f32);
    Global(layer2, f32);
    Global(layer3, f32);
    Global(layer4, f32);

    Global(weights1, f32);
    Global(weights2, f32);
    Global(weights3, f32);


    VoxelizationConfig vcfg;

    int block_size = 4;
    int num_ch1 = 16;
    int num_ch2 = 16;
    int num_ch3 = 16;
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

        root.dense(ijkl, {4, 4, 4, num_ch1 * num_ch2}).place(weights1);
        root.dense(ijkl, {4, 4, 4, num_ch2 * num_ch3}).place(weights2);
        root.dense(ijkl, {4, 4, 4, num_ch3 * num_ch4}).place(weights3);
    });

    // init layer1 data

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


    Kernel(forward_conv1).def([&]{
        bool use_cache = true;
        CacheL1(weights1);
        BlockDim(256);

        For (layer2, [&](Expr i, Expr j, Expr k, Expr c_out) {
            auto sum = Var(0.0f);
            for (int c_in = 0; c_in < num_ch1; c_in++) {
                for (int dx = -1; dx < 2; dx++) {
                    for (int dy = -1; dy < 2; dy++) {
                        for (int dz = -1; dz < 2; dz++) {
                            auto weight = weights1[Expr(dx + 1), Expr(dy + 1), Expr(dz + 1), c_in * num_ch2 + c_out];
                            auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;
                            sum += weight * layer1[i + dx, j + dy, k + dz, c_in2];
                        }
                    }
                }
            }
            layer2[i, j, k, c_out] = sum;
        });
    });

    Kernel(forward_conv2).def([&]{
        bool use_cache = true;
        CacheL1(weights2);
        BlockDim(256);

        For (layer2, [&](Expr i, Expr j, Expr k, Expr c_out) {
            auto sum = Var(0.0f);
            for (int c_in = 0; c_in < num_ch1; c_in++) {
                for (int dx = -1; dx < 2; dx++) {
                    for (int dy = -1; dy < 2; dy++) {
                        for (int dz = -1; dz < 2; dz++) {
                            auto weight = weights1[Expr(dx + 1), Expr(dy + 1), Expr(dz + 1), c_in * num_ch2 + c_out];
                            auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;
                            sum += weight * layer2[i + dx, j + dy, k + dz, c_in2];
                        }
                    }
                }
            }
            layer3[i, j, k, c_out] = sum;
        });
    });


    Kernel(forward_conv3).def([&]{
        bool use_cache = true;
        CacheL1(weights2);
        BlockDim(256);

        For (layer2, [&](Expr i, Expr j, Expr k, Expr c_out) {
            auto sum = Var(0.0f);
            for (int c_in = 0; c_in < num_ch1; c_in++) {
                for (int dx = -1; dx < 2; dx++) {
                    for (int dy = -1; dy < 2; dy++) {
                        for (int dz = -1; dz < 2; dz++) {
                            auto weight = weights1[Expr(dx + 1), Expr(dy + 1), Expr(dz + 1), c_in * num_ch2 + c_out];
                            auto c_in2 = use_cache ? AssumeInRange(c_in, c_out, 0, 1) : c_in;
                            sum += weight * layer3[i + dx, j + dy, k + dz, c_in2];
                        }
                    }
                }
            }
            layer4[i, j, k, c_out] = sum;
        });
    });



    kernel([&] {
        BlockDim(256);
        kernel_name("dilate");
        For(layer1, [&](Expr i, Expr j, Expr k) {
            If(i % block_size == 0 && j % block_size == 0 && block_size == 0)
                .Then([&] {
                    for (int x = -1; x < 2; x++) {
                        for (int y = -1; y < 2; y++) {
                            for (int z = -1; z < 2; z++) {
                                layer2[i + x * block_size, j + y * block_size, k + z * block_size, 0] = 0.0f; // activate the block
                            }
                        }
                    }
                });
        });
    })();

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

    for (int i = 0; i < 50; i++) {
        forward_conv1();
        forward_conv2();
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


