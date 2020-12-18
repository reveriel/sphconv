
#include "kernel.h"
#include "utils.h"
#include "conv_config.h"

#include <taichi/lang.h>
#include <numeric>
// #include <taichi/visual/gui.h>
#include <string>
#include <vector>
#include <cmath>
#include "backbone.h"

#include <torch/extension.h>
#include <torch/script.h>

// test backbone
int main() {
    init();
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .requires_grad(false);
        // .device(torch::kCUDA, 0)

    std::cout <<  "output shape  =  " << OutFeatureShape() << std::endl;

    // torch::Tensor output = torch::zeros({128, 128, 3, 64}, options);
    torch::Tensor output = torch::zeros({OutFeatureShape::W,
                                         OutFeatureShape::D,
                                         OutFeatureShape::H,
                                         OutFeatureShape::C},
                                        options);

    // torch::Tensor weight1 = torch::ones({16, 4, 3, 3, 3}, options);
    // assert(weight1.dtype() == torch::kFloat32);
    // assert(weight1.device().type() == torch::kCUDA); // or device().is_cuda()
    // assert(weight1.device().index() == 0);

    std::vector<torch::Tensor> weights;
    mp_for_each<mp_iota_c<N_layer>>([&](auto I) {
        using Conv = mp_at<BackBoneConvConfigs, decltype(I)>;
        weights.push_back(torch::ones(
            {Conv::Co, Conv::Ci, Conv::K0, Conv::K1, Conv::K2}, options));
    });

    torch::jit::script::Module tensors = torch::jit::load("points0.pt");
    torch::Tensor points = tensors.attr("points").toTensor();

    forward(output, points, weights);

    auto t = get_time();
    for (int i = 0; i < 50; i++) {
        forward(output, points, weights);
    }

    profiler_print();

    fprintf(stdout, "total forward time: %.1f ms\n", 1000 * (get_time() - t));

    std::cout << "finished" << std::endl;

}


