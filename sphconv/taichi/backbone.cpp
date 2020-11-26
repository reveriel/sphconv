
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstdlib>
#include <vector>
#include <taichi/lang.h>
#include <boost/mp11.hpp>

using namespace boost::mp11;
using namespace taichi::Tlang;
namespace py = pybind11;


#include "kernel.h"
#include "backbone.h"
#include "utils.h"

// load convolution configurations
#include "conv_config.h"


struct Backbone {
    // declare all variables
    std::vector<Expr> layers; // of size N_layer + 1
    std::vector<Expr> weights; // of size N_layer
    // define kernels
    std::vector<Program::Kernel> activate_convs;
    std::vector<Program::Kernel> forward_convs;
    Backbone();
} ;

///
// global taichi program
std::unique_ptr<Program> prog;
// the backbone
std::unique_ptr<Backbone> backbone;

Backbone::Backbone()
{
    layers.push_back(declare_global("layer0", DataType::f32));
    for (size_t i = 1; i <= N_layer; i++)
    {
        layers.push_back(declare_global("layer" + i, DataType::f32));
        weights.push_back(declare_global("weights" + i, DataType::f32));
    }

    // define data layout
    layout([&]() {
        auto ijkl = Indices(0, 1, 2, 3);

        mp_for_each<mp_iota_c<N_layer + 1>>([&](auto I) {
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

    // define kernerls
    mp_for_each<mp_iota_c<N_layer>>([&](auto I) {
        using Conv = mp_at<BackBoneConvConfigs, decltype(I)>;
        using Shape = mp_at<BackBoneShape_mp, decltype(I)>;
        using ShapeOut = mp_at<BackBoneShape_mp, decltype(std::integral_constant<std::size_t, I + 1>())>;
        activate_convs.push_back(
            std::move( // NOTE: we move the Kernel from Prgram's public member, luckliy, it's OK
                get_current_program()
                    .kernel("activate_conv" + I)
                    .def(conv_activate<block_size, block_size, block_size,
                                       Conv::K0, Conv::K1, Conv::K2,
                                       Conv::P0, Conv::P1, Conv::P2,
                                       Conv::S0, Conv::S1, Conv::S2,
                                       ShapeOut::H, ShapeOut::W, ShapeOut::D,
                                       Conv::subm>(
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
}


/**
 *  init the backbone taichi program
 */
void init() {
    if (current_program != nullptr) {
        return;
    }
    prog.reset(new Program(Arch::gpu));
    init_taichi_program();
    backbone.reset(new Backbone());
}

/**
 * forward
 */
void forward(torch::Tensor output,
             torch::Tensor points,
             py::tuple weights)
{
    printf("number of weights = %d\n", (int)weights.size());
    // copy weights data
    // copy points data

    //
    for (size_t i = 0; i < N_layer; i++) {
        backbone->activate_convs[i]();
        backbone->forward_convs[i]();
    }

    // fill data to output

    return;
}

void backward(torch::Tensor grad)
{
    printf(" backward not implemented. \n");
    return;
}

void profiler_print() {
    prog->profiler_print();
}
