
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <vector>
#include <taichi/lang.h>
#include <boost/mp11.hpp>
#include <time.h>
#include "utils.h"
#include <cuda_runtime.h>


using namespace boost::mp11;
using namespace taichi::Tlang;
namespace py = pybind11;



#include "kernel.h"
#include "backbone.h"
#include "utils.h"
#include "copy.h"

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
    size_t n_layers() { return N_layer + 1; }
    size_t n_weights() { return N_layer; }
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
                       {Shape::H / block_size, Shape::W / block_size,
                        Shape::D / block_size, 1})
                .pointer()
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
                                       Conv::S0, Conv::S1, Conv::S2,
                                       Conv::P0, Conv::P1, Conv::P2,
                                       ShapeOut::H, ShapeOut::W, ShapeOut::D,
                                       Conv::subm>(
                        layers[I], layers[I + 1]))));

        forward_convs.push_back(
            std::move(
                get_current_program()
                    .kernel("forward_conv" + I)
                    .def(convolution<Conv::K0, Conv::K1, Conv::K2,
                                     Conv::S0, Conv::S1, Conv::S2,
                                     Conv::P0, Conv::P1, Conv::P2,
                                     Conv::Ci, Conv::Co,
                                     Shape::H, Shape::W, Shape::D>(
                        layers[I], layers[I + 1], weights[I]))));
    });
}


/**
 *  init the backbone taichi program
 */
void init(bool debug) {
    if (current_program != nullptr) {
        return;
    }
    prog.reset(new Program(Arch::gpu));
    prog->config.debug = debug;
    prog->config.lower_access = false;
    init_taichi_program();
    backbone.reset(new Backbone());
}


/**
 * forward
 *
 * NOTE: do not print to std::cout and std::err, in this function
 */
void forward(torch::Tensor output,
             torch::Tensor points,
             std::vector<torch::Tensor> weights)
{

    // printf("number of weights = %d\n", (int)weights.size());
    // copy weights data

    // TC_INFO("copy weights ");
    mp_for_each<mp_iota_c<N_layer>>([&] (auto I) {
        using Conv = mp_at<BackBoneConvConfigs, decltype(I)>;
        copy_weights_cpu<Conv>(backbone->weights[I], weights[I]);
        // PyObject *p = PyTuple_GetItem(weights, I);gg
    });

    // copy points data
    // TC_INFO("init layer0 ");
    init_layer0(backbone->layers[0], points, vcfg);

    //
    // TC_INFO("forward ");
    for (size_t i = 0; i < N_layer; i++) {
        // TC_INFO("  layer {}", i);
        backbone->activate_convs[i]();
        backbone->forward_convs[i]();
    }


    // fill data to output
    // TC_INFO("copy_feature_cpu");
    using Shape = mp_at<BackBoneShape_mp,mp_int<N_layer>>;
    // copy_feature_cpu<Shape>(backbone->layers[N_layer], output);
}

void backward(torch::Tensor grad)
{
    printf(" backward not implemented. \n");
    return;
}

void print_last_layer()
{
    using Shape = mp_at_c<BackBoneShape_mp, N_layer>;
    print_3d<taichi::float32>(backbone->layers.back(),
    Shape::H,  Shape::W, Shape::D, 0);
}

void print_first_layer()
{
    using Shape = mp_at_c<BackBoneShape_mp, 0>;
    print_3d<taichi::float32>(backbone->layers[0],
    Shape::H,  Shape::W, Shape::D, 0);
}

void print_first_layer_nz()
{
    using Shape = mp_at_c<BackBoneShape_mp, 0>;
    print_3d_nz<taichi::float32>(backbone->layers[0],
    Shape::H,  Shape::W, Shape::D, 0);
}

void print_last_layer_nz()
{
    using Shape = mp_at_c<BackBoneShape_mp, N_layer>;
    print_3d_nz<taichi::float32>(backbone->layers.back(),
    Shape::H,  Shape::W, Shape::D, 0);
}

// void print_layer(const int i)
// {
//     assert(i >= 0 && i < N_layer + 1, "layer number invalid");
//     using Shape = mp_at_c<BackBoneShape_mp, i>;
//     print_3d<taichi::float32>(backbone->layers.back(),
//     Shape::H, Shape::W, Shape::D, 0);
// }

void profiler_print() {
    prog->profiler_print();
}
