
#include <taichi/lang.h>
#include <numeric>
// #include <taichi/visual/gui.h>
// #include <torch/extension.h>
#include <string>
#include <vector>
#include <cmath>

// test if convolution get a correct result

using namespace taichi::Tlang;

Program *prog;

void init_taichi_program() {
    taichi::CoreState::set_trigger_gdb_when_crash(true);
}

int main() {

    bool gpu = true;
    auto prog = new Program(gpu ? Arch::gpu : Arch::x86_64);
    init_taichi_program();

    Global(layer1, f32);

    int block_size = 4;
    int num_ch1 = 16;
    int num_ch2 = 1;

    int res = 8;

    const int LEN = 64;

    layout([&]() {
        auto i = Indices(0);
        root.dense(i, {LEN}).place(layer1);
    });

    for (int i = 0; i < LEN; i++) {
        layer1.val<taichi::float32>(i) = i + 1;
    }

    // negative, warp back
    // outof bound, warp back
    // 因为taichi 给 coord 做了一个模运算
    // 所以 -1 变成了63
    for (int i = -1; i <= LEN*2; i++) {
        printf("layer1.val(%d) = %f\n", i, layer1.val<taichi::float32>(i));
    }
    delete prog;

    return 0;
}



