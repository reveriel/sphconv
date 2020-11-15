#include <taichi/lang.h>
#include <numeric>
// #include <taichi/visual/gui.h>

TLANG_NAMESPACE_BEGIN

auto foo = []( std::vector<std::string> args)
{
    CoreState::set_trigger_gdb_when_crash(true);
    Program prog(Arch::gpu);

    Global(a, i32);
    auto i = Index(0);

    layout([&]() { root.dense(i, 128).place(a); });

    kernel([&]() {
        Matrix A(2, 2), B(2, 2);
        A(0, 0) = 1;
        A(0, 1) = 1;
        A(1, 0) = 1;
        A(1, 1) = 1;

        B(0, 0) = 1;
        B(0, 1) = 2;
        B(1, 0) = 3;
        B(1, 1) = 4;
        auto C = Var(A * B + A);
        Assert(C(0, 0) == 5);
        Assert(C(0, 1) == 7);
        Assert(C(1, 0) == 5);
        Assert(C(1, 1) == 7);
    })();
};

TC_REGISTER_TASK(foo);

int main_() {
    // for (auto &kv : InterfaceHolder::get_instance()->methods)
    // {
    //     kv.second(&m);
    // }
    Task *t = new Task_foo();
    std::vector<std::string> args = {"foo"};
    Config config = Config();
    config.set("1", "foo");
    t->initialize(config);
    t->run(args);
    return 0;
}

TLANG_NAMESPACE_END

int main() {
    taichi::Tlang::main_();
    return 0;
}


