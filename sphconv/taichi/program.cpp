#include <taichi/lang.h>
#include <numeric>
// #include <taichi/visual/gui.h>

TLANG_NAMESPACE_BEGIN

auto foo = []( std::vector<std::string> args)
{
    CoreState::set_trigger_gdb_when_crash(true);
    Program prog(Arch::gpu);
    int n = 128;

    Global(a, i32);
    auto i = Index(0);

    layout([&]() { root.dense(i, 128).place(a); });

    auto dou = [](Expr a) { return a * 2; };

    kernel([&]() {
        For(0, n, [&](Expr i) {
            auto ret = Var(0);
            If(i % 2 == 0).Then([&] { ret = dou(i); }).Else([&] { ret = i;});
            a[i] = ret;
        });
    })();

    for (int i = 0; i < n; i ++) {
        if (a.val<int32>(i) == (2 - i % 2) * i) {
                ;
            std::cout << "correct " << i << std::endl;
        } else  {
            std::cout << "error " << std::endl;

        }
    }
};

// TC_REGISTER_TASK(foo);

int main_() {
    // for (auto &kv : InterfaceHolder::get_instance()->methods)
    // {
    //     kv.second(&m);
    // }
    // Task *t = new Task_foo();
    std::vector<std::string> args = {"foo"};
    // Config config = Config();
    // config.set("1", "foo");
    // t->initialize(config);
    // t->run(args);
    // return 0;
    foo(args);
}

TLANG_NAMESPACE_END

int main() {
    taichi::Tlang::main_();
    return 0;
}


