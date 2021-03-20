#include <feature.h>

#include <cstdio>

#include <cstdint>
#include "conv.h"

void test()
{
    {
        Feature<float, uint32_t> a0(2,2,2,2,2);
        Feature<float, uint32_t> a1(2,1,2,3,1);
        Feature<float, uint32_t> a2(3,5,1,7,8);
        Feature<float, uint32_t> a3(9,1,5,3,2);
        Feature<float, uint32_t> a4(2,2,7,2,2);
        Feature<float, uint32_t> a5(1,1,1,13,1);
        Feature<float, uint32_t> a6(1,1,13,13,1);
        Feature<float, uint32_t> a7(1,11,13,13,1);
    }

    Feature<float, uint32_t> a(2,2,2,2,2);

    a.set(0, 0, 0, 1, 1, 24.f);
    a.set(0, 0, 0, 1, 0, 23.f);
    a.set(0, 1, 0, 1, 0, 21.f);
    printf("get %.2f\n", a.get(0, 0, 0, 1, 1));
    a.print();
}

int main()
{
    test();
    printf("hello\n");
    return 0;
}