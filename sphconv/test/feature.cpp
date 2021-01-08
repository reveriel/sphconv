#include <feature.h>

#include <cstdio>

#include <cstdint>

void test() {
    {
    Feature<float, uint32_t, 2, 2, 2, 2, 2> a0;
    Feature<float, uint32_t, 2, 2, 2, 2, 4> a1;
    Feature<float, uint32_t, 2, 2, 2, 7, 8> a2;
    Feature<float, uint32_t, 2, 3, 2, 2, 2> a3;
    Feature<float, uint32_t, 2, 2, 7, 2, 2> a4;
    Feature<float, uint32_t, 1, 1, 1, 13, 1> a5;
    Feature<float, uint32_t, 1, 1, 13, 13, 1> a6;
    Feature<float, uint32_t, 1, 11, 13, 13, 1> a7;
    }
    Feature<float, uint32_t, 2, 2, 2, 2, 2> a;

    a.set(0,0,0,1,1, 24.f) ;
    printf("get %.2f\n", a.get(0,0,0,1,1) );
    a.print();

}

int main() {
    test();
    printf("hello\n");
    return 0;


}