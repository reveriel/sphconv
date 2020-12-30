#pragma once

#include <taichi/lang.h>
#include "points.h"

using namespace taichi::Tlang;

void init_taichi_program();

Points gen_points_data(int num_points, int num_channel);

inline
Expr declare_global(std::string name, DataType t) {
    auto var_global = Expr(std::make_shared<IdExpression>(name));
    return global_new(var_global, t);
}

template<typename T>
void print_1d(Expr &arr, int i0, int i1, int N2, int i3) {
    const int N2_max = 30;
    const int N2_head = 20;
    const int N2_tail = N2_max - N2_head;
    if (N2 > N2_max)
    {
        for (int i2 = 0; i2 < N2_head; i2++)
        {
            printf("%4.1f ", arr.val<T>(i0, i1, i2, i3));
        }
        printf(" ... ");
        for (int i2 = N2 - N2_tail; i2 < N2; i2++)
        {
            printf("%4.1f ", arr.val<T>(i0, i1, i2, i3));
        }
    } else {
        for (int i2 = 0; i2 < N2; i2++)
        {
            printf("%4.1f ", arr.val<T>(i0, i1, i2, i3));
        }
    }
    printf("\n");
}

// print
//
template<typename T>
void print_2d(Expr &arr, int i0, int N1, int N2, int i3) {
    const int N1_max = 10;
    const int N1_head = 5;
    const int N1_tail = N1_max - N1_head;

    if (N1 > N1_max) {
        for (int i1 = 0; i1 < N1_head;  i1++)  {
            print_1d<T>(arr, i0, i1, N2, i3);
        }
        printf("... ... ...\n");
        for (int i1 = N1 - N1_tail; i1 < N1; i1++) {
            print_1d<T>(arr, i0, i1, N2, i3);
        }
    } else {
        for (int i1 = 0; i1 < N1; i1++) {
            print_1d<T>(arr, i0, i1, N2, i3);
        }
    }
}

template<typename T>
void print_3d(Expr &arr, int N0, int N1, int N2, int i3) {
    const int N0_max = 10;
    const int N0_head = 5;
    const int N0_tail = N0_max - N0_head;

    if (N0 > N0_max) {
        for (int i0 = 0; i0 < N0_head;  i0++)  {
            printf("i0 = %d\n", i0);
            print_2d<T>(arr, i0, N1, N2, i3);
        }
        printf(" :\n");
        for (int i0 = N0 - N0_tail; i0 < N0; i0++) {
            printf("i0 = %d\n", i0);
            print_2d<T>(arr, i0, N1, N2, i3);
        }
    } else {
        for (int i0 = 0; i0 < N0; i0++) {
            printf("i0 = %d\n", i0);
            print_2d<T>(arr, i0, N1, N2, i3);
        }
    }
}

template<typename T>
void print_1d_nz(Expr &arr, int i0, int i1, int N2, int i3) {
    const int N2_max = 30;
    int count_nz = 0;
    for (int i2 = 0; i2 < N2 && count_nz < N2_max; i2++) {
        T val = arr.val<T>(i0, i1, i2, i3);
        if (val == 0.0f) continue;
        printf("%4.1f ", val);
        count_nz++;
    }
    if (count_nz == 0) {
        return ;
        // printf(" ... ");
    }
    printf("\n");
}

// print
//
template<typename T>
void print_2d_nz(Expr &arr, int i0, int N1, int N2, int i3) {
    const int N1_max = 10;
    const int N1_head = 5;
    const int N1_tail = N1_max - N1_head;

    // if (N1 > N1_max) {
    //     for (int i1 = 0; i1 < N1_head;  i1++)  {
    //         print_1d_nz<T>(arr, i0, i1, N2, i3);
    //     }
    //     printf("... ... ...\n");
    //     for (int i1 = N1 - N1_tail; i1 < N1; i1++) {
    //         print_1d_nz<T>(arr, i0, i1, N2, i3);
    //     }
    // } else {
        for (int i1 = 0; i1 < N1; i1++) {
            print_1d_nz<T>(arr, i0, i1, N2, i3);
        }
    // }
}

template<typename T>
void print_3d_nz(Expr &arr, int N0, int N1, int N2, int i3) {
    const int N0_max = 10;
    const int N0_head = 5;
    const int N0_tail = N0_max - N0_head;

    printf(" print tensor of shape (%d, %d, %d) at channel %d\n", N0, N1, N2, i3);

    // if (N0 > N0_max) {
    //     for (int i0 = 0; i0 < N0_head;  i0++)  {
    //         printf("i0 = %d\n", i0);
    //         print_2d_nz<T>(arr, i0, N1, N2, i3);
    //     }
    //     printf(" :\n");
    //     for (int i0 = N0 - N0_tail; i0 < N0; i0++) {
    //         printf("i0 = %d\n", i0);
    //         print_2d_nz<T>(arr, i0, N1, N2, i3);
    //     }
    // } else {
        for (int i0 = 0; i0 < N0; i0++) {
            printf("i0 = %d\n", i0);
            print_2d_nz<T>(arr, i0, N1, N2, i3);
        }
    // }
}
