
#include <gtest/gtest.h>

#include ""

    void test_nthwc_layout(int n_size, int t_size, int h_size, int w_size, int c_size) {
        int ldc = c_size + 1; // ??
        int ldw = ldc * (w_size + 2);
        int ldh = ldw * (h_size + 3);


    }