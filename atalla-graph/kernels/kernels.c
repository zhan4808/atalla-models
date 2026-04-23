#include "kernels.h"

int matmul_kernel(
    int a_scpad_addr,
    int b_scpad_addr,
    int c_scpad_addr,
    int m_rows,
    int n_cols,
    int k_cols
) {
    (void)a_scpad_addr;
    (void)b_scpad_addr;
    (void)c_scpad_addr;
    (void)m_rows;
    (void)n_cols;
    (void)k_cols;
    return 0;
}

int relu_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols) {
    (void)in_scpad_addr;
    (void)out_scpad_addr;
    (void)rows;
    (void)cols;
    return 0;
}

int softmax_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols) {
    (void)in_scpad_addr;
    (void)out_scpad_addr;
    (void)rows;
    (void)cols;
    return 0;
}

int add_kernel(int lhs_scpad_addr, int rhs_scpad_addr, int out_scpad_addr, int rows, int cols) {
    (void)lhs_scpad_addr;
    (void)rhs_scpad_addr;
    (void)out_scpad_addr;
    (void)rows;
    (void)cols;
    return 0;
}

int conv_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols,
                int kernel_h, int kernel_w, int stride_h, int stride_w,
                int pad_h, int pad_w, int dilation_h, int dilation_w, int groups) {
    (void)in_scpad_addr;
    (void)out_scpad_addr;
    (void)rows;
    (void)cols;
    (void)kernel_h;
    (void)kernel_w;
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    (void)dilation_h;
    (void)dilation_w;
    (void)groups;
    return 0;
}

int maxpool_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w, int ceil_mode) {
    (void)in_scpad_addr;
    (void)out_scpad_addr;
    (void)rows;
    (void)cols;
    (void)kernel_h;
    (void)kernel_w;
    (void)stride_h;
    (void)stride_w;
    (void)pad_h;
    (void)pad_w;
    (void)dilation_h;
    (void)dilation_w;
    (void)ceil_mode;
    return 0;
}
