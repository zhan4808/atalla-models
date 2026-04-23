#ifndef KERNELS_KERNELS_H
#define KERNELS_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define MAX_TENSOR_RANK 8

typedef struct {
    uint32_t base_addr;
} TileDesc32;

typedef struct {
    uint8_t rank;
    uint32_t shape[MAX_TENSOR_RANK];
    uint32_t tiles_per_dim[MAX_TENSOR_RANK];
    size_t count;
    const TileDesc32 *tiles;
} GlobalTile;

/* Compute-only tile kernels */
int matmul_kernel(int a_scpad_addr, int b_scpad_addr, int c_scpad_addr, int m_rows, int n_cols, int k_cols);
int relu_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols);
int softmax_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols);
int add_kernel(int lhs_scpad_addr, int rhs_scpad_addr, int out_scpad_addr, int rows, int cols);
int conv_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols,
                int kernel_h, int kernel_w, int stride_h, int stride_w,
                int pad_h, int pad_w, int dilation_h, int dilation_w, int groups);
int maxpool_kernel(int in_scpad_addr, int out_scpad_addr, int rows, int cols,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w, int ceil_mode);

#endif /* KERNELS_KERNELS_H */
