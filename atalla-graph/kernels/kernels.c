#include "kernels.h"

int relu_kernel(const GlobalTile *input, GlobalTile *output, const void *vector_reg_base) {
    (void)input;
    (void)output;
    (void)vector_reg_base;
    return 0;
}

int softmax_kernel(const GlobalTile *input, GlobalTile *output) {
    (void)input;
    (void)output;
    return 0;
}

int matmul_kernel(const GlobalTile *A, const GlobalTile *B, GlobalTile *C) {
    (void)A;
    (void)B;
    (void)C;
    return 0;
}

int add_kernel(const GlobalTile *lhs, const GlobalTile *rhs, GlobalTile *dst, float alpha) {
    (void)lhs;
    (void)rhs;
    (void)dst;
    (void)alpha;
    return 0;
}
