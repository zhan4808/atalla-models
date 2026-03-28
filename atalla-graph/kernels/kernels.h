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

int relu_kernel(const GlobalTile *input, GlobalTile *output, const void *vector_reg_base);
int softmax_kernel(const GlobalTile *input, GlobalTile *output);
int matmul_kernel(const GlobalTile *A, const GlobalTile *B, GlobalTile *C);
int add_kernel(const GlobalTile *lhs, const GlobalTile *rhs, GlobalTile *dst, float alpha);

#endif /* KERNELS_KERNELS_H */
