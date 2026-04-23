#ifndef KERNELS_SDMA_H
#define KERNELS_SDMA_H

#define ATALLA_TILE 32

/* Atalla intrinsics (here for C correctness) */
void scpad_load(int x, int y, int z);
void scpad_store(int x, int y, int z);

static inline int sdma_tile_dim(int total, int tile_i) {
    int start = tile_i * ATALLA_TILE;
    int rem = total - start;
    if (rem <= 0) return 0;
    return rem < ATALLA_TILE ? rem : ATALLA_TILE;
}

static inline int sdma_control(int rows, int cols, int full_cols) {
    return (((rows > 0 ? rows : 1) - 1) << 25)
         | (((cols > 0 ? cols : 1) - 1) << 20)
         | (((full_cols > 0 ? full_cols : 1) - 1) & 0xFFFFF);
}

#endif /* KERNELS_SDMA_H */
