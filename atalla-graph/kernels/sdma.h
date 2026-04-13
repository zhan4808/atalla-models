#ifndef KERNELS_SDMA_H
#define KERNELS_SDMA_H

#define ATALLA_TILE 32
#define SCPAD_BYTES 1048576u /* 1 MB */
#define SCPAD_TILE_MAX 512

typedef struct {
    int valid;
    int dirty;
    int tensor_id;
    int tile_index;
    int sid;
    int scpad_addr;
    int rows;
    int cols;
    int full_cols;
} ScpadTile;

typedef struct {
    unsigned used_bytes;
    ScpadTile entries[SCPAD_TILE_MAX];
} ScpadState;

static inline int _tile_dim(int total, int tile_i) {
    int start = tile_i * ATALLA_TILE;
    int rem = total - start;
    if (rem <= 0) return 0;
    return rem < ATALLA_TILE ? rem : ATALLA_TILE;
}

static inline int tile_rows(int total_rows, int tile_i) {
    return _tile_dim(total_rows, tile_i);
}

static inline int tile_cols(int total_cols, int tile_j) {
    return _tile_dim(total_cols, tile_j);
}

static inline int sdma_control(int sid, int rows, int cols, int full_cols) {
    return ((sid & 0x3) << 30)
         | (((rows > 0 ? rows : 1) - 1) << 25)
         | (((cols > 0 ? cols : 1) - 1) << 20)
         | (((full_cols > 0 ? full_cols : 1) - 1) & 0xFFFFF);
}

static inline unsigned tile_base_offset_bytes(unsigned row0, unsigned col0, unsigned ld) {
    return ((row0 * ld) + col0) * 2u;
}

#endif /* KERNELS_SDMA_H */
