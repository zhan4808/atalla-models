#ifndef SCRATCHPAD_ALLOC_H
#define SCRATCHPAD_ALLOC_H

#include <stdint.h>

#define SCPAD_SIZE_BYTES (1 * 1024 * 1024) // 1MB per scratchpad
#define BF16_SIZE 2
#define ALIGNMENT 64

void* _alloc_scpad0(int rows, int cols);
void* _alloc_scpad1(int rows, int cols);
void reset_scratchpads();

#endif // SCRATCHPAD_ALLOC_H
