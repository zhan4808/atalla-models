#include <stdint.h>

#include "../scratchpad/scratchpad_alloc.h"

void test_kernel() {
    void* ptr0 = _alloc_scpad0(4, 32);

    uintptr_t scpad_addr = (uintptr_t)ptr0;
    uintptr_t gmem_addr = 0x10000000; // arbitrary DRAM address (will later be passed in)

    asm volatile(
        "li.s $6, 1\n\t"            // sid = 1 
        "li.s $7, 4\n\t"            // num_rows
        "li.s $8, 32\n\t"           // num_cols
        "mv.s $5, %[gmem]\n\t"      // rs2 <- global memory base
        "mv.s $4, %[scpad]\n\t"     // rs1/rd1 <- scratchpad base from allocator
        "sdma 0, $6, $8, $5, $4\n\t" // issue sdma: sid, rows/cols, rs2, rs1
        :
        : [gmem] "r"(gmem_addr), [scpad] "r"(scpad_addr)
        : "$4", "$5", "$6", "$7", "$8");

    reset_scratchpads();
}
