#include "scratchpad_alloc.h"

static uint8_t* scpad0_base = (uint8_t*)0x0; // base Address
static uint8_t* scpad1_base = (uint8_t*)0x0; 

static uint32_t scpad0_offset = 0;
static uint32_t scpad1_offset = 0;


void* _alloc_impl(uint8_t* base_addr, uint32_t* offset_tracker, int rows, int cols) {
   
    uint32_t bytes_needed = rows * cols * BF16_SIZE;
    
    
    uint32_t current_start = *offset_tracker;
    
    if (current_start + bytes_needed > SCPAD_SIZE_BYTES) {
        return nullptr; // Out of memory
    }
    
    *offset_tracker = current_start + bytes_needed;
    
    return (void*)(base_addr + current_start);
}

void* _alloc_scpad0(int rows, int cols) {
    return _alloc_impl(scpad0_base, &scpad0_offset, rows, cols);
}

void* _alloc_scpad1(int rows, int cols) {
    return _alloc_impl(scpad1_base, &scpad1_offset, rows, cols);
}

// Function to reset scratchpad offsets (at the end of each kernel?)
void reset_scratchpads() {
    scpad0_offset = 0;
    scpad1_offset = 0;
}
