#include <iostream>
#include <vector>
#include <cstdint>
#include "../include/ane/sme_stream.hpp"

using namespace ane;

int main() {
    std::cout << "Initializing 4-step 8-bit matmul loop..." << std::endl;
    // 4 rows of 64 bytes (256 bytes) of Int8 input 
    alignas(64) int8_t vecA[256];
    alignas(64) int8_t vecB[256];
    
    // Fill with sample data
    for(int i=0; i<256; i++) {
        vecA[i] = 1;
        vecB[i] = 2;
    }
    
    // Output tile: ZA0 is a 16x16 grid of 32-bit floats/ints (1024 bytes = 256 elements)
    alignas(64) int32_t outTile[256] = {0};

    SmeStreamBuilder stream;
    
    // Allocate hardware Virtual Pointers!
    auto ptrA = stream.alloc_ptr(vecA);
    auto ptrB = stream.alloc_ptr(vecB);
    auto ptrOut = stream.alloc_ptr(outTile);

    stream.zero_za();
    
    // Run a hardware-interpreted loop for 4 combinations
    stream.loop(4, [&](SmeStreamBuilder& s) {
        // Automatically increments pointer A by 64 bytes each loop
        s.load_z_b(0, ptrA, 64);
        
        // Automatically increments pointer B by 64 bytes each loop
        s.load_z_b(1, ptrB, 64);
        
        // Perform 8-bit matrix outer product accumulating into za0.s
        s.smopa(0, 0, 1);
    });

    // Extract the tile
    stream.store_za_s(0, ptrOut, 0);

    std::cout << "Executing stream hardware DSL..." << std::endl;
    stream.execute();
    
    std::cout << "Execution completed!" << std::endl;
    // Expected logic for first element:
    // SMOPA is outer product accumulation. Each iteration multiplies a 64-elem col from A by 64-elem row from B.
    // 4 iterations. Each elem A is 1, each B is 2. 1*2 = 2.
    // Accumulate 4 times = 8. wait!
    // SMOPA for Int8 processes groups of 4 values (dot product of 4-element vectors) 
    // inside the 32-bit output tile elements. 
    // For every output 32-bit pixel, it computes 4 products (1*2 * 4 = 8) per tile execution.
    // Over 4 loop iterations, it accumulates 4 times. 8 * 4 = 32!
    std::cout << "Result Check (outTile[0]): " << outTile[0] << " (Expected 32!)" << std::endl;
    return 0;
}