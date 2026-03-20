#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <functional>

namespace ane {

// Our custom VM Opcodes mapped to SME instructions
enum class SmeOpcode : uint32_t {
    END = 0,         // Clean exit
    ZERO_ZA = 1,     // Zero out the entire ZA array
    LDR_Z_B = 2,     // Load 8-bit data into a Z register from a virtual pointer
    SMOPA_ZA = 3,    // 8-bit Signed Matrix Outer Product
    STR_ZA_S = 4,    // Store a 32-bit ZA tile to memory via a virtual pointer
    SET_PTR = 5,     // Initialize a virtual pointer
    ADD_PTR = 6,     // Add offset to virtual pointer
    SET_LC = 7,      // Initialize a loop counter
    LOOP_END = 8     // Decrement loop counter and conditionally jump back
};

// 16-byte packed instruction
struct alignas(16) SmeCommand {
    SmeOpcode opcode;  // 4 bytes (Offset 0)
    uint8_t   z_reg_1; // 1 byte  (Offset 4)
    uint8_t   z_reg_2; // 1 byte  (Offset 5)
    uint8_t   za_tile; // 1 byte  (Offset 6)
    uint8_t   v_reg;   // 1 byte  (Offset 7) - Virtual register ID (p0-p7 or lc0-lc3)
    int64_t   imm;     // 8 bytes (Offset 8) - Immediate value, pointer, or jump offset
};

// The raw assembly interpreter function
extern "C" void _ane_execute_sme_stream(const SmeCommand* cmds);

// Opaque handles representing hardware virtual registers 
struct VPtr { uint8_t id; };
struct LCont { uint8_t id; };

// The C++ DSL / Builder Interface
class SmeStreamBuilder {
private:
    std::vector<SmeCommand> stream;
    uint8_t next_ptr = 0;
    uint8_t next_lc = 0;

public:
    // ---------------------------------
    // Context Allocation
    // ---------------------------------
    
    // Allocate one of 8 available hardware virtual pointers
    VPtr alloc_ptr(const void* initial_address = nullptr) {
        if (next_ptr >= 8) throw std::runtime_error("Exceeded max 8 virtual pointers in hardware");
        VPtr p = {next_ptr++};
        if (initial_address) set_ptr(p, initial_address);
        return p;
    }

    // Allocate one of 4 available hardware loop counters
    LCont alloc_lc() {
        if (next_lc >= 4) throw std::runtime_error("Exceeded max 4 loop counters in hardware");
        return {next_lc++};
    }

    void reset() {
        stream.clear();
        next_ptr = 0;
        next_lc = 0;
    }

    // ---------------------------------
    // Mathematical Operations (SME)
    // ---------------------------------
    void zero_za() {
        stream.push_back({SmeOpcode::ZERO_ZA, 0, 0, 0, 0, 0});
    }

    void load_z_b(uint8_t z_reg, VPtr ptr, int64_t post_increment = 64) {
        stream.push_back({SmeOpcode::LDR_Z_B, z_reg, 0, 0, ptr.id, post_increment});
    }

    void store_za_s(uint8_t tile_id, VPtr ptr, int64_t post_increment = 256) {
        stream.push_back({SmeOpcode::STR_ZA_S, 0, 0, tile_id, ptr.id, post_increment});
    }

    void smopa(uint8_t tile_id, uint8_t z_reg_n, uint8_t z_reg_m) {
        stream.push_back({SmeOpcode::SMOPA_ZA, z_reg_n, z_reg_m, tile_id, 0, 0});
    }

    // ---------------------------------
    // Pointer and Loop Control (DSL)
    // ---------------------------------
    void set_ptr(VPtr ptr, const void* dest) {
        stream.push_back({SmeOpcode::SET_PTR, 0, 0, 0, ptr.id, reinterpret_cast<int64_t>(dest)});
    }

    void add_ptr(VPtr ptr, int64_t offset) {
        stream.push_back({SmeOpcode::ADD_PTR, 0, 0, 0, ptr.id, offset});
    }

    // Lambda-based C++ DSL Loop mapped to hardware bytecode jumps!
    void loop(uint32_t count, std::function<void(SmeStreamBuilder&)> body) {
        if (count == 0) return; // Prevent zero-running loops mathematically
        
        LCont lc = alloc_lc();
        
        // Push the SET_LC instruction to initialize the loop counter register
        stream.push_back({SmeOpcode::SET_LC, 0, 0, 0, lc.id, static_cast<int64_t>(count)});
        
        // Snapshot the start state of the stream to measure the jump distance later
        size_t start_idx = stream.size();
        
        // Execute the user's closure to populate the body of the loop
        body(*this);
        
        // If the body was empty, no ops generated, so remove SET_LC and skip loop overhead
        if (stream.size() == start_idx) {
            stream.pop_back(); 
            return;
        }

        // Calculate jump back size in instructions. 
        // When LOOP_END executes, the instruction pointer has technically already 
        // advanced 1 block past LOOP_END. So the offset covers the body + LOOP_END
        int64_t jump_back_cmds = stream.size() - start_idx + 1;
        
        stream.push_back({SmeOpcode::LOOP_END, 0, 0, 0, lc.id, jump_back_cmds});
    }

    // ---------------------------------
    // Interpreter Execution Bound
    // ---------------------------------
    void execute() {
        // Appends the halt instruction, hands memory to assembly, and resets
        stream.push_back({SmeOpcode::END, 0, 0, 0, 0, 0});
        _ane_execute_sme_stream(stream.data());
        stream.pop_back(); 
    }
};

} // namespace ane