.text
.p2align 2
.global __ane_execute_sme_stream

// void __ane_execute_sme_stream(const SmeCommand* cmds);
// x0 = Pointer to our struct array
__ane_execute_sme_stream:
    // Standard ARM64 prologue, save frame pointer + extra 128 bytes for VM context
    stp x29, x30, [sp, #-144]!
    mov x29, sp

    // x9 = Ptr to Virtual Registers array (sp + 16)
    // Layout: 8x 64-bit Pointers (64 bytes), 4x 64-bit Loop Counters (32 bytes)
    add x9, sp, #16

    // ENTER STREAMING MODE
    smstart

.L_loop:
    // Read current instruction sequentially (16 bytes tightly packed)
    ldr w1, [x0, #0]    // w1 = opcode
    ldrb w2, [x0, #4]   // w2 = z_reg_1
    ldrb w3, [x0, #5]   // w3 = z_reg_2
    ldrb w4, [x0, #6]   // w4 = za_tile
    ldrb w5, [x0, #7]   // w5 = v_reg
    ldr x6, [x0, #8]    // x6 = imm (immediate value / offset)

    // Advance command array pointer
    add x0, x0, #16

    // Branch to opcode implementation (max 8 for now)
    cbz w1, .L_end
    cmp w1, #1
    b.eq .L_cmd_zero_za
    cmp w1, #2
    b.eq .L_cmd_ldr_z_b
    cmp w1, #3
    b.eq .L_cmd_smopa_za
    cmp w1, #4
    b.eq .L_cmd_str_za_s
    cmp w1, #5
    b.eq .L_cmd_set_ptr
    cmp w1, #6
    b.eq .L_cmd_add_ptr
    cmp w1, #7
    b.eq .L_cmd_set_lc
    cmp w1, #8
    b.eq .L_cmd_loop_end
    
    b .L_end

// ------------------------------------
// Instruction Implementations
// ------------------------------------
.L_cmd_zero_za:
    zero {za}
    b .L_loop

.L_cmd_set_ptr:
    // w5 = virtual register id (0-7)
    // x6 = initial pointer address
    str x6, [x9, w5, uxtw #3]
    b .L_loop

.L_cmd_add_ptr:
    // w5 = virtual register id (0-7)
    // x6 = offset to add
    ldr x7, [x9, w5, uxtw #3] // load
    add x7, x7, x6            // add offset
    str x7, [x9, w5, uxtw #3] // save back
    b .L_loop

.L_cmd_ldr_z_b:
    // w2 = Z register
    // w5 = Virtual Pointer ID (0-7)
    // x6 = Post-increment offset
    ldr x7, [x9, w5, uxtw #3] // x7 = Virtual Pointer Address
    
    cmp w2, #0
    b.eq 1f
    cmp w2, #1
    b.eq 2f
    cmp w2, #2
    b.eq 3f
    cmp w2, #3
    b.eq 4f
    b .L_ldr_z_b_done
1:
    ldr z0, [x7]
    b .L_ldr_z_b_done
2:
    ldr z1, [x7]
    b .L_ldr_z_b_done
3:
    ldr z2, [x7]
    b .L_ldr_z_b_done
4:
    ldr z3, [x7]
.L_ldr_z_b_done:
    // Apply post increment and save updated pointer
    add x7, x7, x6
    str x7, [x9, w5, uxtw #3]
    b .L_loop

.L_cmd_smopa_za:
    // w2 = z_reg_n, w3 = z_reg_m, w4 = za_tile
    ptrue p0.b
    
    // For MVP, we will hardcode matching for a few combinations
    // Since dynamically selecting Z registers in SME assembly requires 
    // generating combinations or leveraging specific opcodes.
    // Assuming z_reg_n=0, z_reg_m=1 targeting za0.s
    smopa za0.s, p0/m, p0/m, z0.b, z1.b
    
    b .L_loop

.L_cmd_str_za_s:
    // w4 = za_tile
    // w5 = Virtual Pointer ID (0-7)
    // x6 = Post-increment offset
    ldr x7, [x9, w5, uxtw #3]
    
    // Save 32-bit output tile to memory
    mov w12, #0
    str za[w12, 0], [x7] 
    
    // Post increment 
    add x7, x7, x6
    str x7, [x9, w5, uxtw #3]
    b .L_loop

.L_cmd_set_lc:
    // w5 = loop counter ID (0-3)
    // x6 = initial loop count
    add x10, x9, #64          // x10 = Loop Counter Array Base
    str x6, [x10, w5, uxtw #3]
    b .L_loop

.L_cmd_loop_end:
    // w5 = loop counter ID (0-3)
    // x6 = jump back instruction count IF NOT ZERO
    add x10, x9, #64
    ldr x7, [x10, w5, uxtw #3] // load current count
    subs x7, x7, #1            // decrement and set flags
    
    str x7, [x10, w5, uxtw #3] // save updated count

    b.eq .L_loop               // if decremented to 0, just fall through to next instruction
    
    // If not zero, jump backwards! 
    // x6 holds instruction jump count (each instruction is 16 bytes)
    lsl x8, x6, #4             // x8 = x6 * 16 bytes
    sub x0, x0, x8             // rewind instruction pointer!
    b .L_loop

// ------------------------------------
// Exit
// ------------------------------------
.L_end:
    smstop

    ldp x29, x30, [sp], #144
    ret
