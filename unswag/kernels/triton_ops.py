import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# TRITON KERNELS (The Firmware)
# -----------------------------------------------------------------------------

@triton.jit
def _pack_1bit_kernel(
    x_ptr,              # Pointer to input (FP16/FP32 activations)
    out_ptr,            # Pointer to output (INT8 packed)
    n_elements,         # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to pack 1-bit activations.
    Maps positive values -> 1, negative values -> 0.
    Packs 8 values into one INT8 byte.
    """
    # 1. Map the program ID to the data block
    pid = tl.program_id(axis=0)
    
    # We process BLOCK_SIZE * 8 elements per thread block
    # because each byte represents 8 original elements.
    block_start = pid * BLOCK_SIZE * 8
    
    # Generate offsets for the 8 bits we need to pack
    # shape: [BLOCK_SIZE]
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 8
    
    # 2. Loop to pack 8 bits into 1 byte
    # We construct the byte by iterating 8 times (unrolled)
    packed_val = tl.zeros([BLOCK_SIZE], dtype=tl.int8)
    
    for i in range(8):
        # Load the specific element for this bit position
        # Mask ensures we don't read out of bounds
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        
        # Load data (FP16/FP32)
        x = tl.load(x_ptr + curr_idx, mask=mask, other=0.0)
        
        # Get Sign: 1 if x > 0 else 0
        # (This is the "UnSwag" 1-bit quantization logic)
        bit = (x > 0).to(tl.int8)
        
        # Shift and accumulate into the packed byte
        # Bit 0 goes to position 0, Bit 1 to position 1, etc.
        packed_val = packed_val | (bit << i)

    # 3. Store the result
    # Output size is n_elements / 8
    out_idx = block_start // 8 + tl.arange(0, BLOCK_SIZE)
    out_mask = out_idx < (n_elements + 7) // 8
    
    tl.store(out_ptr + out_idx, packed_val, mask=out_mask)


@triton.jit
def _unpack_1bit_kernel(
    packed_ptr,         # Pointer to input (INT8 packed)
    out_ptr,            # Pointer to output (FP16/FP32 reconstructed)
    n_elements,         # Total number of unpacked elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Unpacks INT8 bytes back into FP16 values (-1.0, 1.0).
    Used for the forward pass MatMul or backward pass.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 8
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 8
    
    # Load packed bytes
    # Note: We need to load the byte corresponding to each group of 8
    packed_idx = block_start // 8 + tl.arange(0, BLOCK_SIZE)
    # This part requires careful indexing in Triton to broadcast the byte
    # For simplicity v1, we iterate the bits again

    @triton.jit
def _pack_2bit_silu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Each thread block processes BLOCK_SIZE bytes, which is BLOCK_SIZE * 4 elements
    block_start = pid * BLOCK_SIZE * 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 4
    
    packed_val = tl.zeros([BLOCK_SIZE], dtype=tl.int8)
    
    for i in range(4): # 4 elements per byte
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        x = tl.load(x_ptr + curr_idx, mask=mask, other=0.0)
        
        # 2-bit Encoding Logic:
        # bit_0: Sign (x >= 0)
        # bit_1: Magnitude (|x| >= 2.0)
        bit_0 = (x >= 0).to(tl.int8)
        bit_1 = (tl.abs(x) >= 2.0).to(tl.int8)
        
        # Combine: bit_1 is the 2nd bit, bit_0 is the 1st
        two_bits = (bit_1 << 1) | bit_0
        
        # Shift and accumulate (i*2 because each takes 2 bits)
        packed_val = packed_val | (two_bits << (i * 2))

    out_idx = block_start // 4 + tl.arange(0, BLOCK_SIZE)
    out_mask = out_idx < (n_elements + 3) // 4
    tl.store(out_ptr + out_idx, packed_val, mask=out_mask)
    
    for i in range(8):
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        
        # Load the packed byte for this group
        # (Optimization: Load once per 8, but here we load per bit for simplicity)
        byte_idx = (curr_idx // 8)
        packed_val = tl.load(packed_ptr + byte_idx, mask=mask, other=0)
        
        # Extract the specific bit
        # If bit is 1 -> +1.0, if bit is 0 -> -1.0
        bit = (packed_val >> i) & 1
        val = tl.where(bit == 1, 1.0, -1.0)
        
        tl.store(out_ptr + curr_idx, val, mask=mask)
        
# -----------------------------------------------------------------------------
# PYTHON WRAPPERS
# -----------------------------------------------------------------------------

def pack_activations(x: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for the packing kernel.
    Input: Tensor (FP16/32)
    Output: Tensor (INT8), size reduced by 8x (stored) -> 32x vs FP32
    """
    n_elements = x.numel()
    # Output size: ceil(n / 8)
    out_size = (n_elements + 7) // 8
    output = torch.empty(out_size, dtype=torch.int8, device=x.device)
    
    # Tuning: BLOCK_SIZE represents how many BYTES we process per thread block
    BLOCK_SIZE = 1024 
    grid = lambda meta: (triton.cdiv(n_elements // 8, meta['BLOCK_SIZE']), )
    
    _pack_1bit_kernel[grid](
        x, output, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def unpack_activations(packed: torch.Tensor, original_shape, original_dtype=torch.float16):
    """
    Python wrapper for unpacking.
    """
    n_elements = 1
    for dim in original_shape:
        n_elements *= dim
        
    output = torch.empty(original_shape, dtype=original_dtype, device=packed.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements // 8, meta['BLOCK_SIZE']), )
    
    _unpack_1bit_kernel[grid](
        packed, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
