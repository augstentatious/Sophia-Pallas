import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# 1-BIT KERNELS (ReLU Path)
# -----------------------------------------------------------------------------

@triton.jit
def _pack_1bit_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 8
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 8
    
    packed_val = tl.zeros([BLOCK_SIZE], dtype=tl.int8)
    
    for i in range(8):
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        x = tl.load(x_ptr + curr_idx, mask=mask, other=0.0)
        bit = (x > 0).to(tl.int8)
        packed_val = packed_val | (bit << i)

    out_idx = block_start // 8 + tl.arange(0, BLOCK_SIZE)
    out_mask = out_idx < (n_elements + 7) // 8
    tl.store(out_ptr + out_idx, packed_val, mask=out_mask)

@triton.jit
def _unpack_1bit_kernel(
    packed_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 8
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 8
    
    for i in range(8):
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        byte_idx = (curr_idx // 8)
        packed_val = tl.load(packed_ptr + byte_idx, mask=mask, other=0)
        bit = (packed_val >> i) & 1
        val = tl.where(bit == 1, 1.0, -1.0)
        tl.store(out_ptr + curr_idx, val, mask=mask)

# -----------------------------------------------------------------------------
# 2-BIT KERNELS (SiLU/SwiGLU Path)
# -----------------------------------------------------------------------------

@triton.jit
def _pack_2bit_silu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 4
    
    packed_val = tl.zeros([BLOCK_SIZE], dtype=tl.int8)
    
    for i in range(4):
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        x = tl.load(x_ptr + curr_idx, mask=mask, other=0.0)
        
        # bit_0: Sign, bit_1: Magnitude
        bit_0 = (x >= 0).to(tl.int8)
        bit_1 = (tl.abs(x) >= 2.0).to(tl.int8)
        two_bits = (bit_1 << 1) | bit_0
        packed_val = packed_val | (two_bits << (i * 2))

    out_idx = block_start // 4 + tl.arange(0, BLOCK_SIZE)
    out_mask = out_idx < (n_elements + 3) // 4
    tl.store(out_ptr + out_idx, packed_val, mask=out_mask)

@triton.jit
def _unpack_2bit_backward_kernel(
    grad_output_ptr, packed_activation_ptr, grad_input_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * 4
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * 4

    for i in range(4):
        curr_idx = offsets + i
        mask = curr_idx < n_elements
        grad_out = tl.load(grad_output_ptr + curr_idx, mask=mask, other=0.0)
        
        byte_idx = curr_idx // 4
        shift = (curr_idx % 4) * 2
        packed_byte = tl.load(packed_activation_ptr + byte_idx, mask=mask, other=0)
        two_bits = (packed_byte >> shift) & 0b11

        # Approximation: 00->0.0, 01->0.5, 10/11->1.0
        grad_coeff = tl.where(
            two_bits == 0, 0.0,
            tl.where(two_bits == 1, 0.5, 1.0)
        )
        tl.store(grad_input_ptr + curr_idx, grad_out * grad_coeff, mask=mask)

# -----------------------------------------------------------------------------
# PYTHON WRAPPERS
# -----------------------------------------------------------------------------

def pack_activations(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    out_size = (n_elements + 7) // 8
    output = torch.empty(out_size, dtype=torch.int8, device=x.device)
    BLOCK_SIZE = 1024 
    grid = lambda meta: (triton.cdiv(n_elements // 8, meta['BLOCK_SIZE']), )
    _pack_1bit_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

def unpack_activations(packed: torch.Tensor, original_shape, original_dtype=torch.float16):
    n_elements = 1
    for dim in original_shape: n_elements *= dim
    output = torch.empty(original_shape, dtype=original_dtype, device=packed.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements // 8, meta['BLOCK_SIZE']), )
    _unpack_1bit_kernel[grid](packed, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
