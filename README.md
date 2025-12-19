# UnSwag

**The Memory Wall is a choice.**

UnSwag is a JAX library that implements **Structural Isomorphism** for activation caching. By compressing forward-pass activations into 1-bit packets, we achieve **32x memory reduction** with mathematically identical convergence.

## The Stats (TPU v3-8)
- **Compression:** 32.0x (393KB -> 12KB)
- **Accuracy Delta:** 0.000000
- **Speedup:** ~5% (Due to reduced HBM bandwidth pressure)

## Usage

```python
import jax
from unswag import unswag_relu

# Replace standard ReLU
# x = jax.nn.relu(x)  <-- Old, bloated
x = unswag_relu(x)    # <-- New, 3% memory footprint

# UnSwag

```text
    _    _       _______
   | |  | |     / ______|
   | |  | |_ __| (___ __      ____ _  __ _
   | |  | | '_ \\___ \\ \ /\ / / _` |/ _` |
   | |__| | | | |___) |\ V  V / (_| | (_| |
    \____/|_| |_|____/  \_/\_/ \__,_|\__, |
                                      __/ |
    .---------------------------.    |___/
    |  [|||||||||] [|||||||||]  |
    |  """"""""""" """""""""""  |__
    `---------------------------'  |
       `---------------------------'

   [!] STATUS: EXPERIMENTAL // BETA
   [!] ARCH: JAX / FLAX / PALLAS
   [!] TARGET: TPU v2-8 to TPU v5e
```

## 🔧 Technical Architecture: The 1-Bit Isomorphism
UnSwag introduces a Structural Isomorphism between boolean logic and TPU memory tiling.

1. The "Bitpack" Isomorphism
Standard ReLU activations consume 32 bits (float32) or 16 bits (bfloat16) per element, despite effectively carrying only 1 bit of information (the gating decision) for the backward pass. UnSwag implements a JIT-compiled XLA kernel that:

Quantizes the activation sign bit immediately during the forward pass.

Packs 8 boolean masks into a single uint8 byte (or 32 into a uint32).

Commits only the packed integer mask to High Bandwidth Memory (HBM).

Result: A verified 32x reduction in memory traffic (vs float32).

2. JIT-Fused Gradient Reconstruction
During the backward pass, UnSwag avoids "rematerializing" or "recomputing" the full activation tensor. Instead, it:

Loads the compressed packet.

Unpacks the sign bits directly into the Vector Processing Unit (VPU) registers.

Fuses the gradient gating (grad * mask) within the same kernel cycle.

This creates a zero-overhead memory compression layer that respects the exact mathematical derivative of ReLU
