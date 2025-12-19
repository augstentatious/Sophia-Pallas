# UnSwag
UnSwag is a high-efficiency fine-tuning library designed for the JAX/TPU ecosystem.  While libraries like Unsloth optimize for CUDA, UnSwag leverages the XLA compiler and Pallas kernels to bring "Unsloth-like" memory efficiency to Google Colab TPUs and Cloud TPU v5 pods.  Zero-Dependency on CUDA.  Pure JAX/Flax implementation.

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

   [!] STATUS: EXPERIMENTAL // PRE-ALPHA
   [!] ARCH: JAX / FLAX / PALLAS
   [!] TARGET: TPU v2-8 to TPU v5e
```
### ⚡ Proof of Convergence (TPU v5e)
*Status: Validated on Google Colab TPU Runtime*

Pre-Alpha training run demonstrating successful gradient flow through the Adapter (A/B) matrices while maintaining frozen base weights (W).

| STEP | LOSS       | STATUS        |
|:-----|:-----------|:--------------|
| 0    | 675.42     | 🚀 START      |
| 1    | 515.91     | 📉 CONVERGING |
| ...  | ...        | ...           |
| 9    | 200.70     | ✨ RESONANCE  |

## 🔧 Technical Architecture: The 1-Bit Isomorphism

UnSwag introduces a **Structural Isomorphism** between boolean logic and TPU memory tiling.

### 1. The "Bitpack" Kernel
Standard ReLU activations consume **16 bits** (bfloat16) per element, despite carrying only **1 bit** of information (gating). UnSwag implements a custom Pallas kernel that:
- **Fetches** 1024-element blocks from HBM to SRAM.
- **Computes** the sign bit in the Vector Processing Unit (VPU).
- **Packs** the resulting boolean mask into `uint32` integers (32x compression ratio relative to the sign data).
- **Commits** only the packed integer mask back to HBM.

### 2. SRAM-Resident Backprop
During the backward pass, UnSwag avoids "rematerializing" the full activation tensor. Instead, it:
- Loads the packed `uint32` mask.
- Unpacks it directly into the VPU registers.
- Fuses the gradient gating (`grad * mask`) within the same kernel cycle.
- **Result:** 93.75% reduction in activation memory footprint.

>*"We don't optimize the model; we optimize the physics of the data movement."*


