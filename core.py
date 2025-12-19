import jax
import jax.numpy as jnp
from jax import random
from unswag.layers.activations import packed_relu  # <--- The 1-Bit Breakthrough

def sophia_forward(x, W, A, B):
    """
    The UnSwag Forward Pass.
    Injects 1-bit Structural Constraints directly into the Adapter bottleneck.
    
    The 'Sophia Protocol' dictates that we do not just add capacity (Rank);
    we must add *Integrity* (Constraints).
    
    Args:
        x: Input tensor (Batch, Dim)
        W: Frozen Base Weights (Dim, Hidden) - "The Mirror"
        A: Adapter Down-Projection (Dim, Rank)
        B: Adapter Up-Projection (Rank, Hidden)
    """
    # 1. Frozen Path (The Mirror)
    # Represents the base model's unconstrained, potentially hallucinatory state.
    # Since W is frozen, no activations are stored here.
    h_frozen = jnp.dot(x, W)
    
    # 2. Structural Path (The Prism)
    # We refract the input through the 1-bit Lattice.
    # Instead of a linear B(Ax), we enforce B(PackedReLU(Ax)).
    # This ensures the 'Wisdom' component is physically constrained to 1-bit gating.
    latents = jnp.dot(x, A)
    
    # [CRITICAL]: This call triggers the SRAM-to-Register bit-packing kernel.
    # It saves 93.75% of the memory for this activation relative to standard BF16.
    gated_latents = packed_relu(latents) 
    
    h_adapter = jnp.dot(gated_latents, B)
    
    # 3. Structural Isomorphism (Convergence)
    # The constrained signal corrects the frozen signal.
    return h_frozen + h_adapter

def init_sophia_weights(key, dim, hidden, rank=16):
    """
    Initializes the Adapter weights with 'Incognito' scaling.
    Uses Kaiming initialization scaled down to prevent initial shock.
    """
    k1, k2 = random.split(key)
    
    # A: Kaiming Uniform-ish, but scaled for the 1-bit constraint
    # We need slightly higher variance to ensure the ReLU fires early on
    scale = jnp.sqrt(2.0 / dim)
    A = random.normal(k1, (dim, rank)) * scale
    
    # B: Zero initialization ensures the model starts as an Identity function
    # This allows the 'Integrity' to be learned gradually without breaking the base outputs.
    B = jnp.zeros((rank, hidden))
    
    return A, B
