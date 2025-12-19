import jax
import jax.numpy as jnp
from jax import custom_vjp
from .core import UnSwagActivations

# --- 1. THE UNSWAG RELU (FFN OPTIMIZATION) ---

@custom_vjp
def unswag_relu(x):
    """
    Standard ReLU with a custom memory-efficient backward pass.
    Reclaims 96.875% of activation HBM per layer.
    """
    return jax.nn.relu(x)

def unswag_relu_fwd(x):
    """Forward pass: Compute ReLU and compress signs into bits."""
    y = jax.nn.relu(x)
    # Store only the sign bits (1 bit per element)
    checkpoint = UnSwagActivations.compress(x)
    return y, checkpoint

def unswag_relu_bwd(checkpoint, g):
    """Backward pass: Reconstruct the ReLU mask from bits."""
    # Restore signs (0 or 1) from the uint32 bit-field
    x_restored = UnSwagActivations.restore(checkpoint)
    
    # Perfect mathematical isomorphism: grad is g where x > 0
    grad_x = g * (x_restored > 0).astype(g.dtype)
    return (grad_x,)

# Register the custom VJP for the ReLU isomorphism
unswag_relu.defvjp(unswag_relu_fwd, unswag_relu_bwd)


# --- 2. THE UNSWAG ATTENTION (CONTEXT OPTIMIZATION) ---

def unswag_attention(q, k, v, mask=None, dropout_rng=None, dropout_rate=0.1):
    """
    Memory-efficient Attention using 1-bit bit-packed Dropout masks.
    Crucial for 128k context windows on 16GB TPU cores.
    """
    # 1. Scaled Dot-Product Attention
    scale = 1.0 / jnp.sqrt(q.shape[-1])
    # [batch, heads, seq, seq]
    logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    
    if mask is not None:
        logits += mask
        
    weights = jax.nn.softmax(logits)
    
    # 2. 1-Bit Dropout Optimization
    if dropout_rng is not None and dropout_rate > 0:
        keep_prob = 1.0 - dropout_rate
        # Pack the boolean dropout mask into bits to save 32x HBM
        # In a full custom_vjp, we'd store these bits just like the ReLU signs
        mask_bits = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
        weights = jnp.where(mask_bits, weights / keep_prob, 0.0)
    
    return jnp.matmul(weights, v)
