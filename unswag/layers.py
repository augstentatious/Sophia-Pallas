import jax
import jax.numpy as jnp
from jax import custom_vjp
from .core import UnSwagActivations

@custom_vjp
def unswag_relu(x):
    """
    The UnSwag ReLU: 32x Memory Savings.
    Use this exactly like jax.nn.relu(x).
    """
    return jax.nn.relu(x)

def unswag_relu_fwd(x):
    """Forward pass: Compute output, save compressed residual."""
    y = jax.nn.relu(x)
    
    # Compress input 'x' for the backward pass
    checkpoint = UnSwagActivations.compress(x)
    return y, checkpoint

def unswag_relu_bwd(checkpoint, g):
    """Backward pass: Decompress residual to compute gradients."""
    x_restored = UnSwagActivations.restore(checkpoint)
    
    # Standard ReLU Gradient Logic using restored signs
    grad_x = g * (x_restored > 0).astype(g.dtype)
    return (grad_x,)

# Register the VJP
unswag_relu.defvjp(unswag_relu_fwd, unswag_relu_bwd)
