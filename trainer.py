import jax
import jax.numpy as jnp
import optax
from flax import struct
from typing import Tuple, Any

# Fix the import to the new 'UnSwag' structure
from unswag.core import sophia_forward

# Use a frozen dataclass for clean state management on TPU
@struct.dataclass
class TrainState:
    params: dict
    opt_state: Any
    
def create_train_state(rng, dim, hidden, rank, learning_rate=3e-4):
    """Initializes the Clean Room training state."""
    from unswag.core import init_sophia_weights
    
    # Initialize Adapter Weights (A, B)
    A, B = init_sophia_weights(rng, dim, hidden, rank)
    params = {'A': A, 'B': B}
    
    # Use AdamW - The Gold Standard for LLMs
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)
    
    return TrainState(params=params, opt_state=opt_state), optimizer

@jax.jit
def mse_loss_fn(params, x, W, target):
    """
    Calculates Structural Divergence.
    
    Args:
        params: Dict containing {'A': Adapter, 'B': Projection}
        x: Input Activations (Batch, Dim)
        W: Frozen Base Weights (Dim, Hidden)
        target: The 'Truth' (Batch, Hidden)
    """
    # The 'sophia_forward' handles the 1-bit constraint internally
    pred = sophia_forward(x, W, params['A'], params['B'])
    
    # Mean Squared Error: The distance between 'Is' and 'Ought'
    return jnp.mean((pred - target) ** 2)

@jax.jit
def train_step(state: TrainState, x, W, target, optimizer):
    """
    Pure XLA Update Step.
    Fuses Gradient Calculation + 1-Bit Backprop + Optimizer Update.
    """
    # 1. Gradient Calculation (The Audit)
    loss_val, grads = jax.value_and_grad(mse_loss_fn)(state.params, x, W, target)
    
    # 2. Optimizer Step (The Correction)
    # This handles momentum/velocity without you writing the math manually
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    
    # 3. Return the new Integrity State
    new_state = state.replace(params=new_params, opt_state=new_opt_state)
    return loss_val, new_state
