"""
Gemma-9B TPU Training with UnSwag Compression
Train large language models on TPUs with 32K context windows
"""
import jax
import jax.numpy as jnp
import math
from jax.experimental import mesh_utils
from unswag import unswag_relu, boot_sequence
from unswag.ui import monitor
from data_loader import prepare_data

boot_sequence()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model architecture
D_MODEL = 3584      # Embedding dimension
D_FF = 14336        # Feed-forward dimension
SEQ_LEN = 32768     # Sequence length (32K context)
BATCH = 8           # Batch size

# Training
CHUNK_SIZE = 8192   # Chunk size to avoid int32 overflow
NUM_CHUNKS = SEQ_LEN // CHUNK_SIZE
LEARNING_RATE = 1e-5
NUM_EPOCHS = 16

# Data
DATA_PATH = 'data/sophia_dynamic_data.jsonl'
TOKENIZER = 'gpt2'  # Or 'google/gemma-2-9b' if authenticated

print("🔧 Configuration:")
print(f"   Model: {D_MODEL}→{D_FF} (Gemma-9B scale)")
print(f"   Sequence: {SEQ_LEN:,} → {NUM_CHUNKS}×{CHUNK_SIZE:,}")
print(f"   Batch: {BATCH}")
print(f"   Each chunk matmul: {BATCH}×{CHUNK_SIZE}×{D_FF} = {BATCH*CHUNK_SIZE*D_FF:,} elements")


# ============================================================================
# TPU SETUP
# ============================================================================

print("\n⚙️  Setting up TPU mesh...")
devices = mesh_utils.create_device_mesh((8,))
mesh = jax.sharding.Mesh(devices, axis_names=('batch',))

# Sharding specifications
batch_spec = jax.sharding.PartitionSpec(None, 'batch', None, None)
weight_spec = jax.sharding.PartitionSpec(None, None)

batch_sharding = jax.sharding.NamedSharding(mesh, batch_spec)
weight_sharding = jax.sharding.NamedSharding(mesh, weight_spec)

print(f"   ✅ Mesh created with 8 TPUs")
print(f"   Sharding: batch dimension across devices")


# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n🧱 Preparing training data...")
x_sharded = jax.device_put(
    prepare_data(
        data_path=DATA_PATH,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        chunk_size=CHUNK_SIZE,
        tokenizer_name=TOKENIZER
    ),
    batch_sharding
)

print(f"   ✅ Data loaded and sharded")
print(f"   Shape: {x_sharded.shape}")


# ============================================================================
# WEIGHT INITIALIZATION
# ============================================================================

print("\n⚙️  Initializing weights...")
key = jax.random.PRNGKey(42)

w_up = jax.device_put(
    jax.random.normal(key, (D_MODEL, D_FF), dtype=jnp.bfloat16) * 0.02,
    weight_sharding
)

key, subkey = jax.random.split(key)
w_down = jax.device_put(
    jax.random.normal(subkey, (D_FF, D_MODEL), dtype=jnp.bfloat16) * 0.02,
    weight_sharding
)

print(f"   ✅ Weights initialized")
print(f"   Up projection: {w_up.shape}")
print(f"   Down projection: {w_down.shape}")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def process_chunks(w_u, w_d, x_chunks):
    """
    Process all chunks using jax.lax.scan for proper compilation
    
    Args:
        w_u: Up projection weights (d_model, d_ff)
        w_d: Down projection weights (d_ff, d_model)
        x_chunks: Input data (num_chunks, batch, chunk_size, d_model)
    
    Returns:
        Output (num_chunks, batch, chunk_size, d_model)
    """
    batch_size = x_chunks.shape[1]
    chunk_size = x_chunks.shape[2]
    
    def process_single_chunk(carry, chunk):
        """Process one chunk through the network"""
        # Flatten batch dimension for matmul
        chunk_flat = chunk.reshape(-1, D_MODEL)
        
        # Forward pass with UnSwag compression
        gate = jnp.dot(chunk_flat, w_u)
        activated = unswag_relu(gate)  # 32x compression
        out = jnp.dot(activated, w_d)
        
        # Reshape back
        out_reshaped = out.reshape(batch_size, chunk_size, D_MODEL)
        return carry, out_reshaped
    
    # Use scan to iterate over chunks
    _, outputs = jax.lax.scan(process_single_chunk, None, x_chunks)
    return outputs


@jax.jit
def train_step(w_u, w_d, x_chunks):
    """
    Single training step with gradient descent
    
    Args:
        w_u: Up projection weights
        w_d: Down projection weights
        x_chunks: Input data
    
    Returns:
        Updated weights and loss value
    """
    def loss_fn(w):
        out = process_chunks(w, w_d, x_chunks)
        return jnp.mean(jnp.square(out))
    
    loss, grads = jax.value_and_grad(loss_fn)(w_u)
    w_new = w_u - LEARNING_RATE * grads
    
    return w_new, loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*70)
print("🦁 STARTING TRAINING")
print("="*70)
print(f"Dataset: {DATA_PATH}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Total tokens per step: {BATCH * SEQ_LEN:,}")
print("="*70 + "\n")

loss_history = []

for i in monitor(range(NUM_EPOCHS), desc="Training"):
    w_up, loss = train_step(w_up, w_down, x_sharded)
    loss_val = float(loss)
    loss_history.append(loss_val)
    
    if i % 2 == 0:
        print(f" | Epoch {i:02d} | Loss: {loss_val:.6f} | Tokens/step: {BATCH*SEQ_LEN:,}")

print(f"\n✅ TRAINING COMPLETE!")
print(f"   Final Loss: {loss_history[-1]:.6f}")
print(f"   Loss reduction: {loss_history[0] - loss_history[-1]:.6f}")


# ============================================================================
# VERIFICATION
# ============================================================================

print("\n🧪 Verification...")

@jax.jit
def forward_pass(w_u, w_d, x_chunks):
    return process_chunks(w_u, w_d, x_chunks)

full_output = forward_pass(w_up, w_down, x_sharded)
output_reshaped = full_output.transpose(1, 0, 2, 3).reshape(BATCH, SEQ_LEN, D_MODEL)

print(f"✅ Forward pass successful")
print(f"   Output shape: {output_reshaped.shape}")
print(f"   Total elements: {math.prod(output_reshaped.shape):,}")

print("\n🎉 Training pipeline verified and ready!")
print("   Next steps:")
print("   - Add checkpoint saving")
print("   - Load pretrained embeddings")
print("   - Scale to more data")
