import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def prepare_data(
    data_path,
    batch_size=8,
    seq_len=32768,
    d_model=3584,
    chunk_size=8192,
    tokenizer_name='google/gemma-2-9b' # Use the real tokenizer for Sophia
):
    """
    Full data preparation pipeline with TPU Device Sharding.
    """
    # 1. Load and Tokenize (Standard CPU logic)
    texts = load_jsonl_data(data_path)
    
    num_samples = len(texts)
    if num_samples < batch_size:
        print(f"\n⚠️  Only {num_samples} samples, repeating to fill batch...")
        repeats = (batch_size + num_samples - 1) // num_samples
        texts = (texts * repeats)[:batch_size]
    else:
        texts = texts[:batch_size]
    
    token_ids, vocab_size = tokenize_texts(texts, tokenizer_name, seq_len)
    
    # 2. Setup the TPU Mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))
    
    # We shard the CHUNK dimension to keep per-core indices low
    # This prevents the 3.75B element overflow
    data_sharding = NamedSharding(mesh, P('data', None, None, None))
    
    # 3. Create Embeddings in bfloat16
    embeddings = create_embeddings(token_ids, d_model, vocab_size)
    
    # 4. Chunk and Shard
    data_chunked = chunk_sequences(embeddings, chunk_size)
    
    # 5. Move to TPU Mesh
    # This 'device_put' is what actually sends the data to the 8 cores
    sharded_data = jax.device_put(data_chunked, data_sharding)
    
    print(f"🦁 Sharding Complete: Data is now distributed across {len(devices)} cores.")
    return sharded_data
