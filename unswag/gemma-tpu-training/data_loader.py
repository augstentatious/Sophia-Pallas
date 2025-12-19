"""
Data loading utilities for instruction-tuning datasets
"""
import json
import jax.numpy as jnp
from transformers import AutoTokenizer


def load_jsonl_data(path, format_type="instruction"):
    """
    Load instruction-tuning data from JSONL file
    
    Args:
        path: Path to .jsonl file
        format_type: "instruction" for instruction/input/output format
    
    Returns:
        List of formatted prompts
    """
    print(f"📂 Loading data from {path}...")
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            if format_type == "instruction":
                # Format as instruction-following prompt
                prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Output:
{item['output']}"""
            else:
                # Just concatenate all fields
                prompt = " ".join(str(v) for v in item.values())
            
            data.append(prompt)
    
    print(f"   ✅ Loaded {len(data)} examples")
    return data


def tokenize_texts(texts, tokenizer_name='gpt2', seq_len=32768):
    """
    Tokenize text data
    
    Args:
        texts: List of strings
        tokenizer_name: HuggingFace model name (e.g., 'gpt2', 'google/gemma-2-9b')
        seq_len: Maximum sequence length
    
    Returns:
        Array of token IDs with shape (num_samples, seq_len)
    """
    print(f"\n🔤 Tokenizing with {tokenizer_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Handle tokenizers without pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = tokenizer.vocab_size
    print(f"   Vocab size: {vocab_size:,}")
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=seq_len,
        return_tensors='np'
    )
    
    token_ids = tokenized['input_ids']
    print(f"   Token IDs shape: {token_ids.shape}")
    
    return token_ids, vocab_size


def create_embeddings(token_ids, d_model, vocab_size, embedding_matrix=None):
    """
    Convert token IDs to embeddings
    
    Args:
        token_ids: Array of token IDs (batch, seq_len)
        d_model: Embedding dimension
        vocab_size: Vocabulary size
        embedding_matrix: Optional pretrained embeddings (vocab_size, d_model)
    
    Returns:
        Embeddings with shape (batch, seq_len, d_model)
    """
    print(f"\n🎨 Creating embeddings...")
    print(f"   Token IDs shape: {token_ids.shape}")
    print(f"   Embedding dim: {d_model}")
    
    if embedding_matrix is None:
        print("   ⚠️  Using RANDOM embeddings (for testing)")
        import jax
        key = jax.random.PRNGKey(42)
        embedding_matrix = jax.random.normal(
            key,
            (vocab_size, d_model),
            dtype=jnp.bfloat16
        ) * 0.02
    else:
        print("   ✅ Using provided embedding matrix")
    
    # Lookup embeddings
    embeddings = embedding_matrix[token_ids]
    print(f"   Embeddings shape: {embeddings.shape}")
    
    return embeddings


def chunk_sequences(data, chunk_size=8192):
    """
    Chunk sequences to avoid int32 overflow
    
    Args:
        data: Array with shape (batch, seq_len, d_model)
        chunk_size: Size of each chunk
    
    Returns:
        Chunked data with shape (num_chunks, batch, chunk_size, d_model)
    """
    batch, seq_len, d_model = data.shape
    num_chunks = seq_len // chunk_size
    
    print(f"\n✂️  Chunking sequences...")
    print(f"   {seq_len} tokens -> {num_chunks} chunks of {chunk_size}")
    
    # Reshape and transpose
    data_chunked = data.reshape(batch, num_chunks, chunk_size, d_model)
    data_chunked = jnp.transpose(data_chunked, (1, 0, 2, 3))
    
    print(f"   Output shape: {data_chunked.shape}")
    return data_chunked


def prepare_data(
    data_path,
    batch_size=8,
    seq_len=32768,
    d_model=3584,
    chunk_size=8192,
    tokenizer_name='gpt2'
):
    """
    Full data preparation pipeline
    
    Args:
        data_path: Path to .jsonl file
        batch_size: Number of samples per batch
        seq_len: Sequence length
        d_model: Model dimension
        chunk_size: Chunk size for chunking
        tokenizer_name: HuggingFace tokenizer name
    
    Returns:
        Chunked data ready for training (num_chunks, batch, chunk_size, d_model)
    """
    # Load text data
    texts = load_jsonl_data(data_path)
    
    # Handle batch size
    num_samples = len(texts)
    if num_samples < batch_size:
        print(f"\n⚠️  Only {num_samples} samples, repeating to fill batch...")
        repeats = (batch_size + num_samples - 1) // num_samples
        texts = (texts * repeats)[:batch_size]
    else:
        texts = texts[:batch_size]
    
    print(f"   Using {len(texts)} samples")
    
    # Tokenize
    token_ids, vocab_size = tokenize_texts(texts, tokenizer_name, seq_len)
    
    # Convert to embeddings
    embeddings = create_embeddings(token_ids, d_model, vocab_size)
    
    # Chunk
    data_chunked = chunk_sequences(embeddings, chunk_size)
    
    return data_chunked


if __name__ == "__main__":
    # Test the data loader
    print("🧪 Testing data loader...\n")
    
    data = prepare_data(
        data_path="data/sophia_dynamic_data.jsonl",
        batch_size=8,
        seq_len=32768,
        d_model=3584,
        chunk_size=8192,
        tokenizer_name='gpt2'
    )
    
    print("\n" + "="*60)
    print("✅ Data loading successful!")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print("="*60)
