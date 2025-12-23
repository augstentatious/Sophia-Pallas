import torch
from torch.utils.data import IterableDataset
import os

class StreamingContextDataLoader(IterableDataset):
    """
    Project Infinite Text: Streaming Loader for Massive Context Windows.
    Feeds UnSwag with continuous token streams without OOMing system RAM.
    """
    
    def __init__(self, file_path, tokenizer, block_size=16384, overlap=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.overlap = overlap
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Clean Room Error: {file_path} not found.")

    def __iter__(self):
        buffer = ""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                buffer += line
                
                # Heuristic: 4 chars ~= 1 token
                if len(buffer) > self.block_size * 4:
                    tokenized = self.tokenizer(
                        buffer, add_special_tokens=False, return_tensors="pt"
                    ).input_ids[0]
                    
                    while len(tokenized) >= self.block_size:
                        chunk = tokenized[:self.block_size]
                        yield {
                            "input_ids": chunk,
                            "labels": chunk.clone()
                        }
                        
                        # Slide window with overlap
                        tokenized = tokenized[self.block_size - self.overlap:]
                    
                    if len(tokenized) > 0:
                        buffer = self.tokenizer.decode(tokenized)
                    else:
                        buffer = ""
