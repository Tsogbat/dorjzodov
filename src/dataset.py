# dataset.py
import numpy as np
import os
from tqdm import tqdm
from tokenizer import ByteTokenizer

def preprocess_to_tokens(raw_path: str, out_path: str, chunk_size=10_000_000):
    """
    Read raw text, encode to bytes, save as single numpy array of token ids.
    This is memory efficient for up to ~1GB corpus on disk.
    """
    tk = ByteTokenizer()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    parts = []
    total = 0
    with open(raw_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            # chunk is bytes; decode/encode not needed -> interpret bytes
            arr = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64)
            parts.append(arr)
            total += arr.size
    if total == 0:
        raise ValueError("Empty input corpus.")
    data = np.concatenate(parts)
    # append eos tokens between files or at end
    eos = np.array([tk.eos_token], dtype=np.int64)
    data = np.concatenate([data, eos])
    np.save(out_path, data)
    print(f"Saved token array to {out_path}, length {data.size}")
    return out_path

import torch
from torch.utils.data import Dataset

class LMByteDataset(Dataset):
    """
    Provide sequences of length seq_len from the flat token array.
    For efficiency we return contiguous non-overlapping blocks. For heavy augmentation
    configure to sample random offsets instead.
    """
    def __init__(self, tokens_path: str, seq_len: int):
        self.tokens = np.load(tokens_path)
        self.seq_len = seq_len
        # number of full sequences
        self.n = len(self.tokens) // seq_len
        if self.n == 0:
            raise ValueError("Token array too small for seq_len. Reduce seq_len or increase data.")
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        start = idx * self.seq_len
        seq = self.tokens[start:start + self.seq_len]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y
