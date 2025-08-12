# tokenizer.py
# Byte-level tokenizer, no external dependency.
import numpy as np

class ByteTokenizer:
    def __init__(self):
        # 0..255 bytes -> tokens 0..255
        # reserve token 256 as <eos>
        self.vocab_size = 257
        self.eos_token = 256

    def encode(self, text: str) -> np.ndarray:
        # encode string into bytes, then to int tokens
        b = text.encode('utf-8', errors='ignore')
        arr = np.frombuffer(b, dtype=np.uint8).astype(np.int64)
        return arr

    def decode(self, tokens) -> str:
        # tokens: iterable of ints <256 ideally
        toks = [t if t < 256 else 0 for t in tokens]
        b = bytes(toks)
        try:
            return b.decode('utf-8', errors='replace')
        except Exception:
            return b.decode('latin1', errors='replace')
