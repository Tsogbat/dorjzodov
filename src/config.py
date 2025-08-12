# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Model sizes (adjust to target param count)
    vocab_size: int = 257   # 256 byte values + 1 for <eos>
    d_model: int = 1024     # hidden size
    n_heads: int = 16
    n_layers: int = 24
    d_ff: int = 4096
    max_seq_len: int = 512

    # Training
    lr: float = 3e-4
    batch_size: int = 2          # effective batch = batch_size * grad_accum
    grad_accum_steps: int = 8
    seq_len: int = 512
    num_steps: int = 20000
    eval_interval: int = 500
    save_interval: int = 2000

    # Device / precision
    device: str = None  # auto detect in utils
    use_amp: bool = False  # MPS/AMP caution. Default False.

    # Checkpoints
    ckpt_dir: str = "checkpoints"
    output_dir: str = "outputs"

cfg = Config()
