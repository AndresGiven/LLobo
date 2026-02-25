from dataclasses import dataclass


@dataclass
class Hyperparameters:
    batch_size: int = 8
    context_size: int = 1024
    vocab_size: int = 50260
    num_transformers: int = 12
    num_heads: int = 4
    num_embd: int = 1600
    num_kv_groups: int = num_heads
    dropout: float = 0.0
    learn_rate: float = 2e-4
    epochs: int = 3
    device: str = "cuda"
    grad_cp: bool = True
    log_every: int = 1
    save_every: int = 1000
    data_path: str = "data/data.txt"
    checkpoint_dir: str = "models"
