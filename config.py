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
    min_learn_rate: float = 2e-5
    epochs: int = 1
    warmup_steps: int = 200
    scheduler_type: str = "none"  # options: "none", "cosine"
    device: str = "cuda"
    grad_cp: bool = True
    log_every: int = 10
    save_every: int = 4000
    data_path: str = "data/data.txt"
    tokens_dir: str = "data/shards"
    tokens_dtype: str = "uint16"  # "uint16" or "uint32"
    use_pretokenized: bool = True
    checkpoint_dir: str = "models"
    resume_checkpoint: str = "" #Model path
