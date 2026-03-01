import glob
import os

import numpy as np
import torch
import tiktoken
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def build_tokenizer():
    gpt2_base = tiktoken.get_encoding("gpt2")
    enc = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=gpt2_base._pat_str,
        mergeable_ranks=gpt2_base._mergeable_ranks,
        special_tokens={
            **gpt2_base._special_tokens,
            "<USER>": 50257,
            "<BOT>": 50258,
            "<SYSTEM>": 50259,
        },
    )
    allowed = {"<USER>", "<BOT>", "<SYSTEM>"}
    return enc, allowed


def load_training_tokens(data_path, device):
    enc, allowed = build_tokenizer()
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    print("File read")
    training_data = enc.encode(text, allowed_special=allowed)
    training_data = torch.tensor(training_data, dtype=torch.long, device=device)
    print("tokenized sequence")
    print(f"{len(training_data)} tokens")
    return training_data


class TokenDataset(Dataset):
    def __init__(self, data, context_size):
        self.data = data
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - self.context_size - 1

    def __getitem__(self, index):
        return (
            self.data[index : index + self.context_size],
            self.data[index + 1 : index + self.context_size + 1],
        )


def create_dataloader(tokens, batch_size, context_size):
    dataset = TokenDataset(tokens, context_size=context_size)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def _dtype_from_string(dtype_str):
    if dtype_str == "uint16":
        return np.uint16
    if dtype_str == "uint32":
        return np.uint32
    raise ValueError(f"Unsupported tokens_dtype: {dtype_str}")


class ShardedTokenDataset(Dataset):
    def __init__(self, shard_paths, context_size, dtype_str):
        if not shard_paths:
            raise ValueError("No shard paths provided.")
        self.context_size = context_size
        self.dtype = _dtype_from_string(dtype_str)
        self.shards = [np.memmap(p, dtype=self.dtype, mode="r") for p in shard_paths]
        self.usable = [
            max(0, len(s) - self.context_size - 1) for s in self.shards
        ]
        self.cum = np.cumsum(self.usable)
        if len(self.cum) == 0 or self.cum[-1] <= 0:
            raise ValueError("Shards are too small for the configured context_size.")

    def __len__(self):
        return int(self.cum[-1])

    def _locate(self, idx):
        shard_idx = int(np.searchsorted(self.cum, idx, side="right"))
        prev = self.cum[shard_idx - 1] if shard_idx > 0 else 0
        local = idx - prev
        return shard_idx, int(local)

    def __getitem__(self, index):
        shard_idx, local = self._locate(index)
        shard = self.shards[shard_idx]
        x = shard[local : local + self.context_size]
        y = shard[local + 1 : local + self.context_size + 1]
        return torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))


def load_sharded_tokens(tokens_dir, context_size, dtype_str):
    shard_paths = sorted(glob.glob(os.path.join(tokens_dir, "*.bin")))
    if not shard_paths:
        raise FileNotFoundError(f"No .bin shards found in {tokens_dir}")
    return ShardedTokenDataset(shard_paths, context_size=context_size, dtype_str=dtype_str)


def create_sharded_dataloader(tokens_dir, batch_size, context_size, dtype_str):
    dataset = load_sharded_tokens(tokens_dir, context_size=context_size, dtype_str=dtype_str)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
