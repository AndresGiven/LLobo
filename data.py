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
