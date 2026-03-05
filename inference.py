import argparse
from dataclasses import asdict

import os
import sys

import torch
from torch.nn import functional as F

from config import Hyperparameters
from data import build_tokenizer
from model import Model

if os.name == 'nt':
  sys.stdout.reconfigure(encoding='utf-8')


def resolve_device(configured_device):
    if configured_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return configured_device


def apply_model_overrides(params, args):
    if args.context_size is not None:
        params.context_size = args.context_size
    if args.vocab_size is not None:
        params.vocab_size = args.vocab_size
    if args.num_transformers is not None:
        params.num_transformers = args.num_transformers
    if args.num_heads is not None:
        params.num_heads = args.num_heads
    if args.num_embd is not None:
        params.num_embd = args.num_embd
    if args.num_kv_groups is not None:
        params.num_kv_groups = args.num_kv_groups


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, context_size, temperature, top_k):
    for _ in range(max_new_tokens):
        x = input_ids[:, -context_size:]
        logits, _ = model(x)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                cutoff = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < cutoff, torch.full_like(logits, float("-inf")), logits
                )
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained LLobo model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--prompt", default="commercial", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set for deterministic sampling; omit for random output each run.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Optional overrides in case checkpoint was trained with non-default architecture.
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--num-transformers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-embd", type=int, default=None)
    parser.add_argument("--num-kv-groups", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    params = Hyperparameters()
    params.device = resolve_device(args.device)
    apply_model_overrides(params, args)

    enc, allowed = build_tokenizer()
    model = Model(params).to(params.device)

    checkpoint = torch.load(args.checkpoint, map_location=params.device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        ckpt_cfg = checkpoint["config"]
        print(f"Checkpoint config found: {ckpt_cfg}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_tokens = enc.encode(args.prompt, allowed_special=allowed)
    input_ids = torch.tensor(input_tokens, dtype=torch.long, device=params.device).unsqueeze(0)

    out_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        context_size=params.context_size,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    output_text = enc.decode(out_ids[0].tolist())
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Model Output ===")
    print(output_text)
    print("\n=== Runtime Config ===")
    print(asdict(params))


if __name__ == "__main__":
    main()


#py -3.10 inference.py --checkpoint models\model8b_i236000_l3.5015.pth --prompt "Hey!" --max-new-tokens 512 --temperature 0.8
#python inference.py --checkpoint models\your_checkpoint.pth --num-embd 1600 --num-heads 4 --num-transformers 12 --context-size 1024
