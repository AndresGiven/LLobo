import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
from datasets import load_dataset

from data import build_tokenizer


def _select_dtype(vocab_size):
    return np.uint16 if vocab_size <= 65535 else np.uint32


def _write_shard(out_dir, shard_idx, buffer, dtype):
    shard_path = os.path.join(out_dir, f"tokens_{shard_idx:05d}.bin")
    np.array(buffer, dtype=dtype).tofile(shard_path)
    return shard_path, len(buffer)


def main():
    parser = argparse.ArgumentParser(
        description="Stream a HF dataset, pretokenize text, and write token shards."
    )
    parser.add_argument("--dataset", default="tiiuae/falcon-refinedweb")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="content")
    parser.add_argument("--out-dir", default="data/shards")
    parser.add_argument("--vocab-size", type=int, default=50260)
    parser.add_argument("--shard-size", type=int, default=800_000_000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--filter-field", default=None)
    parser.add_argument("--filter-regex", default=None)
    parser.add_argument("--skip-empty", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    enc, allowed = build_tokenizer()
    dtype = _select_dtype(args.vocab_size)
    regex = re.compile(args.filter_regex) if args.filter_regex else None

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    shard_idx = 0
    buffer = []
    total_tokens = 0
    total_samples = 0
    shards = []

    for sample in ds:
        if args.max_samples is not None and total_samples >= args.max_samples:
            break
        if args.max_tokens is not None and total_tokens >= args.max_tokens:
            break

        if args.filter_field and regex:
            field_val = sample.get(args.filter_field)
            if field_val is None or not regex.search(str(field_val)):
                continue

        text = sample.get(args.text_field, "")
        if args.skip_empty and not text:
            continue

        tokens = enc.encode(text, allowed_special=allowed)
        if not tokens:
            continue

        buffer.extend(tokens)
        total_tokens += len(tokens)
        total_samples += 1

        if len(buffer) >= args.shard_size:
            shard_path, count = _write_shard(args.out_dir, shard_idx, buffer, dtype)
            shards.append({"path": shard_path, "tokens": count})
            shard_idx += 1
            buffer = []

    if buffer:
        shard_path, count = _write_shard(args.out_dir, shard_idx, buffer, dtype)
        shards.append({"path": shard_path, "tokens": count})

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "text_field": args.text_field,
        "vocab_size": args.vocab_size,
        "dtype": str(dtype),
        "shard_size": args.shard_size,
        "total_tokens": total_tokens,
        "total_samples": total_samples,
        "shards": shards,
        "created_at": datetime.now(datetime.timezone.utc).isoformat() + "Z",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(shards)} shard(s) to {args.out_dir}")
    print(f"Total samples: {total_samples}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
