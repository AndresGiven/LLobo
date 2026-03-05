import argparse
import json
import os
import re
from datetime import datetime, timezone
from array import array
import time
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from data import build_tokenizer

#py -3.10 pretojenize_hf.py --filter language_score >= 0.85

def _select_dtype(vocab_size):
    return "H" if vocab_size <= 65535 else "I"

def _write_shard(out_dir, shard_idx, arr):
    shard_path = os.path.join(out_dir, f"tokens_{shard_idx:05d}.bin")
    with open(shard_path, "wb") as f:
       arr.tofile(f)
    return shard_path, len(arr)

def _parse_filters(filter_args):
    parsed = []
    for field, op, value in filter_args:
        try:
          if "." in value:
            value = float(value)
          else:
            value = int(value)
        except ValueError:
          pass
        parsed.append((field, op, value))
    return parsed

def main():
    # timers = defaultdict(float)   #TIMER
    # log_every = 1000              #TIMER
    # t_start = time.perf_counter() #TIMER

    parser = argparse.ArgumentParser(
        description="Stream HF Dataset, pretokenize, write to shard(s)."
    )
    parser.add_argument("--dataset", default="HuggingFaceFW/fineWeb")
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--column", default="text")
    parser.add_argument("--out-dir", default="data/shards")
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=500_000_000)
    parser.add_argument(
       "--filter",
        nargs=3,
        action="append",
    )
    args = parser.parse_args()

    filters = _parse_filters(args.filter) if args.filter else None

    os.makedirs(args.out_dir, exist_ok=True)
    enc, allowed = build_tokenizer()

    ds = load_dataset(
    		args.dataset,
        args.subset,
        split=args.split, 
        streaming=True,
        filters=filters,
        columns=[args.column]   
    )
    
    dtype = _select_dtype(args.vocab_size)
    shard_idx = 0
    buffer = array(dtype)
    total_tokens = 0
    total_samples = 0
    shards = []

    for sample in ds:
      # t0 = time.perf_counter()  #TIMER
      
      #-----Limit Check----
      # t = time.perf_counter()   #TIMER
      if args.max_samples is not None and total_samples >= args.max_samples:
        break
      if args.max_tokens is not None and total_tokens >= args.max_tokens:
        break
      # timers["Break_Check"] += time.perf_counter() - t    #TIMER

      #-----String extraction----
      # t = time.perf_counter()   #TIMER
      text = sample.get(args.column, "")
      if not text.strip():
        # timers["text_extract"] += time.perf_counter() - t #TIMER
        continue
      # timers["text_extract"] += time.perf_counter() - t   #TIMER


      #-----Tokenization-----
      # t = time.perf_counter()   #TIMER
      tokens = enc.encode_ordinary(text)
      # timers["tokenization"] += time.perf_counter() - t   #TIMER

      #-----Buffer Op-----
      # t = time.perf_counter()   #TIMER
      buffer.extend(tokens)
      # timers["buffer_extend"] += time.perf_counter() - t  #TIMER

      total_tokens += len(tokens)
      total_samples += 1

      #-----Flush-----
      # t = time.perf_counter()   #TIMER
      while len(buffer) >= args.shard_size:
        shard_buffer = buffer[:args.shard_size]
        shard_path, count = _write_shard(args.out_dir, shard_idx, shard_buffer)
        shards.append({"path": shard_path, "tokens": count})
        shard_idx += 1
        del buffer[:args.shard_size]
      # timers["flush"] += time.perf_counter() - t            #TIMER

      #-----Logging-----
      # if total_samples % log_every == 0:
      #   elapsed = time.perf_counter() - t_start
      #   tok_per_s = total_tokens / elapsed if elapsed > 0 else 0.0

      #   timed_total = sum(timers.values())
      #   print(
      #       f"[{total_samples:,} samples | {total_tokens:,} tokens | "
      #       f"{tok_per_s:,.0f} tok/s] | "
      #       f"Limit Check={timers['Break_Check']:.1f} | "
      #       f"Text Extract={timers['text_extract']:.1f} | "
      #       f"Tokenize={timers['tokenization']:.1f} | "
      #       f"Buffer Extend={timers['buffer_extend']:.1f} | "
      #       f"Flush={timers['flush']:.1f} | "
      #       f"(timed={timed_total:.1f}, wall={elapsed:.1f})"
      #   )

    if buffer:
        shard_path, count = _write_shard(args.out_dir, shard_idx, buffer)
        shards.append({"path": shard_path, "tokens": count})
    
    # elapsed = time.perf_counter() - t_start
    # print("\n=== Timing summary ===")
    # for k, v in sorted(timers.items(), key=lambda kv: kv[1], reverse=True):
    #     print(f"{k:14s}: {v:10.3f} s  ({(v/elapsed*100 if elapsed else 0):5.1f}%)")
    # print(f"{'wall':14s}: {elapsed:10.3f} s")
    # print(f"tokens/sec: {total_tokens/elapsed:,.0f}")
    
    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "text_field": args.column,
        "vocab_size": args.vocab_size,
        "dtype": str(dtype),
        "shard_size": args.shard_size,
        "total_tokens": total_tokens,
        "total_samples": total_samples,
        "shards": shards,
        "created_at": datetime.now(timezone.utc).isoformat() + "Z",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()

           
