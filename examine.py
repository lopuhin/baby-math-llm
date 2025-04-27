#!/usr/bin/env python3
"""
Inspection utility for MLX-Pretrain data files.

Usage:
  python examine.py --count-tokens [data_path] [tokenizer_path]
  
Examples:
  python examine.py --count-tokens train.jsonl tokenizer/tokenizer.json
  python examine.py --count-tokens val.jsonl tokenizer
"""

import argparse
import json
import os
from typing import List, Dict, Any
from tokenizers import Tokenizer
from tqdm import tqdm

def load_jsonl(data_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    line_count = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=line_count, desc="Loading data"):
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def count_tokens(data_path: str, tokenizer_path: str):
    """Count the total number of tokens in a JSONL file."""
    print(f"Loading data from {data_path}")
    data = load_jsonl(data_path)
    
    print(f"Loading tokenizer from {tokenizer_path}")
    if os.path.isdir(tokenizer_path):
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
    else:
        tokenizer_file = tokenizer_path
    tokenizer = Tokenizer.from_file(tokenizer_file)
    
    total_tokens = 0
    for item in tqdm(data, desc="Counting tokens"):
        if "text" in item:
            encoding = tokenizer.encode(item["text"])
            total_tokens += len(encoding.ids)
    
    print(f"Total tokens in {data_path}: {total_tokens}")
    return total_tokens

def main():
    parser = argparse.ArgumentParser(description="Inspection utilities for MLX-Pretrain data")
    
    # This approach allows for adding other inspection commands in the future
    parser.add_argument("--count-tokens", action="store_true", 
                        help="Count tokens in a JSONL file")
    parser.add_argument("data_path", nargs="?", help="Path to the JSONL data file")
    parser.add_argument("tokenizer_path", nargs="?", help="Path to the tokenizer file")
    
    args = parser.parse_args()
    
    if args.count_tokens:
        if not args.data_path or not args.tokenizer_path:
            parser.error("--count-tokens requires data_path and tokenizer_path")
        count_tokens(args.data_path, args.tokenizer_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()