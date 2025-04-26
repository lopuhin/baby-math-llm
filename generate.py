""" Based on generate.py from https://github.com/N8python/mlx-pretrain, © N8python
"""
import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from generate_lite import generate_lite, beam_search
from train import Trainer


mx.set_default_device(mx.gpu)
def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained model')
    parser.add_argument('--run', type=str, required=True,
                       help='Name of the training run to use')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt to start generation')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum number of tokens to generate')
    args = parser.parse_args()

    # Load run configuration and initialize trainer
    config_path = Path('runs') / args.run / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Config not found for run: {args.run}")
    
    trainer = Trainer(str(config_path), for_training=False)
    
    # Load the final checkpoint
    checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final_model.safetensors'
    if not checkpoint_path.exists():
        checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final.safetensors'
        if not checkpoint_path.exists():
            raise ValueError(f"Final checkpoint not found for run: {args.run}")
    checkpoint_path = str(checkpoint_path)
    
    trainer.model.load_weights(checkpoint_path)
    
    # Prepare input
    tokens = [trainer.tokenizer.BOS_TOKEN] + trainer.tokenizer.tokenize(args.prompt) + [trainer.tokenizer.SEP_TOKEN]
    
    # Generate
    greedy_output, greedy_score = generate_lite(
            trainer.model,
            mx.array(tokens),
            max_tokens=args.max_tokens,
            verbose=False,
            stop_tokens=[trainer.tokenizer.EOS_TOKEN],
    )
    print(f"Greedy Output: {trainer.tokenizer.detokenize(greedy_output)}")


if __name__ == "__main__":
    main()
