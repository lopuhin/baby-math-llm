import argparse
import json
import random
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    examples = []
    for i in range(1, 100):
        for j in range(1, 100):
            examples.append({
                'input': f'Compute: {i} + {j}',
                'output': str(i + j),
            })
    
    n_train = int(math.ceil(len(examples) * 0.8))
    random.Random(args.seed).shuffle(examples)
    train_examples = examples[:n_train]
    val_examples = examples[n_train:]

    for split_examples, name in [
            (train_examples, 'train'),
            (val_examples, 'val'),
        ]:
        path = args.output / f'{name}.jsonl'
        with path.open('wt') as f:
            for x in split_examples:
                f.write(json.dumps(x, ensure_ascii=False))
                f.write('\n')
        print(f'written {path}')


if __name__ == '__main__':
    main()
