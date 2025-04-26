import argparse
import json
import random
import math
from pathlib import Path


def addition_example(a, b):
    input = f'Compute: {a} + {b}'
    output = f'{a} + {b} = '
    if b < 0 and a != 0:
        output += subtraction_example(a, -b)['output']
    elif a < 0 and b != 0:
        output += subtraction_example(b, -a)['output']
    else:
       #if a >= 10 and b >= 10:
       #    output += f'{a // 10 * 10} + {a % 10} + {b // 10 * 10} + {b % 10} = '
       #    output += f'{a // 10 * 10} + {b // 10 * 10} + {a % 10} + {b % 10} = '
        output += f'{a + b}'
    return {
        'input': input,
        'output': output,
        'key': f'addsub-{abs(a)}-{abs(b)}',
    }


def subtraction_example(a, b):
    input = f'Compute: {a} - {b}'
    output = f'{a} - {b} = '
    if b < 0 and a != 0:
        output += addition_example(a, -b)['output']
    else:
        if a < 0 and b != 0:
            output += f'-({-a} + {b}) = '
        elif a < b and a != 0 and a != b:
            output += f'-({b} - {a}) = '
        output += f'{a - b}'
    return {
        'input': input,
        'output': output,
        'key': f'addsub-{abs(a)}-{abs(b)}',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_examples = []
    valid_examples = []
    examples = []
    # TODO try stricter separation of training and validation data
    #for a in range(-99, 100):
    for a in range(100):
        #for b in range(-99, 100):
        for b in range(100):
            if abs(a) < 10 and abs(b) < 10:
                lst = train_examples
            else:
                lst = examples
            lst.extend([
                addition_example(a, b),
                # subtraction_example(a, b),
            ])
    
    keys = sorted({x['key'] for x in examples})
    random.Random(args.seed).shuffle(keys)
    keys = keys[:3000]
    n_train = int(math.ceil(len(keys) * 0.8))

    train_keys = set(keys[:n_train])
    valid_keys = set(keys[n_train:])

    train_examples.extend(x for x in examples if x['key'] in train_keys)
    valid_examples.extend(x for x in examples if x['key'] in valid_keys)

    for split_examples, name in [
            (train_examples, 'train'),
            (valid_examples, 'val'),
        ]:
        path = args.output / f'{name}.jsonl'
        with path.open('wt') as f:
            for x in split_examples:
                f.write(json.dumps(x, ensure_ascii=False))
                f.write('\n')
        print(f'written {len(split_examples):,} to {path}')


if __name__ == '__main__':
    main()
