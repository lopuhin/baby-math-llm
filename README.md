# Baby Math LLM

This is a work in progress on teaching a tiny LLM some basic math.

Pre-training code is taken from MLX-Pretain, see https://github.com/N8python/mlx-pretrain, © N8python

The idea is inspired by "Teaching Arithmetic to Small Transformers", see https://arxiv.org/pdf/2307.03381

This needs Apple Silicon to run as it is based on MLX. Install (tested on Python 3.12, should work on 3.10+):

    pip install -r requirements.txt

Prepare datasets:

    python baby_math/make_datasets.py

Train an LLM:

    python train.py --config baby_math/model-config-s.yaml

Generate:

    export MODEL=Baby-Math-S
    python generate.py --run $MODEL --prompt 'Compute: 56 + 13'

You view the loss curve by running:

    python plot-logs.py "$MODEL"
