# Baby Math LLM

This is a work in progress on teaching a tiny LLM some basic math.

Pre-training code is taken from MLX-Pretain, see https://github.com/N8python/mlx-pretrain, © N8python

The idea is inspired by "Teaching Arithmetic to Small Transformers", see https://arxiv.org/pdf/2307.03381

Install (tested on Python 3.12):

    pip install -r requirements.txt

Prepare datasets:

    python baby_math/make_datasets.py baby_math

Train an LLM:

    python train.py --config baby_math/model-config.yaml

Generate:

    export MODEL=Baby-Math-v2-S-ep4
    python generate.py --run $MODEL --prompt 'Compute: -56 + 13'

You view the loss curve by running:

    python plot-logs.py "$MODEL"
