# Literature Style Transfer

## What

What will a sentence look like if it is written by different writers? Interesting? This repo is aiming to achieve style transfer across different writers (currently only chinese writers because I am more familiar with chinese literature).

## How

Deep learning: Bert/Transformer, PyTorch, AllenNLP. Methods similar to StarGAN, CycleGAN, etc..

## Now

Currently I've already finished the code of style transfering between two authors following StyleTransformer (an architecture similar to CycleGAN in CV). Besides, I use [CPT](https://github.com/fastnlp/CPT) as the pretrained backbone.

## Difficulties
During training, to keep the model differentiable, I adopt a token-by-token generation process, which is not time-efficient. On a single A100, one epoch costs 20 hours.
