Dispel
======

Obtain word embeddings from transformer models

## Installation

Simply install with pip (preferably in a virtual env, you know the drill)

```console
pip install git+https://github.com/LoicGrobol/dispel.git
```

## Vectorize a CoNLL file

Here is a short example:

```console
dispel --model roberta-base my_raw_corpus.conll word_embeddings.tab
```

There are other parameters (see `dispel --help` for a comprehensive list).

The `--model` parameter is fed directly to [Huggingface's
`transformers`](https://huggingface.co/transformers) and can be a [pretrained
model](https://huggingface.co/transformers/pretrained_models.html) name or local path.
