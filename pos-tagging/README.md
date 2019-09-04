# Part-of-Speech (POS) Tagging

## Data Preparation

1. Extract `brown.zip` to get `brown.tagged.merged.uniq.ptb`, a slightly refined version of [the Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus).

```
unzip brown.zip
```

2. You might want to train with your own tagset by running a script like `brown2ptb.py`, which converts the Brown Corpus tagset to a Penn Treebank (PTB) like tagset.

```
python3 brown2ptb.py
```
