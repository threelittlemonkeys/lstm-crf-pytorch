# Part-of-Speech (POS) Tagging

This is a simple tutorial for POS tagging using the scripts here.

1. Extract `brown.zip` to get `brown.tagged.merged.uniq.ptb`, a slightly refined version of [the Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus).

```
unzip brown.zip
```

2. You might want to train with your own tagset by running a script like `brown2ptb.py`, which converts the Brown Corpus tagset to a Penn Treebank (PTB) like tagset.

```
python3 brown2ptb.py
```

3. Split the data into training, test and validation sets and run `prepare.py`.

```
shuf brown.tagged.merged.uniq.ptb > data
head -1000 data > test
sed -n '1001,2000p' data > vaild
tail -n +2001 data > train 
python3 ../prepare.py 
```
