# Sentence Classification

This is a simple tutorial for sentence-level sequence labelling using the scripts here.

1. Training and validation data should be formatted as below, sentence-label pairs separated by `\n` and documents by `\n\n`:

```
sentence \t label
sentence \t label

sentence \t label
sentence \t label
sentence \t label

...
```

2. Optionally you can run `block-tokenize.py` to tokenize documents into blocks of sizes L1, L2, L3, ... . Please note that duplicate blocks will be removed by this script.

```
python3 block-tokenize.py L1,L2,L3 train.raw
python3 block-tokenize.py L1,L2,L3 valid.raw
```

3. Modify the following settings in `parameters.py` for hierarchical encoding.

```
UNIT = "sent"
TASK = None
```

4. Run `prepare.py` to make CSV and index files.

```
mv train.raw.block train
mv valid.raw.block valid
python3 prepare.py train
```

5. Train your model. You can also modify the hyperparameters in `parameters.py`.

```
python3 train.py model train.char_to_idx train.word_to_idx train.tag_to_idx train.csv valid N
```

6. Predict and evaluate your model.

```
python3 predict.py model.epochN train.char_to_idx train.word_to_idx train.tag_to_idx test
python3 evaluate.py model.epochN train.char_to_idx train.word_to_idx train.tag_to_idx test
```
