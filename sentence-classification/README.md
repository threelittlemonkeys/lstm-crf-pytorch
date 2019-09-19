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

2. Run `block-tokenize.py` to tokenize documents into blocks of size L.

```
python3 ../block-tokenize.py L train.raw
python3 ../block-tokenize.py L valid.raw
```

3. Set `UNIT` to `sent` in `parameters.py` for hierarchical encoding.

```
UNIT = "sent"
```

4. Run `prepare.py` to make CSV and index files.

```
mv train.raw.block train
mv valid.raw.block valid
python3 ../prepare.py train
python3 ../prepare.py valid
```

5. Train your model. You can modify the hyperparameters in `parameters.py`.

```
python3 ../train.py model train.char_to_idx train.word_to_idx train.tag_to_idx train.csv valid N
```

6. Predict and evaluate your model.

```
python3 ../predict.py model.epochN train.char_to_idx train.word_to_idx train.tag_to_idx test
python3 ../evaluate.py model.epochN train.char_to_idx train.word_to_idx train.tag_to_idx test
```
