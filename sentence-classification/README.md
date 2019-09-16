# Sentence Labelling

This is a simple tutorial for sentence labelling using the scripts here.

1. Training and validation data should be formatted as below, sentence-label pairs separated by `\n` and blocks of sentence-label pairs by `\n\n`:

```
sentence \t label
sentence \t label

sentence \t label
sentence \t label
sentence \t label

...
```

2. Run `prepare.py` to make CSV and index files.

```
python3 ../prepare.py train
```

4. Train your model. You can modify the hyperparameters in `parameters.py`.

```
python3 ../train.py model train.char_to_idx train.word_to_idx train.tag_to_idx train.csv valid 100
```

5. Predict and evaluate your model.

```
python3 ../predict.py model.epoch100 train.char_to_idx train.word_to_idx train.tag_to_idx test
python3 ../evaluate.py model.epoch100 train.char_to_idx train.word_to_idx train.tag_to_idx test
```
