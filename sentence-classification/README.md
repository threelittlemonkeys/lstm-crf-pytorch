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

2. Run `tokenize.py` to tokenize documents into blocks of size L.

```
python3 ../tokenize.py L train
python3 ../tokenize.py L valid
```

3. Set the `hre` (hiearchical recurrent encoder) parameter in `parameters.py`. The dimension of the HRE must be the same as the dimension of the word embedding layer.

```
EMBED = {"char-cnn": 50, "lookup": 250, "hre": 300}
```

4. Run `prepare.py` to make CSV and index files.

```
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
