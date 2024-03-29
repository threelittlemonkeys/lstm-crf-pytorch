# Word classification

This is a simple tutorial for sequence labelling tasks such as part-of-speech (POS) tagging and named entity recognition (NER).

1. Modify the following settings in `parameters.py`.

```
UNIT = "word"
TASK = "word-classification"
```

2. Split the data into training, test and validation sets and run `prepare.py` to make CSV and index files.

```
head -1000 data > test
sed -n '1001,2000p' data > valid
tail +2001 data > train
python3 prepare.py train
```

3. Train your model. You can also modify the hyperparameters in `parameters.py`.

```
python3 train.py model train.char_to_idx train.word_to_idx train.tag_to_idx train.csv valid 100
```

4. Predict and evaluate your model.

```
python3 predict.py model.epoch100 train.char_to_idx train.word_to_idx train.tag_to_idx test
python3 evaluate.py model.epoch100 train.char_to_idx train.word_to_idx train.tag_to_idx test
```
