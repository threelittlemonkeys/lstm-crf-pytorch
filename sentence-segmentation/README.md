# Sentence Segmentation

This is a simple tutorial for sentence segmentation using the scripts here.

1. Training and validation data should be formatted as below, sentences separated by `\n` and sentence blocks by `\n\n`:

```
sentence
sentence

sentence
sentence
sentence

...
```

2. Run `word+iob.py` to tag words in IOB2 format.

```
python3 word+iob.py sample.txt
```

3. Modify the following settings in `parameters.py` for a sentence segmentation task.

```
UNIT = "word"
TASK = "sentence-segmentation"
```


4. Split the data into training, test and validation sets and run `prepare.py` to make CSV and index files.

```
shuf sample.txt.word+iob > data
head -1000 data > test
sed -n '1001,2000p' data > valid
tail +2001 data > train
python3 prepare.py train
```

5. Train your model. You can also modify the hyperparameters in `parameters.py`.

```
python3 train.py model train.char_to_idx train.word_to_idx train.tag_to_idx train.csv valid 100
```

6. Predict and evaluate your model.

```
python3 predict.py model.epoch100 train.char_to_idx train.word_to_idx train.tag_to_idx test
python3 evaluate.py model.epoch100 train.char_to_idx train.word_to_idx train.tag_to_idx test
```
