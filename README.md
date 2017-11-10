# LSTM-CRF in PyTorch

A PyTorch implementation of bidirectional LSTM-CRF for sequence tagging, adapted from [the PyTorch tutorial](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html).

## Usage

To prepare data:
```
python prepare.py training_data
```

To train:
```
python train.py model word_to_idx tag_to_idx training_data.csv num_epoch
```

To predict:
```
python predict.py model word_to_idx tag_to_idx test_data
```
