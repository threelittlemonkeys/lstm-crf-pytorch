# LSTM-CRF in PyTorch

A minimal PyTorch (1.7.1) implementation of bidirectional LSTM-CRF for sequence labelling.

Supported features:
- Mini-batch training with CUDA
- Lookup, CNNs, RNNs and/or self-attention in the embedding layer
- Hierarchical recurrent encoding (HRE)
- A PyTorch implementation of conditional random field (CRF)
- Vectorized computation of CRF loss
- Vectorized Viterbi decoding

## Usage

Training data should be formatted as below:
```
token/tag token/tag token/tag ...
token/tag token/tag token/tag ...
...
```
For more detail, see README.md in each subdirectory.

To prepare data:
```
python3 prepare.py training_data
```

To train:
```
python3 train.py model char_to_idx word_to_idx tag_to_idx training_data.csv (validation_data) num_epoch
```

To predict:
```
python3 predict.py model.epochN word_to_idx tag_to_idx test_data
```

To evaluate:
```
python3 evaluate.py model.epochN word_to_idx tag_to_idx test_data
```

## References

Zhiheng Huang, Wei Xu, Kai Yu. 2015. [Bidirectional LSTM-CRF Models for Sequence Tagging.](https://arxiv.org/abs/1508.01991) arXiv:1508.01991.

Harshit Kumar, Arvind Agarwal, Riddhiman Dasgupta, Sachindra Joshi. 2018. [Dialogue Act Sequence Labeling Using Hierarchical Encoder with CRF.](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16706/16724) In AAAI.

Xuezhe Ma, Eduard Hovy. 2016. [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF.](https://arxiv.org/abs/1603.01354) arXiv:1603.01354.

Shotaro Misawa, Motoki Taniguchi, Yasuhide Miura, Tomoko Ohkuma. 2017. [Character-based Bidirectional LSTM-CRF with Words and Characters for Japanese Named Entity Recognition.](http://www.aclweb.org/anthology/W17-4114) In Proceedings of the 1st Workshop on Subword and Character Level Models in NLP.

Yan Shao, Christian Hardmeier, Jörg Tiedemann, Joakim Nivre. 2017. [Character-based Joint Segmentation and POS Tagging for Chinese using Bidirectional RNN-CRF.](https://arxiv.org/abs/1704.01314) arXiv:1704.01314.

Slav Petrov, Dipanjan Das, Ryan McDonald. 2011. [A Universal Part-of-Speech Tagset.](https://arxiv.org/abs/1104.2086) arXiv:1104.2086.

Nils Reimers, Iryna Gurevych. 2017. [Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labeling Tasks.](https://arxiv.org/abs/1707.06799) arXiv:1707.06799.

Feifei Zhai, Saloni Potdar, Bing Xiang, Bowen Zhou. 2017. [Neural Models for Sequence Chunking.](https://arxiv.org/abs/1701.04027) In AAAI.

Zenan Zhai, Dat Quoc Nguyen, Karin Verspoor. 2018. [Comparing CNN and LSTM Character-level Embeddings in BiLSTM-CRF Models for Chemical and Disease Named Entity Recognition.](https://arxiv.org/abs/1808.08450) arXiv:1808.08450.
