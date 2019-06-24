import torch
import torch.nn as nn

UNIT = "word" # unit of tokenization (char, word)
FORMAT = None # format (None, word-segmentation, sentence-segmentation)
CASING = "uncased" # casing for normalization (cased, uncased, ul+uncased)
RNN_TYPE = "LSTM" # LSTM or GRU
NUM_DIRS = 2 # unidirectional: 1, bidirectional: 2
NUM_LAYERS = 2
BATCH_SIZE = 128
EMBED = ["char-cnn", "lookup"] # embeddings (char-cnn, lookup, sae)
EMBED_SIZE = 300
HIDDEN_SIZE = 1000
DROPOUT = 0.5
LEARNING_RATE = 1e-4
EVAL_EVERY = 10
SAVE_EVERY = 10

PAD = "<PAD>" # padding
SOS = "<SOS>" # start of sequence
EOS = "<EOS>" # end of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

KEEP_IDX = False # use the existing indices when preparing additional data
NUM_DIGITS = 4 # number of digits to print
