import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "sent" # unit of tokenization (char, word, sent)
FORMAT = None # format (None, word-segmentation, sentence-segmentation)
RNN_TYPE = "LSTM" # LSTM or GRU
NUM_DIRS = 2 # unidirectional: 1, bidirectional: 2
NUM_LAYERS = 2
BATCH_SIZE = 64
HRE = (UNIT == "sent") # hierarchical recurrent encoding
EMBED = {"lookup": 300} # embeddings (char-cnn, char-rnn, lookup, sae)
EMBED_SIZE = sum(EMBED.values())
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

torch.manual_seed(0)
CUDA = torch.cuda.is_available()

KEEP_IDX = False # use the existing indices when preparing additional data
NUM_DIGITS = 4 # number of digits to print
