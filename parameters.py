import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNIT = "word" # unit of tokenization (char, char+space, word, sent)
TASK = None # task (None, pos-tagging, word-segmentation, sentence-segmentation)
RNN_TYPE = "GRU" # LSTM or GRU
NUM_DIRS = 2 # unidirectional: 1, bidirectional: 2
NUM_LAYERS = 2
BATCH_SIZE = 64
HRE = (UNIT == "sent") # hierarchical recurrent encoding
EMBED = {"lookup": 300} # embeddings (char-cnn, char-rnn, lookup, sae)
EMBED_SIZE = sum(EMBED.values())
HIDDEN_SIZE = 1000
DROPOUT = 0.5
LEARNING_RATE = 2e-4
EVAL_EVERY = 10
SAVE_EVERY = 10

PAD, PAD_IDX = "<PAD>", 0 # padding
SOS, SOS_IDX = "<SOS>", 1 # start of sequence
EOS, EOS_IDX = "<EOS>", 2 # end of sequence
UNK, UNK_IDX = "<UNK>", 3 # unknown token

CUDA = torch.cuda.is_available()
torch.manual_seed(0) # for reproducibility
# torch.cuda.set_device(0)

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
randn = lambda *x: torch.randn(*x).cuda() if CUDA else torch.randn
zeros = lambda *x: torch.zeros(*x).cuda() if CUDA else torch.zeros

KEEP_IDX = False # use the existing indices when adding more training data
NUM_DIGITS = 4 # number of decimal places to print
