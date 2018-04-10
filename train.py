import sys
import re
import time
from model import *
from utils import *
from os.path import isfile

def load_data():
    data = []
    batch_x = []
    batch_y = []
    batch_len = 0 # maximum sequence length of a mini-batch
    print("loading data...")
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    fo = open(sys.argv[4], "r")
    for line in fo:
        line = line.strip()
        tokens = [int(i) for i in line.split(" ")]
        seq_len = len(tokens) // 2
        if len(batch_x) == 0: # the first line has the maximum sequence length
            batch_len = seq_len
        pad = [PAD_IDX] * (batch_len - seq_len)
        batch_x.append(tokens[:seq_len] + [EOS_IDX] + pad)
        batch_y.append(tokens[seq_len:] + [EOS_IDX] + pad)
        if len(batch_x) == BATCH_SIZE:
            data.append((Var(LongTensor(batch_x)), LongTensor(batch_y))) # append a mini-batch
            batch_x = []
            batch_y = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, word_to_idx, tag_to_idx

def train():
    num_epochs = int(sys.argv[5])
    data, word_to_idx, tag_to_idx = load_data()
    model = lstm_crf(len(word_to_idx), len(tag_to_idx))
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    if CUDA:
        model = model.cuda()
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(model)
    print("training model...")
    for i in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        t = time.time()
        for x, y in data:
            model.zero_grad()
            loss = torch.mean(model(x, y)) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss = scalar(loss)
            loss_sum += loss
        t = time.time() - t
        loss_sum /= len(data)
        if i % SAVE_EVERY and i != epoch + num_epochs:
            save_checkpoint("", "", i, loss_sum, t)
        else:
            save_checkpoint(filename, model, i, loss_sum, t)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model word_to_idx tag_to_idx training_data num_epoch" % sys.argv[0])
    print("cuda: %s" % CUDA)
    train()
