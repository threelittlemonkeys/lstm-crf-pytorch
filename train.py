import sys
import re
import time
from model import *
from utils import *
from os.path import isfile

def load_data():
    data = []
    batch_in = []
    batch_out = []
    batch_len = 0 # maximum sequence length of a mini-batch
    print("loading data...")
    fo = open(sys.argv[4], "r")
    for line in fo:
        line = line.strip()
        words = [int(i) for i in line.split(" ")]
        z = words.pop()
        if len(batch_in) == 0: # the first line has the maximum sequence length
            batch_len = z
        pad = [0] * (batch_len - z)
        batch_in.append(words[:z] + pad)
        batch_out.append(words[z:] + pad)
        if len(batch_in) == BATCH_SIZE:
            data.append((Var(LongTensor(batch_in)), LongTensor(batch_out))) # append a mini-batch
            batch_in = []
            batch_out = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data

def train():
    print("cuda: %s" % CUDA)
    num_epochs = int(sys.argv[5])
    data = load_data()
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    model = lstm_crf(len(word_to_idx), tag_to_idx)
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    if CUDA:
        model = model.cuda()
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print(model)
    print("training model...")
    for i in range(epoch + 1, epoch + num_epochs + 1):
        avrg_loss = 0
        for j, (x, y) in enumerate(data):
            model.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optim.step()
            loss = scalar(loss)
            avrg_loss += loss
            print("epoch = %d, iteration = %d, loss = %f" % (i, j + 1, loss))
        avrg_loss /= len(data)
        save_checkpoint(filename, model, i, avrg_loss)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model word_to_idx tag_to_idx training_data num_epoch" % sys.argv[0])
    train()
