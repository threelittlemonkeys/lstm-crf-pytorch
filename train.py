import sys
import os.path
import re
import time
from model import *
from utils import *

def load_data():
    data = []
    tag_to_idx = {}
    tag_to_idx[START_TAG] = len(tag_to_idx)
    tag_to_idx[STOP_TAG] = len(tag_to_idx)
    print("loading data...")
    fo = open(sys.argv[4], "r")
    for line in fo:
        line = re.sub("\s+", " ", line)
        line = re.sub("^ | $", "", line)
        if line == "":
            continue
        tokens = line.split(" ")
        sent = []
        tags = []
        for token in tokens:
            word = re.sub("/[A-Z]+", "", token)
            tag = re.sub(".+/", "", token)
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
            sent += list(word)
            tags += [tag] * len(word)
        data.append((sent, tags))
    fo.close()
    print("training data size = %d" % len(data))
    return data, tag_to_idx

def train():
    print("cuda: %s" % CUDA)
    print("batch size: %d" % BATCH_SIZE)
    num_epochs = int(sys.argv[5])
    data, tag_to_idx = load_data()
    update = 1 if os.path.isfile(sys.argv[1]) else 0
    if update:
        tag_to_idx = load_tag_to_idx(sys.argv[2])
        word_to_idx = load_word_to_idx(sys.argv[3])
    else:
        epoch = 0
        save_tag_to_idx(sys.argv[2], tag_to_idx)
        word_to_idx = save_word_to_idx(sys.argv[3], data)
    model = lstm_crf(len(word_to_idx), tag_to_idx)
    if CUDA:
        model = model.cuda()
    print(model)
    if update:
        epoch = load_checkpoint(sys.argv[1], model)
        sys.argv[1] = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for i in range(epoch + 1, epoch + num_epochs + 1):
        timestamp = time.time()
        for sent, tags in data:
            model.zero_grad()
            input = sent_to_idx(sent, word_to_idx)
            output = LongTensor([tag_to_idx[t] for t in tags])
            loss = model.neg_log_likelihood(input, output)
            loss.backward()
            model.optim.step()
        print("epoch = %d, nll = %f, training time = %d " % (i, scalar(loss), time.time() - timestamp))
        save_checkpoint(sys.argv[1], i, model)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model tag_to_idx word_to_idx training_data num_epoch" % sys.argv[0])
    train()
