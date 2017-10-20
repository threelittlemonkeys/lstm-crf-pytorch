import sys
import os.path
import re
import time
from model import *
from utils import *

def load_data():
    cnt = 0
    data = []
    print("loading data...")
    fo = open(sys.argv[4], "r")
    dim = int(fo.readline()) # dimension
    for line in fo:
        line = line.strip()
        tkns = [int(x) for x in line.split(" ")]
        data.append((tkns[:dim], tkns[dim:]))
        cnt += 1
    fo.close()
    print("data size: %d" % cnt)
    '''
    data = TensorDataset(Var(LongTensor(inputs)), LongTensor(outputs))
    data_loader = DataLoader(data, batch_size = BATCH_SIZE)
    inputs = Var(LongTensor(inputs))
    outputs = LongTensor(outputs)
    '''
    return data

def train():
    print("cuda: %s" % CUDA)
    print("batch size: %d" % BATCH_SIZE)
    num_epochs = int(sys.argv[5])
    data = load_data()
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    model = lstm_crf(len(word_to_idx), tag_to_idx)
    epoch = load_checkpoint(sys.argv[1], model) if os.path.isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    if CUDA:
        model = model.cuda()
    print(model)
    print("training model...")
    for i in range(epoch + 1, epoch + num_epochs + 1):
        timestamp = time.time()
        for sent, tags in data:
            model.zero_grad()
            input = Var(LongTensor(sent))
            output = LongTensor(tags)
            loss = model.neg_log_likelihood(input, output)
            loss.backward()
            model.optim.step()
        print("epoch = %d, nll = %f, training time = %d " % (i, scalar(loss), time.time() - timestamp))
        save_checkpoint(filename, i, model)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model word_to_idx tag_to_idx training_data num_epoch" % sys.argv[0])
    train()
