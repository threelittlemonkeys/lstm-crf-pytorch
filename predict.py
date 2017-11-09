import sys
import re
from model import *
from utils import *

def load_model():
    word_to_idx = load_word_to_idx(sys.argv[2])
    tag_to_idx = load_tag_to_idx(sys.argv[3])
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = lstm_crf(len(word_to_idx), tag_to_idx)
    if CUDA:
        model = model.cuda()
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, word_to_idx, tag_to_idx, idx_to_tag

def predict():
    data = []
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    fo = open(sys.argv[4])
    for line in fo:
        line = re.sub("\s+", "", line)
        data.append((line, [word_to_idx[i] for i in line] + [word_to_idx[EOS]]))
        if len(data) == BATCH_SIZE:
            result = run_batch(model, idx_to_tag, data)
            data = []
            for x in result:
                print(x)
    fo.close()
    if len(data):
        result = run_batch(model, idx_to_tag, data)
        for x in result:
            print(x)

def run_batch(model, idx_to_tag, data):
    sent = []
    pred = []
    batch = []
    while len(data) < BATCH_SIZE:
        data.append(("", [1]))
    data.sort(key = lambda x: len(x[1]), reverse = True)
    maxlen = len(data[0][1])
    for x, y in data:
        sent.append(x)
        batch.append(y + [0] * (maxlen - len(y)))
    batch = Var(LongTensor(batch))
    for seq in model(batch):
        pred.append([idx_to_tag[i] for i in seq])
    return [(x, y[:-1]) for x, y in zip(sent, pred) if len(x) > 0]

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    predict()
