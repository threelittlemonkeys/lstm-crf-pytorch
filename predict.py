import sys
import re
from model import *
from utils import *

def predict():
    tag_to_idx = load_tag_to_idx(sys.argv[2])
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    word_to_idx = load_word_to_idx(sys.argv[3])
    model = lstm_crf(len(word_to_idx), tag_to_idx)
    if CUDA:
        model = model.cuda()
    print(model)
    load_checkpoint(sys.argv[1], model)
    fo = open(sys.argv[4])
    for line in fo:
        line = re.sub("\s+", "", line)
        sent = sent_to_idx(list(line), word_to_idx)
        pred = [idx_to_tag[i] for i in model(sent)[1]]
        print()
        print(line)
        print(pred)
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model tag_to_idx word_to_idx test_data" % sys.argv[0])
    predict()
