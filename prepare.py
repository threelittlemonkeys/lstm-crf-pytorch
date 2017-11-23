import sys
import re
from model import SOS, EOS, PAD
from utils import *

def load_data():
    data = []
    word_to_idx = {PAD: 0, EOS: 1}
    tag_to_idx = {PAD: 0, EOS: 1, SOS: 2}
    fo = open(sys.argv[1])
    for line in fo:
        line = re.sub("\s+", " ", line)
        line = re.sub("^ | $", "", line)
        if line == "":
            continue
        sent = []
        tags = []
        for tkn in line.split(" "):
            word = re.sub("/[A-Z]+", "", tkn)
            tag = re.sub(".+/", "", tkn)
            word = normalize_str(word)
            for c in word:
                if c not in word_to_idx:
                    word_to_idx[c] = len(word_to_idx)
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
            sent.extend(list(word))
            tags.extend([tag] * len(word))
        sent.append(EOS)
        tags.append(EOS)
        data.append([word_to_idx[i] for i in sent] + [tag_to_idx[i] for i in tags])
    data.sort(key = len, reverse = True)
    fo.close()
    return data, word_to_idx, tag_to_idx

def save_data(data):
    fo = open(sys.argv[1] + ".csv", "w")
    for seq in data:
        fo.write("%s %d\n" % (" ".join([str(i) for i in seq]), len(seq) // 2))
    fo.close()

def save_word_to_idx(word_to_idx):
    fo = open("word_to_idx", "w")
    for word, _ in sorted(word_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()

def save_tag_to_idx(tag_to_idx):
    fo = open("tag_to_idx", "w")
    for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tag)
    fo.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, word_to_idx, tag_to_idx = load_data()
    save_data(data)
    save_word_to_idx(word_to_idx)
    save_tag_to_idx(tag_to_idx)
