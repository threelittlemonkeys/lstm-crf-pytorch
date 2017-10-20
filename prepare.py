import sys
import re
from model import BOS, EOS

def load_data():
    data = []
    maxlen = 0
    word_to_idx = {}
    word_to_idx[EOS] = len(word_to_idx)
    tag_to_idx = {}
    tag_to_idx[EOS] = len(tag_to_idx)
    tag_to_idx[BOS] = len(tag_to_idx)
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
            for c in word:
                if c not in word_to_idx:
                    word_to_idx[c] = len(word_to_idx)
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
            sent += list(word)
            tags += [tag] * len(word)
        if len(sent) > maxlen:
            maxlen = len(sent)
        data.append([[word_to_idx[x] for x in sent], [tag_to_idx[x] for x in tags]])
    data.sort(key = lambda x: len(x[0]), reverse = True)
    fo.close()
    return data, maxlen, word_to_idx, tag_to_idx

def save_data(data, maxlen):
    fo = open(sys.argv[1] + ".csv", "w")
    fo.write("%d\n" % maxlen)
    for x in data:
        n = maxlen - len(x[0])
        padding = [0] * n
        x = x[0] + padding + x[1] + padding
        fo.write("%s\n" % " ".join([str(y) for y in x]))
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
    data, maxlen, word_to_idx, tag_to_idx = load_data()
    save_data(data, maxlen)
    save_word_to_idx(word_to_idx)
    save_tag_to_idx(tag_to_idx)
