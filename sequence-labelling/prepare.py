import sys
import re
sys.path.append("..")
from model import SOS, EOS, PAD, UNK, SOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX
from utils import normalize

MIN_LENGTH = 10
MAX_LENGTH = 50

def load_data():
    data = []
    word_to_idx = {PAD: PAD_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
    tag_to_idx = {PAD: PAD_IDX, EOS: EOS_IDX, SOS: SOS_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        line = line.strip()
        tokens = line.split(" ")
        if len(tokens) < MIN_LENGTH or len(tokens) > MAX_LENGTH:
            continue
        seq = []
        tags = []
        for tkn in tokens:
            word, tag = re.split("/(?=[^/]+$)", tkn)
            word = normalize(word)
            for c in word:
                if c not in word_to_idx:
                    word_to_idx[c] = len(word_to_idx)
            if tag not in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
            seq.extend([str(word_to_idx[c]) for c in word])
            tags.extend([str(tag_to_idx[tag])] * len(word))
        data.append(seq + tags)
    data.sort(key = len, reverse = True)
    fo.close()
    return data, word_to_idx, tag_to_idx

def save_data(data):
    fo = open(sys.argv[1] + ".csv", "w")
    for seq in data:
        fo.write(" ".join(seq) + "\n")
    fo.close()

def save_word_to_idx(word_to_idx):
    fo = open(sys.argv[1] + ".word_to_idx", "w")
    for word, _ in sorted(word_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()

def save_tag_to_idx(tag_to_idx):
    fo = open(sys.argv[1] + ".tag_to_idx", "w")
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
