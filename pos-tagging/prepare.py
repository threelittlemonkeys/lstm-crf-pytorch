import sys
from utils import *

KEEP_IDX = False # use the existing indices

def load_data():
    data = []
    if KEEP_IDX:
        char_to_idx = load_tkn_to_idx(sys.argv[1] + ".char_to_idx")
        word_to_idx = load_tkn_to_idx(sys.argv[1] + ".word_to_idx")
        tag_to_idx = load_tkn_to_idx(sys.argv[1] + ".tag_to_idx")
    else:
        char_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        word_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tag_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        line = line.strip()
        tokens = line.split(" ")
        x = []
        y = []
        for tkn in tokens:
            word, tag = re.split("/(?=[^/]+$)", tkn)
            word = normalize(word)
            if not KEEP_IDX:
                for c in word:
                    if c not in char_to_idx:
                        char_to_idx[c] = len(char_to_idx)
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = len(tag_to_idx)
            x.append(str(word_to_idx[word]) if word in word_to_idx else str(UNK_IDX))
            y.append(str(tag_to_idx[tag]))
        data.append(x + y)
    data.sort(key = lambda x: -len(x))
    fo.close()
    return data, char_to_idx, word_to_idx, tag_to_idx

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, char_to_idx, word_to_idx, tag_to_idx = load_data()
    save_data(sys.argv[1] + ".csv", data)
    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + ".char_to_idx", char_to_idx)
        save_tkn_to_idx(sys.argv[1] + ".word_to_idx", word_to_idx)
        save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tag_to_idx)
