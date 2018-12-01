import sys
from utils import *

MIN_LEN = 2
MAX_LEN = 50
KEEP_IDX = False # use the existing indices

def load_data():
    data = []
    if KEEP_IDX:
        word_to_idx = load_word_to_idx(sys.argv[1] + ".word_to_idx")
        tag_to_idx = load_tag_to_idx(sys.argv[1] + ".tag_to_idx")
    else:
        word_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tag_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
        # IOB tags
        tag_to_idx["B"] = len(tag_to_idx)
        tag_to_idx["I"] = len(tag_to_idx)
    fo = open(sys.argv[1])
    for line in fo:
        line = line.strip() 
        tokens = line.split(" ")
        if len(tokens) < MIN_LEN or len(tokens) > MAX_LEN:
            continue
        seq = []
        tags = []
        for word in tokens:
            if not KEEP_IDX:
                for c in word:
                    if c not in word_to_idx:
                        word_to_idx[c] = len(word_to_idx)
            ctags = ["B" if i == 0 else "I" for i in range(len(word))]
            seq.extend([str(word_to_idx[c]) if c in word_to_idx else str(UNK_IDX) for c in word])
            tags.extend([str(tag_to_idx[t]) for t in ctags])
        data.append(seq + tags)
    data.sort(key = len, reverse = True)
    fo.close()
    return data, word_to_idx, tag_to_idx

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, word_to_idx, tag_to_idx = load_data()
    save_data(sys.argv[1], data)
    if not KEEP_IDX:
        save_word_to_idx(sys.argv[1], word_to_idx)
        save_tag_to_idx(sys.argv[1], tag_to_idx)
