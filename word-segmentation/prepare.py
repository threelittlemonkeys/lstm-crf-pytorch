import sys
from utils import *
from parameters import *

KEEP_IDX = False # use the existing indices

def load_data():
    data = []
    if KEEP_IDX:
        char_to_idx = load_tkn_to_idx(sys.argv[1] + ".char_to_idx")
        tag_to_idx = load_tkn_to_idx(sys.argv[1] + ".tag_to_idx")
    else:
        char_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tag_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
        # IOB tags
        tag_to_idx["B"] = len(tag_to_idx)
        tag_to_idx["I"] = len(tag_to_idx)
    fo = open(sys.argv[1])
    for line in fo:
        line = line.strip() 
        tokens = line.split(" ")
        seq = []
        tags = []
        for word in tokens:
            if not KEEP_IDX:
                for c in word:
                    if c not in char_to_idx:
                        char_to_idx[c] = len(char_to_idx)
            ctags = ["B" if i == 0 else "I" for i in range(len(word))]
            seq.extend([str(char_to_idx[c]) if c in char_to_idx else str(UNK_IDX) for c in word])
            tags.extend([str(tag_to_idx[t]) for t in ctags])
        data.append(seq + tags)
    data.sort(key = lambda x: -len(x))
    fo.close()
    return data, char_to_idx, tag_to_idx

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, char_to_idx, tag_to_idx = load_data()
    save_data(sys.argv[1] + ".csv", data)
    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + ".char_to_idx", char_to_idx)
        save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tag_to_idx)
