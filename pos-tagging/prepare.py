import sys
from utils import *

def load_data():
    char_data = []
    word_data = []
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
        if len(tokens) < MIN_LEN or len(tokens) > MAX_LEN:
            continue
        char_x = []
        char_y = []
        word_x = []
        word_y = []
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
            char_x.extend([str(char_to_idx[c]) if c in char_to_idx else str(UNK_IDX) for c in word])
            char_y.extend([str(tag_to_idx[tag])] * len(word))
            word_x.append(str(word_to_idx[word]) if word in word_to_idx else str(UNK_IDX))
            word_y.append(str(tag_to_idx[tag]))
        char_data.append(char_x + char_y)
        word_data.append(word_x + word_y)
    char_data.sort(key = lambda x: -len(x))
    word_data.sort(key = lambda x: -len(x))
    fo.close()
    return char_data, word_data, char_to_idx, word_to_idx, tag_to_idx

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    char_data, word_data, char_to_idx, word_to_idx, tag_to_idx = load_data()
    save_data(sys.argv[1] + ".char.csv", char_data)
    save_data(sys.argv[1] + ".word.csv", word_data)
    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + ".char_to_idx", char_to_idx)
        save_tkn_to_idx(sys.argv[1] + ".word_to_idx", word_to_idx)
        save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tag_to_idx)
