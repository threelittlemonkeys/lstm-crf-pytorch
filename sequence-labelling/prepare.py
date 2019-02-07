import sys
from utils import *

def load_data():
    data = []
    if KEEP_IDX:
        word_to_idx = load_word_to_idx(sys.argv[1] + ".word_to_idx")
        tag_to_idx = load_tag_to_idx(sys.argv[1] + ".tag_to_idx")
    else:
        word_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tag_to_idx = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
    fo = open(sys.argv[1])
    for line in fo:
        line = line.strip()
        tokens = line.split(" ")
        if len(tokens) < MIN_LEN or len(tokens) > MAX_LEN:
            continue
        seq = []
        tags = []
        for tkn in tokens:
            word, tag = re.split("/(?=[^/]+$)", tkn)
            word = normalize(word)
            if UNIT == "word":
                word = [word]
            if not KEEP_IDX:
                for x in word:
                    if x not in word_to_idx:
                        word_to_idx[x] = len(word_to_idx)
                if tag not in tag_to_idx:
                    tag_to_idx[tag] = len(tag_to_idx)
            seq.extend([str(word_to_idx[x]) if x in word_to_idx else str(UNK_IDX) for x in word])
            tags.extend([str(tag_to_idx[tag])] * len(word))
        data.append(seq + tags)
    data.sort(key = lambda x: -len(x))
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
