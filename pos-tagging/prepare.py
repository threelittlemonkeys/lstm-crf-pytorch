from utils import *

def load_data():
    data = []
    if KEEP_IDX:
        cti = load_tkn_to_idx(sys.argv[1] + ".char_to_idx")
        wti = load_tkn_to_idx(sys.argv[1] + ".word_to_idx")
        tti = load_tkn_to_idx(sys.argv[1] + ".tag_to_idx")
    else:
        cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
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
                    if c not in cti:
                        cti[c] = len(cti)
                if word not in wti:
                    wti[word] = len(wti)
                if tag not in tti:
                    tti[tag] = len(tti)
            x.append("+".join(str(cti[c]) for c in word) + ":%d" % wti[word])
            y.append(str(tti[tag]))
        data.append(x + y)
    data.sort(key = lambda x: -len(x))
    fo.close()
    return data, cti, wti, tti 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, cti, wti, tti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + ".char_to_idx", cti)
        save_tkn_to_idx(sys.argv[1] + ".word_to_idx", wti)
        save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tti)
