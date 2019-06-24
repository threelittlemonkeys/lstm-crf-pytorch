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
        for w in tokens:
            w, tag = re.split("/(?=[^/]+$)", w)
            w0 = normalize(w) # for character embedding
            w1 = w0.lower() # for word embedding
            if CASING[:2] == "ul": # prepend the caseness of the first letter
                w0 = ("U" if w[0].isupper() else "L") + w0
            if not KEEP_IDX:
                for c in w0:
                    if c not in cti:
                        cti[c] = len(cti)
                if w1 not in wti:
                    wti[w1] = len(wti)
                if tag not in tti:
                    tti[tag] = len(tti)
            x.append("+".join(str(cti[c]) for c in w0) + ":%d" % wti[w1])
            y.append(str(tti[tag]))
        data.append(x + y)
    fo.close()
    data.sort(key = lambda x: -len(x))
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
