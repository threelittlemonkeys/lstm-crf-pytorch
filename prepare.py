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
    if HRE:
        txt = fo.read().strip().split("\n\n")
        for doc in txt:
            data.append([])
            for line in doc.split("\n"):
                x, y = load_line(line, cti, wti, tti)
                data[-1].append((x, y))
        _data = []
        for doc in sorted(data, key = lambda x: -len(x)):
            _data.extend(doc)
            _data.append(None)
        data = _data[:-1]
    else:
        for line in fo:
            x, y = load_line(line, cti, wti, tti)
            data.append((x, y))
        data.sort(key = lambda x: -len(x[0])) # sort by source sequence length
    fo.close()

    return data, cti, wti, tti

def load_line(line, cti, wti, tti):

    line = line.strip()
    x, y = [], []

    if HRE:
        line, y = line.split("\t")
        if y not in tti:
            tti[y] = len(tti)
        y = [str(tti[y])]

    for w in line.split(" "):
        w, tag = (w, None) if HRE else re.split("/(?=[^/]+$)", w)
        w0 = normalize(w) # for character embedding
        w1 = w0.lower() # for word embedding
        if not KEEP_IDX:
            for c in w0:
                if c not in cti:
                    cti[c] = len(cti)
            if w1 not in wti:
                wti[w1] = len(wti)
            if tag and tag not in tti:
                tti[tag] = len(tti)
        x.append("+".join(str(cti[c]) for c in w0) + ":%d" % wti[w1])
        if tag:
            y.append(str(tti[tag]))

    return x, y

def save_data(filename, data):

    fo = open(filename, "w")
    for seq in data:
        if not seq:
            print(file = fo)
            continue
        print(*seq[0], end = "\t", file = fo)
        print(*seq[1], file = fo)
    fo.close()

def save_tkn_to_idx(filename, tkn_to_idx):

    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])

    data, cti, wti, tti = load_data()
    save_data(sys.argv[1] + ".csv", data)

    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + ".char_to_idx", cti)
        save_tkn_to_idx(sys.argv[1] + ".word_to_idx", wti)
        save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tti)
