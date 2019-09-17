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
    hre = "hre" in EMBED
    if hre: # sentence level
        txt = fo.read()
        txt = txt.strip()
        txt = txt.split("\n\n")
        for block in txt:
            for line in block.split("\n"):
                x, y = _proc(line, cti, wti, tti, hre)
                print(x)
                print(y)
                # TODO
                data.append(x + [y])
    else: # word level
        for line in fo:
            line = line.strip()
            x, y = _proc(line, cti, wti, tti)
            data.append(x + y)
    fo.close()
    data.sort(key = lambda x: -len(x))
    return data, cti, wti, tti

def _proc(line, cti, wti, tti, hre = False):
    x = []
    y = []
    if hre:
        line, y = line.split("\t")
        if y not in tti:
            tti[y] = len(tti)
        y = str(tti[y])
    for w in line.split(" "):
        w, tag = (w, None) if hre else re.split("/(?=[^/]+$)", w)
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
        if not hre:
            y.append(str(tti[tag]))
    return x, y

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, cti, wti, tti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + ".char_to_idx", cti)
        save_tkn_to_idx(sys.argv[1] + ".word_to_idx", wti)
        save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tti)
