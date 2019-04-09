from utils import *

def load_data():
    data = []
    if KEEP_IDX:
        cti = load_tkn_to_idx(sys.argv[1] + ".char_to_idx")
        tti = load_tkn_to_idx(sys.argv[1] + ".tag_to_idx")
    else:
        cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
        # IOB tags
        tti["B"] = len(tti)
        tti["I"] = len(tti)
    fo = open(sys.argv[1])
    for line in fo:
        line = line.strip()
        tokens = line.split(" ")
        x = []
        y = []
        for word in tokens:
            if not KEEP_IDX:
                for c in word:
                    if c not in cti:
                        cti[c] = len(cti)
            x.extend(["%d:%d" % (cti[c], cti[c]) if c in cti else str(UNK_IDX) for c in word])
            y.extend([str(tti["B"])] + [str(tti["I"])] * (len(word) - 1))
        data.append(x + y)
    data.sort(key = lambda x: -len(x))
    fo.close()
    return data, cti, tti

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    data, cti, tti = load_data()
    save_data(sys.argv[1] + ".csv", data)
    if not KEEP_IDX:
        save_tkn_to_idx(sys.argv[1] + ".char_to_idx", cti)
        save_tkn_to_idx(sys.argv[1] + ".word_to_idx", cti)
        save_tkn_to_idx(sys.argv[1] + ".tag_to_idx", tti)
