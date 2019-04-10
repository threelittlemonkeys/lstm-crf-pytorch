from argparse import ArgumentParser
import re

from utils import save_tkn_to_idx, load_tkn_to_idx, save_data, normalize
from parameters import KEEP_IDX, PAD, SOS, EOS, UNK, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX


def load_data(filename):
    data = []
    if KEEP_IDX:
        cti = load_tkn_to_idx(filename + ".char_to_idx")
        wti = load_tkn_to_idx(filename + ".word_to_idx")
        tti = load_tkn_to_idx(filename + ".tag_to_idx")
    else:
        cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
        tti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
        if FORMAT == "char+iob":
            tti["B"] = len(tti)
            tti["I"] = len(tti)

    with open(filename) as infile:
        for line in infile:
            tokens = line.strip().split(" ")
            x = []
            y = []
            for word in tokens:
                if FORMAT == "word+tag":
                    w, tag = re.split("/(?=[^/]+$)", word)
                    w = normalize(w)
                if not KEEP_IDX:
                    for c in word:
                        if c not in cti:
                            cti[c] = len(cti)
                    if word not in wti:
                        wti[word] = len(wti)
                    if FORMAT == "word+tag":
                        if tag not in tti:
                            tti[tag] = len(tti)
                if FORMAT == "char+iob":
                    x.extend(["%d:%d" % (cti[c], cti[c]) for c in w])
                    y.extend([str(tti["B"])] + [str(tti["I"])] * (len(w) - 1))
                elif FORMAT == "word+tag":
                    x.append("+".join(str(cti[c]) for c in word) + ":%d" % wti[word])
                    y.append(str(tti[tag]))
            print(line)
            print(x)
            print(y)
            data.append(x + y)
    data.sort(key=lambda x: -len(x))
    return data, cti, wti, tti


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("training_data")
    return parser.parse_args()


def main():
    training_filename = parse_args().training_data
    data, cti, wti, tti = load_data()
    save_data(training_filename + ".csv", data)
    if not KEEP_IDX:
        save_tkn_to_idx(training_filename + ".char_to_idx", cti)
        save_tkn_to_idx(training_filename + ".word_to_idx", wti)
        save_tkn_to_idx(training_filename + ".tag_to_idx", tti)


if __name__ == '__main__':
    main()
