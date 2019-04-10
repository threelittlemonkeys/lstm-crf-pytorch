import re
import torch

from argparse import ArgumentParser

from model import RNN_CRF as rnn_crf
from utils import load_tkn_to_idx, load_idx_to_tkn, load_checkpoint, batchify, normalize, tokenize, iob_to_txt
from parameters import BATCH_SIZE, EOS_IDX, CUDA, UNK_IDX, UNIT, FORMAT


def load_model(args):
    cti = load_tkn_to_idx(args.char_to_idx)  # char_to_idx
    wti = load_tkn_to_idx(args.word_to_idx)  # word_to_idx
    itt = load_idx_to_tkn(args.idx_to_tag)  # idx_to_tag
    model = rnn_crf(len(cti), len(wti), len(itt))
    print(model)
    load_checkpoint(args.model, model)
    return model, cti, wti, itt


def run_model(model, itt, batch):
    batch_size = len(batch)  # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [[]], [EOS_IDX], []])
    batch.sort(key=lambda x: -len(x[3]))
    xc, xw = batchify(*zip(*[(x[2], x[3]) for x in batch]), sos=False, eos=False)
    result = model.decode(xc, xw)
    for i in range(batch_size):
        batch[i].append([itt[j] for j in result[i]])
    return [(x[1], x[4], x[5]) for x in sorted(batch[:batch_size])]


def predict(filename, model, cti, wti, itt):
    data = []
    with open(filename) as infile:
        for idx, line in enumerate(infile):
            line = line.strip()
            if FORMAT == "char+iob":
                wti = cti
                x, y = tokenize(line, UNIT), []
                for w in line.split(" "):
                    y.extend(["B"] + ["I"] * (len(w) - 1))
            elif re.match("(\S+/\S+( |$))+", line):  # if FORMAT == "word+tag":
                x, y = zip(*[re.split("/(?=[^/]+$)", x) for x in line.split()])
                x = [normalize(x) for x in x]
            else:
                x, y = tokenize(line, UNIT), None
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x]
            xw = [wti[w] if w in wti else UNK_IDX for w in x]
            data.append([idx, line, xc, xw, y])
    with torch.no_grad():
        model.eval()
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            for y in run_model(model, itt, batch):
                yield y


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("char_to_idx")
    parser.add_argument("word_to_idx")
    parser.add_argument("idx_to_tag")
    parser.add_argument("test_data")
    return parser.parse_args()


def main():
    args = parse_args()
    print("cuda: %s" % CUDA)
    result = predict(args.test_data, *load_model(args))
    for x, y0, y1 in result:
        if FORMAT == "char+iob":
            print((x, iob_to_txt(x, y1, UNIT)))
        else:
            print((x, y0, y1) if y0 else (x, y1))


if __name__ == '__main__':
    main()
