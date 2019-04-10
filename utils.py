import re
import torch

from functools import wraps

from parameters import SOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, CUDA


def normalize(x):
    # x = re.sub("[\uAC00-\uD7A3]+", "\uAC00", x) £ convert Hangeul to 가
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x


def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        return re.sub(" ", "", x)
    elif unit == "word":
        return x.split(" ")


def save_data(filename, data):
    with open(filename, "w") as outfile:
        for seq in data:
            outfile.write(" ".join(seq) + "\n")


def load_tkn_to_idx(filename):
    print("loading {}".format(filename))
    tkn_to_idx = {}
    with open(filename) as infile:
        for line in infile:
            line = line[:-1]
            tkn_to_idx[line] = len(tkn_to_idx)
    return tkn_to_idx


def load_idx_to_tkn(filename):
    print("loading {}".format(filename))
    idx_to_tkn = []
    with open(filename) as infile:
        for line in infile:
            line = line[:-1]
            idx_to_tkn.append(line)
    return idx_to_tkn


def save_tkn_to_idx(filename, tkn_to_idx):
    with open(filename, "w") as outfile:
        for tkn, _ in sorted(tkn_to_idx.items(), key=lambda x: x[1]):
            outfile.write("%s\n" % tkn)


def load_checkpoint(filename, model=None):
    print("loading {}".format(filename))
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = {:d}, loss = {:f}".format(checkpoint["epoch"], checkpoint["loss"]))
    return epoch


def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = {:d}, loss = {:f}, time = {:f}".format(epoch, loss, time))
    if filename and model:
        print("saving {}".format(filename))
        checkpoint = {"state_dict": model.state_dict(), "epoch": epoch, "loss": loss}
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch {:d}".format(epoch))


def cudify(f):
    @wraps(f)
    def cudified(*args):
        x = f(*args)
        return x.cuda() if CUDA else x
    return cudified


Tensor = cudify(torch.Tensor)
LongTensor = cudify(torch.LongTensor)
randn = cudify(torch.randn)
zeros = cudify(torch.zeros)


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))


def batchify(xc, xw, minlen=0, sos=True, eos=True):
    xw_len = max(minlen, max(len(x) for x in xw))
    if xc:
        xc_len = max(minlen, max(len(w) for x in xc for w in x))
        pad = [[PAD_IDX] * (xc_len + 2)]
        xc = [[[SOS_IDX] + w + [EOS_IDX] + [PAD_IDX] * (xc_len - len(w)) for w in x] for x in xc]
        xc = [(pad if sos else []) + x + (pad * (xw_len - len(x) + eos)) for x in xc]
        xc = LongTensor(xc)
    sos = [SOS_IDX] if sos else []
    eos = [EOS_IDX] if eos else []
    xw = [sos + list(x) + eos + [PAD_IDX] * (xw_len - len(x)) for x in xw]
    return xc, LongTensor(xw)


def iob_to_txt(x, y, unit):
    out = ""
    x = tokenize(x, unit)
    for i, j in enumerate(y):
        if i and j[0] == "B":
            out += " "
        out += x[i]
    return out


def f1(prec, recall):
    return 2 * prec * recall / (prec + recall) if prec + recall else 0
