import sys
import re
from time import time
from os.path import isfile
from collections import defaultdict
from parameters import *

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
randn = lambda *x: torch.randn(*x).cuda() if CUDA else torch.randn
zeros = lambda *x: torch.zeros(*x).cuda() if CUDA else torch.zeros

def normalize(x):

    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()

    return x

def tokenize(x):

    if UNIT == "char":
        return list(re.sub(" ", "", x))
    if UNIT == "char+space":
        return [x.replace("_", "__").replace(" ", "_") for x in x]
    if UNIT in ("word", "sent"):
        return x.split(" ")

def load_tkn_to_idx(filename):

    print("loading %s" % filename)
    tti = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tti[line] = len(tti)
    fo.close()

    return tti

def load_idx_to_tkn(filename):

    print("loading %s" % filename)
    itt = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        itt.append(line)
    fo.close()

    return itt

def load_checkpoint(filename, model = None):

    print("loading %s" % filename)
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))

    return epoch

def save_checkpoint(filename, model, epoch, loss, time):

    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved %s" % filename)

def log_sum_exp(x):

    m = torch.max(x, -1)[0]

    return m + (x - m.unsqueeze(-1)).exp().sum(-1).log()

def tag_to_txt(xs, ys):

    _xs, _ys = [], []
    for x, y in zip(xs, ys):
        if UNIT == "char+space":
            if x == "_":
                y = "_"
            x = x.replace("__", "_")
        if len(_xs) and y in ("I", "E", "I-" + _ys[-1], "E-" + _ys[-1]):
            _xs[-1] += x
            continue
        if y[:2] in ("B-", "I-", "E-", "S-"):
            y = y[2:]
        _xs.append(x)
        _ys.append(y)

    if TASK == "word-classification":
        return " ".join(x + "/" + y for x, y in zip(_xs, _ys))

    if TASK == "word-segmentation":
        return " ".join("".join(x) for x in _xs)

    if TASK == "sentence-segmentation":
        return "\n".join(" ".join(x) for x in _xs)

def f1(p, r):

    return 2 * p * r / (p + r) if p + r else 0
