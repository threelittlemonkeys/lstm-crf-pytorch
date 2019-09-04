import sys
import re
from parameters import *

def normalize(x):
    # x = re.sub("[\uAC00-\uD7A3]+", "\uAC00", x) £ convert Hangeul to 가
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, norm = True):
    if norm:
        x = normalize(x)
    if UNIT == "char":
        return re.sub(" ", "", x)
    if UNIT == "word":
        return x.split(" ")

def save_data(filename, data):
    fo = open(filename, "w")
    for seq in data:
        fo.write(" ".join(seq) + "\n")
    fo.close()

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def save_tkn_to_idx(filename, tkn_to_idx):
    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

def load_checkpoint(filename, model = None):
    print("loading %s" % filename)
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f\n" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving %s" % filename)
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def cudify(f):
    return lambda *x: f(*x).cuda() if CUDA else f(*x)

Tensor = cudify(torch.Tensor)
LongTensor = cudify(torch.LongTensor)
randn = cudify(torch.randn)
zeros = cudify(torch.zeros)

def batchify(bxc, bxw, sos = False, eos = False, minlen = 0):
    bxw_len = max(minlen, max(len(x) for x in bxw))
    if bxc:
        bxc_len = max(minlen, max(len(w) for x in bxc for w in x))
        pad = [[PAD_IDX] * (bxc_len + 2)]
        bxc = [[[SOS_IDX, *w, EOS_IDX, *[PAD_IDX] * (bxc_len - len(w))] for w in x] for x in bxc]
        bxc = [(pad if sos else []) + x + (pad * (bxw_len - len(x) + eos)) for x in bxc]
        bxc = LongTensor(bxc)
    sos = [SOS_IDX] if sos else []
    eos = [EOS_IDX] if eos else []
    bxw = [sos + list(x) + eos + [PAD_IDX] * (bxw_len - len(x)) for x in bxw]
    return bxc, LongTensor(bxw)

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

def iob_to_txt(x, y): # for word/sentence segmentation
    out = [[]]
    if re.match("(\S+/\S+( |$))+", x): # token/tag
        x = re.sub(r"/[^ /]+\b", "", x) # remove tags
    for i, (j, k) in enumerate(zip(tokenize(x, False), y)):
        if i and k[0] == "B":
            out.append([])
        out[-1].append(j)
    if FORMAT == "word-segmentation":
        d1, d2 = "", " "
    if FORMAT == "sentence-segmentation":
        d1, d2 = " ", "\n"
    return d2.join(d1.join(x) for x in out)

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0
