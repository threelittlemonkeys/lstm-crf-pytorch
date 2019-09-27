import sys
import re
from time import time
from os.path import isfile
from parameters import *
from collections import defaultdict

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
    if UNIT in ("word", "sent"):
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
    if model: model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
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

class dataset():
    def __init__(self):
        self.idx = []
        self.x = [[]] # input sequences
        self.xc = [[]] # input character sequences
        self.xw = [[]] # input word sequences
        self.y0 = [[]] if HRE else [] # actual labels
        self.y1 = [] # predicted labels

    def append_item(self, idx = -1, x = None, xc = None, xw = None, y0 = None, y1 = None):
        if idx >= 0 : self.idx.append(idx)
        if x: self.x[-1].append(x)
        if xc: self.xc[-1].append(xc)
        if xw: self.xw[-1].append(xw)
        if y0: (self.y0[-1] if HRE else self.y0).append(y0)
        if y1: self.y1.extend(y1)

    def append_list(self):
        self.x.append([])
        self.xc.append([])
        self.xw.append([])
        if HRE: self.y0.append([])

    def strip(self):
        while len(self.xw[-1]) == 0:
            self.x.pop()
            self.xc.pop()
            self.xw.pop()
            if HRE: self.y0.pop()

    def sort(self):
        self.idx = list(range(len(self.x)))
        self.idx.sort(key = lambda x: -len(self.xw[x] if HRE else self.xw[x][0]))
        self.xc = [self.xc[i] for i in self.idx]
        self.xw = [self.xw[i] for i in self.idx]

    def unsort(self):
        idx = sorted(range(len(self.idx)), key = lambda x: self.idx[x])
        self.idx = list(range(len(self.x)))
        self.xc = [self.xc[i] for i in idx]
        self.xw = [self.xw[i] for i in idx]
        self.y1 = [self.y1[i] for i in idx]

    def split(self): # split into batches
        for i in range(0, len(self.y0), BATCH_SIZE):
            j = i + BATCH_SIZE
            y0 = self.y0[i:j]
            y0_lens = [len(x) for x in self.xw[i:j]] if HRE else None
            if HRE:
                xc = [list(x) for x in self.xc[i:j] for x in x]
                xw = [list(x) for x in self.xw[i:j] for x in x]
            else:
                xc = [list(*x) for x in self.xc[i:j]]
                xw = [list(*x) for x in self.xw[i:j]]
            yield xc, xw, y0, y0_lens

    def tensor(self, bc, bw, _sos = False, _eos = False, doc_lens = None):
        sos, eos, pad = [SOS_IDX], [EOS_IDX], [PAD_IDX]
        if doc_lens:
            d_len = max(doc_lens) # doc_len (Ld)
            i, _bc, _bw = 0, [], []
            for j in doc_lens:
                _bc.extend(bc[i:i + j] + [[pad]] * (d_len - j))
                _bw.extend(bw[i:i + j] + [pad] * (d_len - j))
                i += j
            bc, bw = _bc, _bw
        if bw:
            s_len = max(map(len, bw)) # sent_len (Ls)
            bw = [sos * _sos + x + eos * _eos + pad * (s_len - len(x)) for x in bw]
            bw = LongTensor(bw) # [B * Ld, Ls]
        if bc:
            w_len = max(max(map(len, x)) for x in bc) # word_len (Lw)
            w_pad = [pad * (w_len + 2)]
            bc = [[sos + w + eos + pad * (w_len - len(w)) for w in x] for x in bc]
            bc = [w_pad * _sos + x + w_pad * (s_len - len(x) + _eos) for x in bc]
            bc = LongTensor(bc) # [B * Ld, Ls, Lw]
        return bc, bw

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
