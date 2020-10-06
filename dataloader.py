from utils import *

class data():
    def __init__(self):
        self.idx = None # input index
        self.x0 = [[]] # string input, raw sentence
        self.x1 = [[]] # string input, tokenized
        self.xc = [[]] # indexed input, character-level
        self.xw = [[]] # indexed input, word-level
        self.y0 = [[]] # actual output
        self.y1 = None # predicted output
        self.prob = None # probability
        self.attn = None # attention heatmap

    def sort(self):
        self.idx = list(range(len(self.xw)))
        self.idx.sort(key = lambda x: -len(self.xw[x]))
        xc = [self.xc[i] for i in self.idx]
        xw = [self.xw[i] for i in self.idx]
        y0 = [self.y0[i] for i in self.idx]
        lens = list(map(len, xw))
        return xc, xw, y0, lens

    def unsort(self):
        self.idx = sorted(range(len(self.x0)), key = lambda x: self.idx[x])
        self.y1 = [self.y1[i] for i in self.idx]
        if self.prob: self.prob = [self.prob[i] for i in self.idx]
        if self.attn: self.attn = [self.attn[i] for i in self.idx]

class dataloader():
    def __init__(self):
        for a, b in data().__dict__.items():
            setattr(self, a, b)

    def append_item(self, x0 = None, x1 = None, xc = None, xw = None, y0 = None):
        if x0: self.x0[-1].append(x0)
        if x1: self.x1[-1].append(x1)
        if xc: self.xc[-1].append(xc)
        if xw: self.xw[-1].append(xw)
        if y0: self.y0[-1].extend(y0)

    def append_row(self):
        self.x0.append([])
        self.x1.append([])
        self.xc.append([])
        self.xw.append([])
        self.y0.append([])

    def strip(self):
        if len(self.xw[-1]):
            return
        self.x0.pop()
        self.x1.pop()
        self.xc.pop()
        self.xw.pop()
        self.y0.pop()

    @staticmethod
    def flatten(ls):
        if HRE:
            return [list(x) for x in ls for x in x]
        return [list(*x) for x in ls]

    def split(self): # split into batches
        for i in range(0, len(self.y0), BATCH_SIZE):
            batch = data()
            j = i + min(BATCH_SIZE, len(self.x0) - i)
            batch.x0 = self.x0[i:j]
            batch.x1 = self.x1[i:j] if HRE else self.flatten(self.x1[i:j])
            batch.xc = self.xc[i:j] if HRE else self.flatten(self.xc[i:j])
            batch.xw = self.xw[i:j] if HRE else self.flatten(self.xw[i:j])
            batch.y0 = self.y0[i:j]
            yield batch

    def tensor(self, bc, bw, lens = None, sos = False, eos = False):
        _p, _s, _e = [PAD_IDX], [SOS_IDX], [EOS_IDX]
        if HRE and lens:
            d_len = max(lens) # document length (Ld)
            i, _bc, _bw = 0, [], []
            for j in lens:
                if sos:
                    _bc.append([[]])
                    _bw.append([])
                _bc += self.flatten(bc[i:i + j])
                _bw += self.flatten(bw[i:i + j])
                _bc += [[[]] for _ in range(d_len - j)]
                _bw += [[] for _ in range(d_len - j)]
                if eos:
                    _bc.append([[]])
                    _bw.append([])
                i += j
            bc, bw = _bc, _bw # [B * Ld, ...]
        if bw:
            s_len = max(map(len, bw)) # sentence length (Ls)
            bw = [_s * sos + x + _e * eos + _p * (s_len - len(x)) for x in bw]
            bw = LongTensor(bw) # [B * Ld, Ls]
        if bc:
            w_len = max(max(map(len, x)) for x in bc) # word length (Lw)
            w_pad = [_p * (w_len + 2)]
            bc = [[_s + w + _e + _p * (w_len - len(w)) for w in x] for x in bc]
            bc = [w_pad * sos + x + w_pad * (s_len - len(x) + eos) for x in bc]
            bc = LongTensor(bc) # [B * Ld, Ls, Lw]
        return bc, bw
