from utils import *

class dataset:
    _vars = ("x0", "x1", "xc", "xw", "y0")

    def __init__(self):
        self.x0 = [] # input strings, raw
        self.x1 = [] # input strings, tokenized
        self.xc = [] # input indices, character-level
        self.xw = [] # input indices, word-level
        self.y0 = [] # actual output
        self.y1 = None # predicted output
        self.lens = None # sequence lengths
        self.prob = None # probability
        self.attn = None # attention heatmap

    '''
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
    '''

class dataloader(dataset):
    def __init__(self):
        super().__init__()

    def append_row(self):
        for x in self._vars:
            getattr(self, x).append([])

    def append_item(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k)[-1].append(v)

    @staticmethod
    def flatten(x): # [Ld, Ls, Lw] -> [Ld * Ls, Lw]
        if HRE:
            return [list(x) for x in x for x in x]
        return [list(*x) for x in x]

    def split(self): # split into batches
        if HRE:
            self.y0 = [[tuple(x[0] for x in x)] for x in self.y0]
        for i in range(0, len(self.y0), BATCH_SIZE):
            batch = dataset()
            j = i + min(BATCH_SIZE, len(self.x0) - i)
            batch.lens = list(map(len, self.xw[i:j]))
            for x in self._vars:
                setattr(batch, x, self.flatten(getattr(self, x)[i:j]))
            yield batch

    def tensor(self, bc = None, bw = None, lens = None, sos = False, eos = False):
        p, s, e = [PAD_IDX], [SOS_IDX], [EOS_IDX]
        if HRE and lens:
            dl = max(lens) # document length (Ld)
            i, _bc, _bw = 0, [], []
            for j in lens:
                if bc:
                    if sos: _bc.append([[]])
                    _bc += bc[i:i + j] + [[[]] for _ in range(dl - j)]
                    if eos: _bc.append([[]])
                if bw:
                    if sos: _bw.append([])
                    _bw += bw[i:i + j] + [[] for _ in range(dl - j)]
                    if eos: _bw.append([])
                i += j
            bc, bw = _bc, _bw # [B * Ld, ...]
        if bw:
            sl = max(map(len, bw)) # sentence length (Ls)
            bw = [s * sos + x + e * eos + p * (sl - len(x)) for x in bw]
            bw = LongTensor(bw).transpose(0, 1) # [Ls, B * Ld]
        if bc:
            wl = max(max(map(len, x)) for x in bc) # word length (Lw)
            wp = [p * (wl + 2)]
            bc = [[s + x + e + p * (wl - len(x)) for x in x] for x in bc]
            bc = [wp * sos + x + wp * (sl - len(x) + eos) for x in bc]
            bc = LongTensor(bc).transpose(0, 1) # [Ls, B * Ld, Lw]
        return bc, bw
