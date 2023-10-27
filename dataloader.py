from utils import *

class dataset():

    _vars = ("x0", "x1", "xc", "xw", "y0")

    def __init__(self):

        self.idx = None # input index
        self.x0 = [] # text input, raw
        self.x1 = [] # text input, tokenized
        self.xc = [] # indexed input, character-level
        self.xw = [] # indexed input, word-level
        self.y0 = [] # actual output
        self.y1 = None # predicted output
        self.lens = None # sequence lengths (for HRE)
        self.prob = None # output probabilities
        self.attn = None # attention weights
        self.copy = None # copy weights

    def sort(self): # HRE = False

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
        if self.prob:
            self.prob = [self.prob[i] for i in self.idx]
        if self.attn:
            self.attn = [self.attn[i] for i in self.idx]

class dataloader(dataset):

    def __init__(self, batch_first = False, hre = False):

        super().__init__()
        self.batch_first = batch_first
        self.hre = hre # hierarchical recurrent encoding

    def append_row(self):

        for x in self._vars:
            getattr(self, x).append([])

    def append_item(self, **kwargs):

        for k, v in kwargs.items():
            getattr(self, k)[-1].append(v)

    def clone_row(self):

        for x in self._vars:
            getattr(self, x).append(getattr(self, x)[-1])

    def flatten(self, x): # [Ld, Ls, Lw] -> [Ld * Ls, Lw]

        if self.hre:
            return [list(x) for x in x for x in x]
        try:
            return [x if type(x[0]) == str else list(*x) for x in x]
        except:
            return [x for x in x for x in x]

    def batchify(self, batch_size):

        if self.hre:
            self.x0 = [[x] for x in self.x0]
            self.y0 = [[[y[0] if y else None for y in y]] for y in self.y0]

        for i in range(0, len(self.y0), batch_size):
            batch = dataset()
            j = i + min(batch_size, len(self.x0) - i)
            batch.lens = list(map(len, self.xw[i:j]))
            for x in self._vars:
                setattr(batch, x, self.flatten(getattr(self, x)[i:j]))
            yield batch

    def to_tensor(self, bc = None, bw = None, lens = None, sos = False, eos = False):

        p, s, e = [PAD_IDX], [SOS_IDX], [EOS_IDX]

        if self.hre and lens:
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
            bw = LongTensor(bw) # [B * Ld, Ls]
            if not self.batch_first:
                bw.transpose_(0, 1)


        if bc:
            wl = max(max(map(len, x)) for x in bc) # word length (Lw)
            wp = [p * (wl + 2)]
            bc = [[s + x + e + p * (wl - len(x)) for x in x] for x in bc]
            bc = [wp * sos + x + wp * (sl - len(x) + eos) for x in bc]
            bc = LongTensor(bc) # [B * Ld, Ls, Lw]
            if not self.batch_first:
                bc.transpose_(0, 1)

        return bc, bw
