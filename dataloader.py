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

class dataloader(dataset):
    def __init__(self, hre = False):
        super().__init__()
        self.hre = hre # hierarchical recurrent encoding

    def append_row(self):
        for x in self._vars:
            getattr(self, x).append([])

    def append_item(self, **kwargs):
        for k, v in kwargs.items():
            getattr(self, k)[-1].append(v)

    def flatten(self, x): # [Ld, Ls, Lw] -> [Ld * Ls, Lw]
        if self.hre:
            return [list(x) for x in x for x in x]
        try:
            return [x if type(x[0]) == str else list(*x) for x in x]
        except:
            return [x for x in x for x in x]

    def split(self): # split into batches
        if self.hre:
            self.y0 = [[tuple(y[0] for y in y)] for y in self.y0]
        for i in range(0, len(self.y0), BATCH_SIZE):
            batch = dataset()
            j = i + min(BATCH_SIZE, len(self.x0) - i)
            batch.lens = list(map(len, self.xw[i:j]))
            for x in self._vars:
                setattr(batch, x, self.flatten(getattr(self, x)[i:j]))
            yield batch

    def tensor(self, bc = None, bw = None, lens = None, sos = False, eos = False):
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
            try:
                sl = max(map(len, bw)) # sentence length (Ls)
                bw = [s * sos + x + e * eos + p * (sl - len(x)) for x in bw]
                bw = LongTensor(bw).transpose(0, 1) # [Ls, B * Ld]
            except:
                bw = LongTensor(bw)
        if bc:
            wl = max(max(map(len, x)) for x in bc) # word length (Lw)
            wp = [p * (wl + 2)]
            bc = [[s + x + e + p * (wl - len(x)) for x in x] for x in bc]
            bc = [wp * sos + x + wp * (sl - len(x) + eos) for x in bc]
            bc = LongTensor(bc).transpose(0, 1) # [Ls, B * Ld, Lw]
        return bc, bw
