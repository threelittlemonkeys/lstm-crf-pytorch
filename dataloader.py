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
    def flatten(ls): # -> [[], ...]
        if not HRE:
            return [list(*x) for x in ls] # [[()], ...]
        return [list(x) for x in ls for x in x] # [[(), ...], ...]

    def split(self): # split into batches
        for i in range(0, len(self.y0), BATCH_SIZE):
            batch = dataset()
            j = i + min(BATCH_SIZE, len(self.x0) - i)
            for x in self._vars:
                setattr(batch, x, self.flatten(getattr(self, x)[i:j]))
            batch.lens = list(map(len, batch.xw))
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
