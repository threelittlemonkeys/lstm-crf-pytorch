from utils import *
from embedding import embed

class rnn_crf(nn.Module):
    def __init__(self, cti_size, wti_size, num_tags):
        super().__init__()
        self.rnn = rnn(cti_size, wti_size, num_tags)
        self.crf = crf(num_tags)
        if CUDA: self = self.cuda()

    def forward(self, xc, xw, y0): # for training
        self.zero_grad()
        self.rnn.batch_size = y0.size(1)
        self.crf.batch_size = y0.size(1)
        mask = y0[1:].gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y0, mask)
        return torch.mean(Z - score) # NLL loss

    def decode(self, xc, xw, lens): # for inference
        self.rnn.batch_size = len(lens)
        self.crf.batch_size = len(lens)
        if HRE:
            mask = [[1] * x + [PAD_IDX] * (lens[0] - x) for x in lens]
            mask = Tensor(mask).transpose(0, 1)
        else:
            mask = xw.gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        return self.crf.decode(h, mask)

class rnn(nn.Module):
    def __init__(self, cti_size, wti_size, num_tags):
        super().__init__()
        self.batch_size = 0

        # architecture
        self.embed = embed(EMBED, cti_size, wti_size, HRE)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags) # RNN output to tag

    def init_state(self): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        b = self.batch_size
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, xc, xw, mask):
        hs = self.init_state()
        x = self.embed(xc, xw)
        if HRE: # [B * Ld, 1, H] -> [Ld, B, H]
            x = x.view(-1, self.batch_size, EMBED_SIZE)
        lens = mask.sum(0).int().cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted = False)
        h, _ = self.rnn(x, hs)
        h, _ = nn.utils.rnn.pad_packed_sequence(h)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h

class crf(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.batch_size = 0
        self.num_tags = num_tags

        # transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000 # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000 # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000 # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0
        self.trans.data[PAD_IDX, PAD_IDX] = 0

    def forward(self, h, mask): # forward algorithm
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for _h, _mask in zip(h, mask):
            _mask = _mask.unsqueeze(1)
            _emit = _h.unsqueeze(2) # [B, C, 1]
            _score = score.unsqueeze(1) + _emit + trans # [B, 1, C] -> [B, C, C]
            _score = log_sum_exp(_score) # [B, C, C] -> [B, C]
            score = _score * _mask + score * (1 - _mask)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score # partition function

    def score(self, h, y0, mask):
        score = Tensor(self.batch_size).fill_(0.)
        h = h.unsqueeze(3) # [L, B, C, 1]
        trans = self.trans.unsqueeze(2) # [C, C, 1]
        for t, (_h, _mask) in enumerate(zip(h, mask)):
            _emit = torch.cat([_h[y0] for _h, y0 in zip(_h, y0[t + 1])])
            _trans = torch.cat([trans[x] for x in zip(y0[t + 1], y0[t])])
            score += (_emit + _trans) * _mask
        last_tag = y0.gather(0, mask.sum(0).long().unsqueeze(0)).squeeze(0)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        bptr = LongTensor()
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, SOS_IDX] = 0.
        for _h, _mask in zip(h, mask):
            _mask = _mask.unsqueeze(1)
            _score = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            _score, _bptr = _score.max(2) # best previous scores and tags
            _score += _h # add emission scores
            bptr = torch.cat((bptr, _bptr.unsqueeze(1)), 1)
            score = _score * _mask + score * (1 - _mask)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            i = best_tag[b]
            j = mask[:, b].sum().int()
            for _bptr in reversed(bptr[b][:j]):
                i = _bptr[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path
