from utils import *
from embedding import embed

class rnn_crf(nn.Module):
    def __init__(self, cti_size, wti_size, num_tags):
        super().__init__()
        self.rnn = rnn(cti_size, wti_size, num_tags)
        self.crf = crf(num_tags)
        self = self.cuda() if CUDA else self

    def forward(self, xc, xw, y0): # for training
        self.zero_grad()
        self.rnn.batch_size = y0.size(0)
        self.crf.batch_size = y0.size(0)
        mask = y0[:, 1:].gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y0, mask)
        return torch.mean(Z - score) # NLL loss

    def decode(self, xc, xw, lens): # for inference
        self.rnn.batch_size = len(lens)
        self.crf.batch_size = len(lens)
        if HRE:
            mask = Tensor([[1] * x + [PAD_IDX] * (lens[0] - x) for x in lens])
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
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags) # RNN output to tag

    def init_state(self, b): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, xc, xw, mask):
        hs = self.init_state(self.batch_size)
        x = self.embed(xc, xw)
        if HRE: # [B * doc_len, 1, H] -> [B, doc_len, H]
            x = x.view(self.batch_size, -1, EMBED_SIZE)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first = True)
        h, _ = self.rnn(x, hs)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h

class crf(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.batch_size = 0
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000 # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000 # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000 # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0
        self.trans.data[PAD_IDX, PAD_IDX] = 0

    def forward(self, h, mask): # forward algorithm
        # initialize forward variables in log space
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000) # [B, C]
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score # partition function

    def score(self, h, y0, mask): # calculate the score of a given sequence
        score = Tensor(self.batch_size).fill_(0.)
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y0[t + 1]] for h, y0 in zip(h, y0)])
            trans_t = torch.cat([trans[y0[t + 1], y0[t]] for y0 in y0])
            score += (emit_t + trans_t) * mask_t
        last_tag = y0.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = LongTensor()
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, SOS_IDX] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            i = best_tag[b] # best tag
            j = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:j]):
                i = bptr_t[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path
