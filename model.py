import torch
import torch.nn as nn
from torch.autograd import Variable as Var

BATCH_SIZE = 64
EMBED_SIZE = 50
HIDDEN_SIZE = 300
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 10

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class lstm_crf(nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()

        # architecture
        self.lstm = lstm(vocab_size, num_tags)
        self.crf = crf(num_tags)

    def forward(self, x, y0):
        y, lens = self.lstm(x)
        '''
        # iterative training
        Z = self.crf.forward_iter(y, lens)
        score = self.crf.score_iter(y, y0, lens)
        '''
        # mini-batch training
        mask = x.data.gt(0).float()
        y = y * Var(mask.unsqueeze(-1).expand_as(y))
        Z = self.crf.forward(y, mask)
        score = self.crf.score(y, y0, mask)

        return Z - score # negative log likelihood

    def decode(self, x):
        result = []
        y = self.lstm(x)
        for i in range(len(self.lstm.lens)):
            if self.lstm.lens[i] > 1:
                best_path = self.crf.decode(y[i][:self.lstm.lens[i]])
            else:
                best_path = []
            result.append(best_path)
        return result

class lstm(nn.Module):

    def __init__(self, vocab_size, num_tags):
        super().__init__()
        # self.num_tags = num_tags # Python 2

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx = PAD_IDX)
        self.lstm = nn.LSTM(
            input_size = EMBED_SIZE,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = BIDIRECTIONAL
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags) # LSTM output to tag

    def init_hidden(self): # initialize hidden states
        h = Var(zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS)) # hidden states
        c = Var(zeros(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS)) # cell states
        return (h, c)

    def forward(self, x):
        self.hidden = self.init_hidden()
        self.lens = [len_unpadded(seq) for seq in x]
        embed = self.embed(x)
        embed = nn.utils.rnn.pack_padded_sequence(embed, self.lens, batch_first = True)
        y, _ = self.lstm(embed, self.hidden)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first = True)
        # y = y.contiguous().view(-1, HIDDEN_SIZE) # Python 2
        y = self.out(y)
        # y = y.view(BATCH_SIZE, -1, self.num_tags) # Python 2
        return y, self.lens

class crf(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000. # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000. # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000. # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0.
        self.trans.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, y, mask): # forward algorithm for mini-batch training
        # initialize forward variables in log space
        Z = Tensor(BATCH_SIZE, self.num_tags).fill_(-10000.)
        Z[:, SOS_IDX] = 0.
        Z = Var(Z)
        for t in range(y.size(1)): # iterate through the sequence
            score_t = [] # forward variables at this timestep
            len_t = int(torch.sum(mask[:, t])) # masked batch length
            for f in range(self.num_tags): # for each next tag
                emit = torch.cat([y[b, t, f].expand(1, self.num_tags) for b in range(len_t)])
                trans = self.trans[f].expand(len_t, self.num_tags)
                z = log_sum_exp_batch2(Z, emit + trans)
                score_t.append(z)
            Z = torch.cat(score_t, 1)
        Z = log_sum_exp_batch1(Z).view(BATCH_SIZE) # partition function
        return Z

    def score(self, y, y0, mask): # scoring a label for mini-batch training
        score = Var(Tensor(BATCH_SIZE).fill_(0.))
        y0 = torch.cat([LongTensor(BATCH_SIZE, 1).fill_(SOS_IDX), y0], 1)
        for t in range(y.size(1)): # iterate through the sequence
            mask_t = Var(mask[:, t])
            emit = torch.cat([y[b, t, y0[b, t + 1]] for b in range(BATCH_SIZE)])
            trans = torch.cat([self.trans[seq[t + 1], seq[t]] for seq in y0]) * mask_t
            score = score + emit + trans
        return score
    '''
    def forward_iter(self, y, lens): # forward algorithm for iterative training
        # initialize forward variables in log space
        Z = Tensor(BATCH_SIZE, self.num_tags).fill_(-10000.)
        Z[:, SOS_IDX] = 0.
        Z = Var(Z)
        for b in range(len(lens)):
            for t in range(lens[b]): # iterate through the sequence
                score_t = [] # forward variables at this timestep
                for f in range(self.num_tags): # for each next tag
                    emit = y[b, t, f].expand(self.num_tags)
                    trans = self.trans[f].expand(self.num_tags)
                    z = log_sum_exp(Z[b] + emit + trans)
                    score_t.append(z)
                Z[b] = torch.cat(score_t)
        Z = torch.cat([log_sum_exp(i) for i in Z]) # partition function
        return Z

    def score_iter(self, y, y0, lens): # scoring a label for iterative training
        score = Var(Tensor(BATCH_SIZE).fill_(0.))
        y0 = torch.cat([LongTensor(BATCH_SIZE, 1).fill_(SOS_IDX), y0], 1)
        for b in range(len(lens)):
            for t in range(lens[b]): # iterate through the sequence
                emit = y[b, t, y0[b, t + 1]]
                trans = self.trans[y0[b, t + 1], y0[b, t]]
                score[b] = score[b] + emit + trans
        return score
    '''
    def decode(self, y): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = []
        score = Tensor(self.num_tags).fill_(-10000.)
        score[SOS_IDX] = 0.
        score = Var(score)

        for t in range(len(y)): # iterate through the sequence
            # backpointers and viterbi variables at this timestep
            bptr_t = []
            score_t = []
            for i in range(self.num_tags): # for each next tag
                z = score + self.trans[i]
                best_tag = argmax(z) # find the best previous tag
                bptr_t.append(best_tag)
                score_t.append(z[best_tag])
            bptr.append(bptr_t)
            score = torch.cat(score_t) + y[t]
        best_tag = argmax(score)
        best_score = score[best_tag]

        # back-tracking
        best_path = [best_tag]
        for bptr_t in reversed(bptr):
            best_path.append(bptr_t[best_tag])
        best_path = reversed(best_path[:-1])

        return best_path

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def len_unpadded(x): # get unpadded sequence length
    return next((i for i, j in enumerate(x) if scalar(j) == 0), len(x))

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x):
    return scalar(torch.max(x, 0)[1]) # for 1D tensor

def log_sum_exp(x):
    max_score = x[argmax(x)]
    max_score_broadcast = max_score.expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast)))

def log_sum_exp_batch1(x): # for a single vector
    max_score = torch.cat([i[argmax(i)] for i in x])
    max_score_broadcast = max_score.view(-1, 1).expand_as(x)
    z = max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), 1))
    return z

def log_sum_exp_batch2(x, y): # for two vectors of different sizes
    z = x[:len(y)] + y
    max_score = torch.cat([i[argmax(i)] for i in z])
    max_score_broadcast = max_score.view(-1, 1).expand_as(z)
    z = max_score + torch.log(torch.sum(torch.exp(z - max_score_broadcast), 1))
    if len(x) > len(z):
        z = torch.cat((z, torch.cat([i[argmax(i)] for i in x[len(y):]])))
    return z.view(len(x), 1)
