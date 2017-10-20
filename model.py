import torch
import torch.nn as nn
from torch.autograd import Variable as Var

BATCH_SIZE = 1
EMBED_SIZE = 500
HIDDEN_SIZE = 1000
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4

NULL = "<NULL>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class lstm_crf(nn.Module):

    def __init__(self, vocab_size, tag_to_idx):
        super(lstm_crf, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.num_tags = len(tag_to_idx)

        # architecture
        self.embed = nn.Embedding(vocab_size, EMBED_SIZE)
        self.lstm = nn.LSTM( \
            input_size = EMBED_SIZE, \
            hidden_size = HIDDEN_SIZE // NUM_DIRS, \
            num_layers = NUM_LAYERS, \
            dropout = DROPOUT, \
            bidirectional = BIDIRECTIONAL \
        )
        self.hidden2tag = nn.Linear(HIDDEN_SIZE, self.num_tags) # LSTM output to tags

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(self.num_tags, self.num_tags))
        self.trans.data[tag_to_idx[START_TAG], :] = -10000 # no transition to START_TAG
        self.trans.data[:, tag_to_idx[STOP_TAG]] = -10000 # no transition from STOP_TAG

        # optimizer
        self.optim = torch.optim.SGD(self.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    def init_hidden(self): # initialize hidden states
        h1 = Var(randn(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS))
        h2 = Var(randn(NUM_LAYERS * NUM_DIRS, BATCH_SIZE, HIDDEN_SIZE // NUM_DIRS))
        return (h1, h2)

    def lstm_forward(self, sent): # LSTM forward pass
        self.hidden = self.init_hidden()
        embed = self.embed(sent).view(len(sent), 1, -1)
        out, self.hidden = self.lstm(embed, self.hidden)
        out = out.view(len(sent), HIDDEN_SIZE)
        out = self.hidden2tag(out)
        return out

    def crf_forward(self, lstm_out): # forward algorithm for CRF
        # initialize forward variables in log space
        alpha = Tensor(1, self.num_tags).fill_(-10000.)
        alpha[0][self.tag_to_idx[START_TAG]] = 0.
        alpha = Var(alpha)
        for feat in lstm_out: # iterate through the sentence
            alpha_t = [] # forward variables at this timestep
            for tag in range(self.num_tags): # for each next tag
                emit_score = feat[tag].view(1, -1).expand(1, self.num_tags)
                trans_score = self.trans[tag].view(1, -1)
                sum = alpha + emit_score + trans_score
                alpha_t.append(log_sum_exp(sum))
            alpha = torch.cat(alpha_t).view(1, -1)
        alpha += self.trans[self.tag_to_idx[STOP_TAG]]
        alpha = log_sum_exp(alpha) # partition function
        return alpha

    def crf_score(self, lstm_out, tags):
        score = Var(Tensor([0]))
        tags = torch.cat([LongTensor([self.tag_to_idx[START_TAG]]), tags])
        for i, emit_score in enumerate(lstm_out):
            score += emit_score[tags[i + 1]] + self.trans[tags[i + 1], tags[i]]
        score += self.trans[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score

    def viterbi(self, lstm_out):
        # initialize backpointers and viterbi variables in log space
        bptr = []
        score = Tensor(1, self.num_tags).fill_(-10000.)
        score[0][self.tag_to_idx[START_TAG]] = 0
        score = Var(score)

        for feat in lstm_out: # iterate through the sentence
            # backpointers and viterbi variables at this timestep
            bptr_t = []
            score_t = []
            for tag in range(self.num_tags): # for each next tag
                sum = score + self.trans[tag]
                best_tag = argmax(sum) # find the best current tag
                bptr_t.append(best_tag)
                score_t.append(sum[0][best_tag])
            score = (torch.cat(score_t) + feat).view(1, -1)
            bptr.append(bptr_t)
        score += self.trans[self.tag_to_idx[STOP_TAG]]
        best_tag = argmax(score)
        best_score = score[0][best_tag]

        best_path = [best_tag]
        for bptr_t in reversed(bptr):
            best_path.append(bptr_t[best_tag])
        best_path = best_path[:-1][::-1]

        return best_score, best_path

    def neg_log_likelihood(self, sent, tags):
        lstm_out = self.lstm_forward(sent)
        score = self.crf_score(lstm_out, tags)
        forward_score = self.crf_forward(lstm_out)
        return forward_score - score # negative log probability

    def forward(self, sent): # LSTM-CRF forward for prediction
        lstm_out = self.lstm_forward(sent)
        score, seq = self.viterbi(lstm_out)
        return score, seq

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x):
    _, i = torch.max(x, 1)
    return scalar(i)

def log_sum_exp(x):
    max_score = x[0, argmax(x)]
    max_score_broadcast = max_score.view(1, -1).expand(1, x.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast)))
