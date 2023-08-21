from utils import *
from rnn_encoder import *
from crf import *

class rnn_crf(nn.Module):

    def __init__(self, cti_size, wti_size, num_tags):

        super().__init__()
        self.rnn = rnn_encoder(cti_size, wti_size, num_tags)
        self.crf = crf(num_tags)
        if CUDA: self = self.cuda()

    def forward(self, xc, xw, y0): # for training

        self.zero_grad()
        mask = y0[1:].gt(PAD_IDX).float()
        h = self.rnn(y0.size(1), xc, xw, mask)
        L = self.crf(h, y0, mask)

        return L

    def decode(self, xc, xw, lens): # for inference

        if HRE:
            mask = [[i > j for j in range(lens[0])] for i in lens]
            mask = Tensor(mask).transpose(0, 1)
        else:
            mask = xw.gt(PAD_IDX).float()

        h = self.rnn(len(lens), xc, xw, mask)
        y = self.crf.decode(h, mask)

        return y
