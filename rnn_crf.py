from utils import *
from rnn_encoder import *
from crf import *

class rnn_crf(nn.Module):

    def __init__(self, cti, wti, num_tags):

        super().__init__()
        self.rnn = rnn_encoder(cti, wti, num_tags)
        self.crf = crf(num_tags)
        if CUDA: self = self.cuda()

    def forward(self, xc, xw, y0): # for training

        self.zero_grad()
        mask = y0[1:].gt(PAD_IDX).float()

        h = self.rnn(xc, xw, mask)
        loss = self.crf(h, y0, mask)

        return loss

    def decode(self, xc, xw, lens): # for inference

        if HRE:
            mask = [[i > j for j in range(lens[0])] for i in lens]
            mask = Tensor(mask).transpose(0, 1)
        else:
            mask = xw.gt(PAD_IDX).float()

        h = self.rnn(xc, xw, mask)
        y = self.crf.decode(h, mask)

        return y
