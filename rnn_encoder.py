from utils import *
from embedding import *

class rnn_encoder(nn.Module):

    def __init__(self, cti, wti, num_tags):

        super().__init__()

        # architecture
        self.embed = embed(EMBED, cti, wti, hre = HRE)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.out = nn.Linear(HIDDEN_SIZE, num_tags) # RNN output to tag

    def init_state(self, b): # initialize RNN states

        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "GRU":
            return hs
        cs = zeros(n, b, h) # LSTM cell state
        return (hs, cs)

    def forward(self, xc, xw, mask):

        b = mask.size(1)
        s = self.init_state(b)
        lens = mask.sum(0).int().cpu()

        x = self.embed(b, xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted = False)
        h, _ = self.rnn(x, s)
        h, _ = nn.utils.rnn.pad_packed_sequence(h)
        h = self.out(h)
        h *= mask.unsqueeze(2)

        return h
