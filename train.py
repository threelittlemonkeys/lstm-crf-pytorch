import sys
import time
from utils import *
from os.path import isfile

def load_data():
    data = []
    batch_cx = [] # character input
    batch_wx = [] # word input
    batch_y = []
    cx_maxlen = 0 # maximum length of character sequence
    wx_maxlen = 0 # maximum length of word sequence
    char_to_idx = load_tkn_to_idx(sys.argv[2])
    idx_to_word = load_idx_to_tkn(sys.argv[3])
    tag_to_idx = load_tkn_to_idx(sys.argv[4])
    print("loading %s" % sys.argv[5])
    fo = open(sys.argv[5], "r")
    for line in fo:
        line = line.strip()
        tokens = [int(i) for i in line.split(" ")]
        wx_len = len(tokens) // 2
        wx = tokens[:wx_len]
        wx_maxlen = wx_maxlen if wx_maxlen else wx_len # the first line is longest in its mini-batch
        wx_pad = [PAD_IDX] * (wx_maxlen - wx_len)
        batch_wx.append(wx + wx_pad)
        batch_y.append([SOS_IDX] + tokens[wx_len:] + wx_pad)
        if EMBED_UNIT[:4] == "char":
            cx = [idx_to_word[i] for i in wx]
            cx_maxlen = max(cx_maxlen, len(max(cx, key = len)))
            batch_cx.append([[SOS_IDX] + [char_to_idx[c] for c in w] + [EOS_IDX] for w in cx])
        if len(batch_wx) == BATCH_SIZE:
            if EMBED_UNIT[:4] == "char":
                for cx in batch_cx:
                    for w in cx:
                        w += [PAD_IDX] * (cx_maxlen - len(w) + 2)
                    cx += [[PAD_IDX] * (cx_maxlen + 2)] * (wx_maxlen - len(cx))
            data.append((LongTensor(batch_cx), LongTensor(batch_wx), LongTensor(batch_y))) # append a mini-batch
            batch_cx = []
            batch_wx = []
            batch_y = []
            cx_maxlen = 0
            wx_maxlen = 0
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, len(char_to_idx), len(idx_to_word), len(tag_to_idx)

def train():
    num_epochs = int(sys.argv[6])
    data, char_vocab_size, word_vocab_size, num_tags = load_data()
    model = rnn_crf(char_vocab_size, word_vocab_size, num_tags)
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time.time()
        for cx, wx, y in data:
            model.zero_grad()
            loss = torch.mean(model(cx, wx, y)) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss = loss.item()
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage:")
        print("%s model char_to_idx word_to_idx tag_to_idx training_data.csv num_epoch" % sys.argv[0])
        sys.exit()
    print("cuda: %s" % CUDA)
    train()
