import sys
from utils import *

def load_model():
    char_to_idx = load_tkn_to_idx(sys.argv[2])
    word_to_idx = load_tkn_to_idx(sys.argv[3])
    idx_to_tag = load_idx_to_tkn(sys.argv[4])
    model = rnn_crf(len(char_to_idx), len(word_to_idx), len(idx_to_tag))
    print(model)
    model.eval()
    load_checkpoint(sys.argv[1], model)
    return model, char_to_idx, word_to_idx, idx_to_tag

def run_model(model, idx_to_tag, data, cx_maxlen):
    z = len(data)
    while len(data) < BATCH_SIZE:
        data.append([-1, "", [[]], [EOS_IDX], []])
    data.sort(key = lambda x: -len(x[3]))
    wx_maxlen = len(data[0][3])
    batch_wx = [x[3] + [PAD_IDX] * (wx_maxlen - len(x[3])) for x in data]
    batch_cx = []
    if EMBED_UNIT[:4] == "char":
        for x in data:
            cx = [w + [PAD_IDX] * (cx_maxlen - len(w) + 2) for w in x[2]]
            cx += [[PAD_IDX] * (cx_maxlen + 2)] * (wx_maxlen - len(cx))
            batch_cx.append(cx)
    result = model.decode(LongTensor(batch_cx), LongTensor(batch_wx))
    for i in range(z):
        data[i].append(tuple(idx_to_tag[j] for j in result[i]))
    return [(x[1], x[4], x[5]) for x in sorted(data[:z])]

def predict(lb = False):
    idx = 0
    data = []
    result = []
    cx_maxlen = 0 # maximum length of character sequence
    model, char_to_idx, word_to_idx, idx_to_tag = load_model()
    fo = open(sys.argv[5])
    for line in fo:
        line = line.strip()
        if lb:
            wx, y = zip(*[re.split("/(?=[^/]+$)", x) for x in line.split()])
            wx = [normalize(x) for x in wx]
        else:
            wx, y = tokenize(line, UNIT), ()
        cx = []
        if EMBED_UNIT[:4] == "char":
            for w in wx:
                cx += [[SOS_IDX] + [char_to_idx[c] if c in char_to_idx else UNK_IDX for c in w] + [EOS_IDX]]
            cx_maxlen = max(cx_maxlen, len(max(wx, key = len)))
        wx = [word_to_idx[w] if w in word_to_idx else UNK_IDX for w in wx]
        data.append([idx, line, cx, wx, y])
        if len(data) == BATCH_SIZE:
            result.extend(run_model(model, idx_to_tag, data, cx_maxlen))
            data = []
            cx_maxlen = 0
        idx += 1
    fo.close()
    if len(data):
        result.extend(run_model(model, idx_to_tag, data, cx_maxlen))
    return result

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        result = predict()
        for x, y0, y1 in result:
            print((x, y0, y1))
            # print(iob_to_txt(x, y1, UNIT))
