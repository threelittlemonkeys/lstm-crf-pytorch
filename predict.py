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

def run_model(model, idx_to_tag, batch):
    batch_size = len(batch) # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [[]], [EOS_IDX], []])
    batch.sort(key = lambda x: -len(x[3]))
    cx_len = max(len(max(x[2], key = len)) for x in batch[:batch_size])
    wx_len = len(batch[0][3])
    batch_cx = []
    if EMBED_UNIT[:4] == "char":
        for x in batch:
            cx = [w + [PAD_IDX] * (cx_len - len(w)) for w in x[2]]
            cx.extend([[PAD_IDX] * cx_len] * (wx_len - len(cx)))
            batch_cx.append(cx)
    batch_wx = [x[3] + [PAD_IDX] * (wx_len - len(x[3])) for x in batch]
    result = model.decode(LongTensor(batch_cx), LongTensor(batch_wx))
    for i in range(batch_size):
        batch[i].append(tuple(idx_to_tag[j] for j in result[i]))
    return [(x[1], x[4], x[5]) for x in sorted(batch[:batch_size])]

def predict(filename, lb, model, char_to_idx, word_to_idx, idx_to_tag):
    data = []
    fo = open(filename)
    for idx, line in enumerate(fo):
        line = line.strip()
        if lb:
            wx, y = zip(*[re.split("/(?=[^/]+$)", x) for x in line.split()])
            wx = [normalize(x) for x in wx]
        else:
            wx, y = tokenize(line, UNIT), ()
        cx = []
        if EMBED_UNIT[:4] == "char":
            for w in wx:
                w = [char_to_idx[c] if c in char_to_idx else UNK_IDX for c in w]
                cx.append([SOS_IDX] + w + [EOS_IDX])
        wx = [word_to_idx[w] if w in word_to_idx else UNK_IDX for w in wx]
        data.append([idx, line, cx, wx, y])
    fo.close()
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i + BATCH_SIZE]
        for y in run_model(model, idx_to_tag, batch):
            yield y

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        result = predict(sys.argv[5], False, *load_model())
        for x, y0, y1 in result:
            print((x, y0, y1))
            # print(iob_to_txt(x, y1, UNIT))
