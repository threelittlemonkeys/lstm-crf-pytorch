from model import *
from utils import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tag
    model = rnn_crf(len(cti), len(wti), len(itt))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti, itt

def run_model(model, itt, batch):
    batch_size = len(batch) # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [[]], [EOS_IDX], []])
    batch.sort(key = lambda x: -len(x[3]))
    xc, xw = batchify(*zip(*[(x[2], x[3]) for x in batch]))
    result = model.decode(xc, xw)
    for i in range(batch_size):
        batch[i].append(tuple(itt[j] for j in result[i]))
    return [(x[1], x[4], x[5]) for x in sorted(batch[:batch_size])]

def predict(filename, model, cti, wti, itt):
    data = []
    fo = open(filename)
    for idx, line in enumerate(fo):
        line = line.strip()
        if re.match("(\S+/\S+( |$))+", line):
            x, y = zip(*[re.split("/(?=[^/]+$)", x) for x in line.split()])
            x = [normalize(x) for x in x]
        else:
            x, y = tokenize(line, UNIT), None
        xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x]
        xw = [wti[w] if w in wti else UNK_IDX for w in x]
        data.append([idx, line, xc, xw, y])
    fo.close()
    with torch.no_grad():
         model.eval()
         for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            for y in run_model(model, itt, batch):
                yield y

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    result = predict(sys.argv[5], *load_model())
    for x, y0, y1 in result:
        print((x, y0, y1) if y0 else (x, y1))
        # print(iob_to_txt(x, y1, UNIT))
