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

def run_model(model, itt, data):
    data.sort()
    for xc, xw, _, y0_lens in data.split():
        xc, xw = data.tensor(xc, xw, doc_lens = y0_lens)
        y1 = model.decode(xc, xw, y0_lens)
        data.append_item(y1 = [[itt[i] for i in x] for x in y1])
    data.unsort()
    for x, y0, y1 in zip(data.x, data.y0, data.y1):
        if HRE:
            for x, y0, y1 in zip(x, y0, y1):
                yield x, y0, y1
        else:
            yield x, y0, y1

def predict(filename, model, cti, wti, itt):
    data = dataset()
    fo = open(filename)
    for line in fo:
        line = line.strip()
        if line:
            if re.match("(\S+/\S+( |$))+$", line): # word/tag
                x, y = zip(*[re.split("/(?=[^/]+$)", x) for x in line.split(" ")])
                line = " ".join(x)
            elif re.match("\S+( \S+)*\t\S+$", line): # sentence \t label
                line, y = line.split("\t")
            else: # no ground truth provided
                y = []
            x = tokenize(line)
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x]
            xw = [wti[w] if w in wti else UNK_IDX for w in x]
            data.append_item(x = line, xc = xc, xw = xw, y0 = y)
        if not (HRE and line): # delimiters (\n, \n\n)
            data.append_row()
    fo.close()
    data.strip()
    with torch.no_grad():
        model.eval()
        return run_model(model, itt, data)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    result = predict(sys.argv[5], *load_model())
    for x, y0, y1 in result:
        if not FORMAT:
            print((x, y0, y1) if y0 else (x, y1))
        else: # word/sentence segmentation
            if y0:
                print(iob_to_txt(x, y0))
            print(iob_to_txt(x, y1))
            print()
