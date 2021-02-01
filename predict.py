from model import *
from utils import *
from dataloader import *

def load_model(args):
    cti = load_tkn_to_idx(args[1]) # char_to_idx
    wti = load_tkn_to_idx(args[2]) # word_to_idx
    itt = load_idx_to_tkn(args[3]) # idx_to_tag
    model = rnn_crf(len(cti), len(wti), len(itt))
    print(model)
    load_checkpoint(args[0], model)
    return model, cti, wti, itt

def run_model(model, data, itt):
    with torch.no_grad():
        model.eval()
        for batch in data.split():
            xc, xw, lens = batch.xc, batch.xw, batch.lens
            xc, xw = data.tensor(bc = xc, bw = xw, lens = lens)
            y1 = model.decode(xc, xw, lens)
            batch.y1 = [[itt[i] for i in x] for x in y1]
            for x0, y0, y1 in zip(batch.x0, batch.y0, batch.y1):
                if not HRE:
                    x0, y0, y1 = [x0], [y0], [y1]
                for x0, y0, y1 in zip(x0, y0, y1):
                    yield x0, y0, y1

def predict(model, cti, wti, itt, filename):
    data = dataloader()
    with open(filename) as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))
    for block in text:
        data.append_row()
        for x0 in block.split("\n"):
            if re.match("\S+/\S+( \S+/\S+)*$", x0): # word/tag
                x0, y0 = zip(*[re.split("/(?=[^/]+$)", x) for x in x0.split(" ")])
                x1 = list(map(normalize, x0))
            else:
                y0 = []
                if re.match("[^\t]+\t\S+$", x0): # sentence \t label
                    x0, *y0 = x0.split("\t")
                x0 = tokenize(x0)
                x1 = list(map(normalize, x0))
            xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
            xw = [wti[w] if w in wti else UNK_IDX for w in x1]
            data.append_item(x0 = x0, x1 = x1, xc = xc, xw = xw, y0 = y0)
    return run_model(model, data, itt)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    result = predict(*load_model(sys.argv[1:5]), sys.argv[5])
    func = tag_to_txt if TASK else lambda *x: x
    for x0, y0, y1 in result:
        if y0:
            print(func(x0, y0))
        print(func(x0, y1))
