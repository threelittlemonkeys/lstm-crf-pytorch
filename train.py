from model import *
from utils import *
from evaluate import *

def load_data():
    data = dataset()
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tkn
    print("loading %s..." % sys.argv[5])
    fo = open(sys.argv[5], "r")
    for line in fo:
        line = line.strip()
        if line:
            seq = line.split(" ")
            y = int(seq.pop()) if HRE else [int(i) for i in seq[len(seq) // 2:]]
            x = [i.split(":") for i in (seq if HRE else seq[:len(seq) // 2])]
            xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
            data.append(xc = xc, xw = xw, y0 = y)
        elif HRE: # empty line as document delimiter
            data.y0.append([])
    for xc, xw, y0, y0_lens in data.batchiter():
        xc, xw = data.batchify(xc, xw, doc_lens = y0_lens)
        _, y0 = data.batchify(None, y0, sos = True)
        data.batch.append((xc, xw, y0))
    fo.close()
    print("data size: %d" % (len(data.batch) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data.batch, cti, wti, itt

def train():
    num_epochs = int(sys.argv[-1])
    data, cti, wti, itt = load_data()
    model = rnn_crf(len(cti), len(wti), len(itt))
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time()
        for xc, xw, y0 in data:
            loss = model(xc, xw, y0) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, cti, wti, itt]
            evaluate(predict(sys.argv[6], *args), True)
            model.train()
            print()

if __name__ == "__main__":
    if len(sys.argv) not in [7, 8]:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx training_data (validation_data) num_epoch" % sys.argv[0])
    if len(sys.argv) == 7:
        EVAL_EVERY = False
    train()
