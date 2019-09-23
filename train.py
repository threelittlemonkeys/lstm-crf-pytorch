from model import *
from utils import *
from evaluate import *

def load_data():
    bxc = [] # character sequence batch
    bxw = [] # word sequence batch
    by = [[]] if HRE else [] # label batch
    data = []
    doc_lens = [] # document lengths for HRE
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
            bxc.append(xc)
            bxw.append(xw)
            (by[-1] if HRE else by).append(y)
        elif HRE: # empty line as document delimiter
            doc_lens.append(len(by[-1]))
            by.append([])
        if len(by) == BATCH_SIZE + HRE:
            bxc, bxw = batchify(bxc, bxw, doc_lens = doc_lens)
            _, by = batchify(None, by[:len(by) - HRE], sos = True)
            data.append((bxc, bxw, by))
            bxc = []
            bxw = []
            by = [[]] if HRE else []
            doc_lens = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, cti, wti, itt

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
        for xc, xw, y in data:
            loss = model(xc, xw, y) # forward pass and compute loss
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
