from model import *
from utils import *
from evaluate import *
from dataloader import *

def load_data(args):
    data = dataloader()
    batch = []
    cti = load_tkn_to_idx(args[1]) # char_to_idx
    wti = load_tkn_to_idx(args[2]) # word_to_idx
    itt = load_idx_to_tkn(args[3]) # idx_to_tkn
    print("loading %s..." % args[4])
    with open(args[4]) as fo:
        text = fo.read().strip().split("\n" * (HRE + 1))
    for block in text:
        data.append_row()
        for line in block.split("\n"):
            x, y = line.split("\t")
            x = [x.split(":") for x in x.split(" ")]
            y = tuple(map(int, y.split(" ")))
            xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
            data.append_item(xc = xc, xw = xw, y0 = y)
    for _batch in data.split():
        xc, xw, y0, lens = _batch.xc, _batch.xw, _batch.y0, _batch.lens
        xc, xw = data.tensor(bc = xc, bw = xw, lens = lens)
        _, y0 = data.tensor(bw = y0, sos = True)
        batch.append((xc, xw, y0))
    print("data size: %d" % len(data.y0))
    print("batch size: %d" % BATCH_SIZE)
    return batch, cti, wti, itt

def train(args):
    num_epochs = int(args[-1])
    batch, cti, wti, itt = load_data(args)
    model = rnn_crf(len(cti), len(wti), len(itt))
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint(args[0], model) if isfile(args[0]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", args[0])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time()
        for xc, xw, y0 in batch:
            loss = model(xc, xw, y0) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(batch)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if len(args) == 7 and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            evaluate(predict(model, cti, wti, itt, args[5]), True)
            model.train()
            print()

if __name__ == "__main__":
    if len(sys.argv) not in [7, 8]:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx training_data (validation_data) num_epoch" % sys.argv[0])
    train(sys.argv[1:])
