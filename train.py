import torch
import time
import re

from argparse import ArgumentParser
from os.path import isfile

from evaluate import evaluate
from predict import predict
from model import RNN_CRF as rnn_crf
from utils import load_tkn_to_idx, load_idx_to_tkn, batchify, load_checkpoint, save_checkpoint
from parameters import BATCH_SIZE, EVAL_EVERY, SAVE_EVERY, LEARNING_RATE, CUDA


def load_data(args):
    data = []
    char_seq_batch = []  # character sequence batch
    word_seq_batch = []  # word sequence batch
    label_batch = []  # label batch
    cti = load_tkn_to_idx(args.char_to_idx)  # char_to_idx
    wti = load_tkn_to_idx(args.word_to_idx)  # word_to_idx
    itt = load_idx_to_tkn(args.idx_to_tag)  # idx_to_tkn
    print("loading {}".format(args.training_data))
    with open(args.training_data, "r") as infile:
        for line in infile:
            seq = line.strip().split(" ")
            x = [x.split(":") for x in seq[:len(seq) // 2]]
            y = [int(x) for x in seq[len(seq) // 2:]]
            xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
            char_seq_batch.append(xc)
            word_seq_batch.append(xw)
            label_batch.append(y)
            if len(label_batch) == BATCH_SIZE:
                char_seq_batch, word_seq_batch = batchify(char_seq_batch, word_seq_batch, sos=False, eos=False)
                _, label_batch = batchify(None, label_batch, eos=False)
                data.append((char_seq_batch, word_seq_batch, label_batch))
                char_seq_batch = []
                word_seq_batch = []
                label_batch = []
    print("data size: {:d}".format(len(data) * BATCH_SIZE))
    print("batch size: {:d}".format(BATCH_SIZE))
    return data, cti, wti, itt


def train(args):
    num_epochs = args.num_epochs
    data, cti, wti, itt = load_data(args)
    model = rnn_crf(len(cti), len(wti), len(itt))
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch = load_checkpoint(args.model, model) if isfile(args.model) else 0
    filename = re.sub("\.epoch[0-9]+$", "", args.model)
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time.time()
        for xc, xw, y in data:
            model.zero_grad()
            loss = torch.mean(model(xc, xw, y))  # forward pass and compute loss
            loss.backward()  # compute gradients
            optim.step()  # update parameters
            loss = loss.item()
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs) and args.validation_data:
            evaluate(predict(args.validation_data, model, cti, wti, itt), True)
            model.train()
            print()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("char_to_idx")
    parser.add_argument("word_to_idx")
    parser.add_argument("idx_to_tag")
    parser.add_argument("training_data")
    parser.add_argument("validation_data", nargs="?")
    parser.add_argument("num_epochs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    print("cuda: %s" % CUDA)
    train(args)


if __name__ == '__main__':
    main()
