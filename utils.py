import re
from model import *

def normalize_word(s):
    s = re.sub("[" + chr(0x3040) + "-" + chr(0x30FF) + "]+", chr(0x3042), s) # convert Hiragana and Katakana to あ
    s = re.sub("[" + chr(0x4E00) + "-" + chr(0x9FFF) + "]+", chr(0x6F22), s) # convert CJK unified ideographs to 漢
    return s

def load_tag_to_idx(filename):
    print("loading tag_to_idx...")
    tag_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        tag_to_idx[line] = len(tag_to_idx)
    fo.close()
    return tag_to_idx

def load_word_to_idx(filename):
    print("loading word_to_idx...")
    word_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        word_to_idx[line] = len(word_to_idx)
    fo.close()
    return word_to_idx

def load_checkpoint(filename, model = None, g2c = False):
    print("loading model...")
    if g2c: # load weights into CPU
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss):
    print("saving model...")
    checkpoint = {}
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["loss"] = loss
    torch.save(checkpoint, filename + ".epoch%d" % epoch)
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))

def gpu2cpu(filename):
    checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    torch.save(checkpoint, filename + ".cpu")
