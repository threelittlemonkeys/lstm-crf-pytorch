import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    fo = open(sys.argv[1])
    data = fo.read()
    data = data.strip()
    data = data.split("\n\n")
    for sents in data:
        sents = [sent.split(" ") for sent in sents.split("\n")]
        sents = [[sent[0] + "/B"] + [w + "/I" for w in sent[1:]] for sent in sents]
        print(" ".join([" ".join(sent) for sent in sents]))
    fo.close()
