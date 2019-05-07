import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    with open(sys.argv[1]) as fo:
        data = fo.read()
    with open(sys.argv[1] + ".iob", "w") as fo:
        data = data.strip()
        data = data.split("\n\n")
        for sents in data:
            sents = [sent.split(" ") for sent in sents.split("\n")]
            sents = [[sent[0] + "/B"] + [w + "/I" for w in sent[1:]] for sent in sents]
            fo.write(" ".join([" ".join(sent) for sent in sents]) + "\n")
