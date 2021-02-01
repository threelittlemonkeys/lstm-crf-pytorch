import sys

if __name__ == "__main__": # IOB2 tagging for sentence segmentation
    if len(sys.argv) != 2:
        sys.exit("Usage: %s data" % sys.argv[0])
    fi = open(sys.argv[1])
    fo = open(sys.argv[1] + ".IOB", "w")
    data = fi.read()
    data = data.strip()
    data = data.split("\n\n")
    for sents in data:
        sents = sents.split("\n")
        sents = [x.split(" ") for x in sents]
        sents = [[x[0] + "/B"] + [w + "/I" for w in x[1:]] for x in sents]
        fo.write(" ".join(" ".join(x) for x in sents) + "\n")
    fi.close()
    fo.close()
