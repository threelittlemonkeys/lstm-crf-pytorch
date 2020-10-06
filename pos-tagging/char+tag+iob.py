import sys
import re

if __name__ == "__main__": # tag every character/tag in IOB2 format for POS tagging
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    fi = open(sys.argv[1])
    fo = open(sys.argv[1] + ".char+tag+iob", "w")
    for line in fi:
        line = line.strip().split(" ")
        out = []
        for token in line:
            word, tag = re.split("/(?=[^/]+$)", token)
            out.extend(["%s/%s-%s" % (x, tag, "I" if i else "B") for i, x in enumerate(word)])
        fo.write(" ".join(out) + "\n")
    fi.close()
    fo.close()
