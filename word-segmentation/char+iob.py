import sys

if __name__ == "__main__": # tag every character in IOB2 format for word segmentation
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    fi = open(sys.argv[1])
    fo = open(sys.argv[1] + ".char+iob", "w")
    for line in fi:
        line = line.strip()
        line = line.split(" ")
        line = [[w[0] + "/B"] + [c + "/I" for c in w[1:]] for w in line]
        fo.write(" ".join(" ".join(w) for w in line) + "\n")
    fi.close()
    fo.close()
