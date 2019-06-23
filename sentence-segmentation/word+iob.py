import sys

if __name__ == "__main__": # tag every word in IOB2 format for sentence segmentation
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    fi = open(sys.argv[1])
    fo = open(sys.argv[1] + ".word+iob", "w")
    data = fi.read()
    data = data.strip()
    data = data.split("\n\n")
    for line in data:
        line = line.split("\n")
        line = [x.split(" ") for x in line]
        line = [[x[0] + "/B"] + [w + "/I" for w in x[1:]] for x in line]
        fo.write(" ".join(" ".join(x) for x in line) + "\n")
    fi.close()
    fo.close()
