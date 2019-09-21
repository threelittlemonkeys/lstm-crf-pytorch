import sys

if __name__ == "__main__": # tokenize documents into blocks
    if len(sys.argv) != 3:
        sys.exit("Usage: %s block_size data" % sys.argv[0])
    z = int(sys.argv[1])
    fi = open(sys.argv[2])
    fo = open(sys.argv[2] + ".block", "w")
    data = fi.read()
    data = data.strip()
    data = data.split("\n\n")
    for doc in data:
        doc = doc.split("\n")
        for i in range(len(doc)):
            fo.write("\n".join(doc[i:i + z]))
            fo.write("\n" if i == len(doc) - 1 else "\n\n")
    fi.close()
    fo.close()
