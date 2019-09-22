import sys

if __name__ == "__main__": # tokenize documents into blocks
    if len(sys.argv) != 3:
        sys.exit("Usage: %s block_size data" % sys.argv[0])
    z = int(sys.argv[1])
    fi = open(sys.argv[2])
    fo = open(sys.argv[2] + ".block", "w")
    data = fi.read().strip().split("\n\n")
    blocks = []
    for doc in data:
        doc = doc.split("\n")
        for i in range(len(doc)):
            blocks.append("\n".join(doc[i:i + z]))
    fo.write("\n\n".join(blocks) + "\n")
    fi.close()
    fo.close()
