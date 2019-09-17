import sys

if __name__ == "__main__": # tokenize documents into blocks
    if len(sys.argv) != 3:
        sys.exit("Usage: %s block_size training_data" % sys.argv[0])
    z = int(sys.argv[1])
    fi = open(sys.argv[2])
    fo = open(sys.argv[2] + ".tokenized", "w")
    data = fi.read()
    data = data.strip()
    data = data.split("\n\n")
    blocks = []
    for doc in data:
        doc = doc.split("\n")
        for i in range(0, len(doc) - z + 1, z):
            blocks.append("\n".join(doc[i:i + z]))
        if i + z < len(doc):
            blocks.append("\n".join(doc[i + z:]))
    fo.write("\n\n".join(blocks))
    fi.close()
    fo.close()
