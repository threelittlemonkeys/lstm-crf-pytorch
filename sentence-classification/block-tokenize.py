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
    blocks = []
    for doc in data:
        i = 0
        doc = doc.split("\n")
        while i <= len(doc) - z:
            blocks.append("\n".join(doc[i:i + z]))
            i += z
        if i < len(doc):
            blocks.append("\n".join(doc[i:]))
    fo.write("\n\n".join(blocks))
    fi.close()
    fo.close()
