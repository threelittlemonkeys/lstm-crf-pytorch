import sys
import random

if __name__ == "__main__": # tokenize documents into blocks
    if len(sys.argv) != 3:
        sys.exit("Usage: %s sizes data" % sys.argv[0])
    fi = open(sys.argv[2])
    fo = open(sys.argv[2] + ".blocks", "w")
    data = fi.read().strip().split("\n\n")
    sizes = list(map(int, sys.argv[1].split(",")))
    blocks = dict()
    for doc in data:
        doc = doc.split("\n")
        for i in range(len(doc)):
            for z in sizes:
                blocks["\n".join(doc[i:i + z])] = True
                # duplicate blocks are removed
    blocks = list(blocks.keys())
    random.shuffle(blocks)
    fo.write("\n\n".join(blocks) + "\n")
    fi.close()
    fo.close()
