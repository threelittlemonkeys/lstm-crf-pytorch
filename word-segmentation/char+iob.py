import sys

if __name__ == "__main__": # BI(ES) tagging for character-based word segmentation
    if len(sys.argv) != 3 or sys.argv[1] not in ("BI", "BIES"):
        sys.exit("Usage: %s BI|BIES training_data" % sys.argv[0])
    tagset = sys.argv[1]
    fi = open(sys.argv[2])
    fo = open(sys.argv[2] + "." + tagset, "w")
    for line in fi:
        line = line.strip()
        line = line.split(" ")
        if tagset  == "BI":
            line = [[w[0] + "/B"] + [c + "/I" for c in w[1:]] for w in line]
        else: # BIES
            line = [
                [w + "/S"] if len(w) == 1
                else [w[0] + "/B"] + [c + "/I" for c in w[1:-1]] + [w[-1] + "/E"]
                for w in line
            ]
        fo.write(" ".join(" ".join(w) for w in line) + "\n")
    fi.close()
    fo.close()
