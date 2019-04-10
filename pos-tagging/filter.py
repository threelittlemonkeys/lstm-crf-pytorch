import re
import sys
from collections import defaultdict


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: %s brown|ptb word|tag|word/tag" % sys.argv[0])

    pl = defaultdict(int)

    if sys.argv[1] == "brown":
        fo = open("brown.tagged.merged.uniq")
    if sys.argv[1] == "ptb":
        fo = open("brown.tagged.merged.uniq.ptb")

    for line in fo:
        line = line.strip()
        tkn = [re.split("/(?=[^/]+$)", x) for x in line.split()]
        for i, (word, tag) in enumerate(tkn):
            word = word.lower()
            tag = tag.upper()
            if len(sys.argv) == 2:
                pl[tag] += 1
            elif word == sys.argv[2] or tag == sys.argv[2]:
                pl[word + " " + tag] += 1
            elif sys.argv[2] == word + "/" + tag:
                out = tkn[max(0, i - 2):i]
                out += [tkn[i]]
                out += tkn[i + 1:min(len(tkn), i + 3)]
                print(" ".join(["/".join(x) for x in out]))
    fo.close()

    for k, v in sorted(pl.items(), key=lambda x: -x[1]):
        print(k, v)

    print("%d in total" % len(pl))


if __name__ == '__main__':
    main()
