import sys
import re

fo = open("brown.tagged.merged.uniq")
for line in fo:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        if len(sys.argv) == 1:
            print(tag)
        elif tag == sys.argv[1]:
            print(word, tag)
        continue
        if re.search("'s", word):
            if tag.count("+") == 2:
                tags = tag.split("+")
                print(word[:-2], tags[0], end = " ")
                print(word[-2:], tags[1])
            else:
                if tag[-1] != "$":
                    print(word, tag)
                else:
                    print(word[:-2], tag[:-1], end = " ")
                    print(word[-2:], "POS")
fo.close()
