import sys
import re

for line in sys.stdin:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        if tag == "RP":
            print(word, tag)
        continue
        if re.search("'s", word):
            if tag.count("+") > 0:
                tags = tag.split("+")
                print(word[:-2], tags[0])
                print(word[-2:], tags[1])
            else:
                if tag[-1] != "$":
                    print(word, tag)
                else:
                    print(word[:-2], tag[:-1])
                    print(word[-2:], "POS")
