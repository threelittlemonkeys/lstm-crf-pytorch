import sys
import re

pl = {}
fo = open("brown.tagged.merged.uniq")
for line in fo:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        if len(sys.argv) == 1:
            if tag in pl:
                continue
            pl[tag] = True
            print(tag)
        elif tag == sys.argv[1]:
            if (word, tag) in pl:
                continue
            pl[word, tag] = True
            print(word, tag)
fo.close()
