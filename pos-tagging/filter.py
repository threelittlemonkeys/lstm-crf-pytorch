import sys
import re

pl = {}
fo = open("brown.tagged.merged.uniq")
for line in fo:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        word = word.lower()
        if sys.argv[1] == "word":
            x = word
        elif sys.argv[1] == "tag":
            x = tag
        else:
            break
        if len(sys.argv) == 2:
           if x in pl:
               continue
           pl[x] = True
           print(x)
        elif x == sys.argv[2]:
           if (word, tag) in pl:
               continue
           pl[word, tag] = True
           print(word, tag)
fo.close()
