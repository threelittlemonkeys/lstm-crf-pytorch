import sys
import re

pl = {}
fo = open("brown.tagged.merged.uniq.converted")
for line in fo:
    line = line.strip()
    tkn = [re.split("/(?=[^/]+$)", x) for x in line.split()]
    for i, (word, tag) in enumerate(tkn):
        word = word.lower()
        # tag = tag.upper()
        if len(sys.argv) == 1:
            if tag in pl:
                continue
            pl[tag] = True
            print(tag)
        elif word == sys.argv[1] or tag == sys.argv[1]:
            if (word, tag) in pl:
                continue
            pl[word, tag] = True
            print(word, tag)
        elif sys.argv[1] == word + "/" + tag:
            out = tkn[max(0, i - 2):i]
            out += [tkn[i]]
            out += tkn[i + 1:min(len(tkn), i + 3)]
            print(" ".join(["/".join(x) for x in out]))
fo.close()
