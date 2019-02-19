import sys
import re

if len(sys.argv) not in [2, 3]:
    sys.exit("Usage: %s brown|ptb word|tag|word/tag" % sys.argv[0])

pl = {}

if sys.argv[1] == "brown":
    fo = open("brown.tagged.merged.uniq")
if sys.argv[1] == "ptb":
    fo = open("brown.tagged.merged.uniq.ptb")

for line in fo:
    line = line.strip()
    tkn = [re.split("/(?=[^/]+$)", x) for x in line.split()]
    for i, (word, tag) in enumerate(tkn):
        word = word.lower()
        # tag = tag.upper()
        if len(sys.argv) == 2:
            if tag in pl:
                continue
            pl[tag] = True
            print(tag)
        elif word == sys.argv[2] or tag == sys.argv[2]:
            if (word, tag) in pl:
                continue
            pl[word, tag] = True
            print(word, tag)
        elif sys.argv[2] == word + "/" + tag:
            out = tkn[max(0, i - 2):i]
            out += [tkn[i]]
            out += tkn[i + 1:min(len(tkn), i + 3)]
            print(" ".join(["/".join(x) for x in out]))
fo.close()

print("%d in total" % len(pl))
