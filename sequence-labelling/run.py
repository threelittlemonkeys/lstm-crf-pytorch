import sys
import re

for line in sys.stdin:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        # tag = re.sub("AT", "DT", tag)
        # out.append(word + "/" + tag)
        # continue
        if re.search("'s", word):
            print(word, tag)
    # print(" ".join(out))
