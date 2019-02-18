import sys
import re

fin = open("brown.tagged.merged.uniq", "r")
fout = open("brown.tagged.merged.uniq.ptb", "w")
for line in fin:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        tags = []
        for x in tag.split("+"):
            neg = False
            pos = False
            if x[:3] == "FW-":
                tags = ["FW"]
                break
            if x[-3:] == "-NC": x = x[:-3]
            if x[-3:] == "-HL": x = x[:-3]
            if x[-3:] == "-TL": x = x[:-3]
            if len(x) and x[-1] == "*":
                x = x[:-1]
                neg = True
            if re.match("[PW]P\$$", x): tags.append("DT")
            if x == "PP$$": tags.append("PN")
            if len(x) and x[-1] == "$":
                x = x[:-1]
                pos = True
            if re.match("[^A-Z]+$", x): tags.append(x) # other special characters
            if re.match("A(B[LNX]|[PT])$", x): tags.append("DT")
            if re.match("BE(DZ?|[GMNRZ])?$", x): tags.append("VB")
            if x == "CC": tags.append("CC")
            if x == "CS": tags.append("CS")
            if x == "CD": tags.append("JJ")
            if x == "OD": tags.append("JJ")
            if re.match("DO[DZ]?$", x): tags.append("VB")
            if re.match("DT[ISX]?$", x): tags.append("DT")
            if x == "EX": tags.append("RB")
            if re.match("HV[DGNZ]?$", x): tags.append("VB")
            if x == "IN": tags.append("IN")
            if re.match("JJ[RST]?$", x): tags.append("JJ")
            if x == "MD": tags.append("MD")
            if x == "NIL": tags.append("X")
            if re.match("N([NPR]S?)$", x): tags.append("NN")
            if re.match("P(N|P[LOS]S?)$", x): tags.append("PN")
            if re.match("QLP?$", x): tags.append("RB")
            if re.match("RB[RT]?$", x): tags.append("RB")
            if x == "RN": tags.append("RB")
            if x == "RP": tags.append("RP")
            if x == "TO": tags.append("RP")
            if x == "UH": tags.append("UH")
            if re.match("VB[DGNPZ]?$", x): tags.append("VB")
            if x == "WDT": tags.append("DT")
            if re.match("WP[OS]?$", x): tags.append("PN")
            if re.match("W(QL|RB)$", x): tags.append("RB")
            if neg:
                tags.append("RP")
            if pos:
                tags.append("POS")
        out.append(word + "/" + "+".join(tags))
    fout.write(" ".join(out) + "\n")
fin.close()
fout.close()
