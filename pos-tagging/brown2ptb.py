import sys
import re

def convert(tkn):
    out = []
    for word, tag in tkn:
        tags = []
        for x in tag.split("+"):
            neg = False
            pos = False
            if x[:3] == "FW-": # foreign word
                tags = ["FW"]
                break
            if x[-3:] == "-NC": x = x[:-3]
            if x[-3:] == "-HL": x = x[:-3]
            if x[-3:] == "-TL": x = x[:-3]
            if len(x) and x[-1] == "*":
                x = x[:-1]
                neg = True
            if len(x) and x[-1] == "$":
                if x == "PP$": tags.append("DT") # possessive pronoun
                elif x == "PP$$": tags.append("PRO") # possessive pronoun
                elif x == "WP$": tags.append("WH") # possessive wh-pronoun
                else:
                    x = x[:-1]
                    pos = True
            if re.match("[^A-Z]+$", x): tags.append(x) # other special characters
            if x == "ABL": tags.append("RB")
            if re.match("A(B[NX]|[PT])$", x): tags.append("DT")
            if x == "BE": tags.append("VB")
            if re.match("BEDZ?$", x): tags.append("VB")
            if re.match("BE[GN]$", x): tags.append("VB")
            if re.match("BE[MRZ]$", x): tags.append("VB")
            if x == "CC": tags.append("CC")
            if x == "CS": tags.append("CC")
            if x == "CD": tags.append("CD") # cardinal numeral
            if x == "OD": tags.append("JJ") # ordinal numeral
            if re.match("DO[DZ]?$", x): tags.append("VB")
            if re.match("DT[ISX]?$", x): tags.append("DT")
            if x == "EX": tags.append("RB")
            if x == "HV": tags.append("VB")
            if re.match("HV[DGNZ]$", x): tags.append("VB")
            if x == "IN": tags.append("IN")
            if re.match("JJ[RST]?$", x): tags.append("JJ")
            if x == "MD": tags.append("AUX")
            if x == "NIL": tags.append("UNK")
            if re.match("NNS?$", x): tags.append("NN") # noun
            if re.match("NPS?$", x): tags.append("NN") # proper noun
            if re.match("NRS?$", x): tags.append("NN") # adverbial noun
            if re.match("P(N|P[LOS]S?)$", x): tags.append("PRO")
            if re.match("QLP?$", x): tags.append("RB")
            if re.match("RB[RT]?$", x): tags.append("RB")
            if x == "RN": tags.append("RB") # nominal adverb
            if x == "RP": tags.append("RP") # particle
            if x == "TO": tags.append("RP")
            if x == "UH": tags.append("UH") # interjection
            if re.match("VB[DGNZ]?$", x): tags.append("VB")
            if x == "WDT": tags.append("WH")
            if re.match("WP[OS]$", x): tags.append("WH")
            if re.match("W(QL|RB)$", x): tags.append("WH")
            if neg:
                tags.append("NEG")
            if pos:
                tags.append("POS")
        out.append(word + "/" + "+".join(tags))
    return out

if __name__ == "__main__":
    fin = open("brown.tagged.merged.uniq", "r")
    fout = open("brown.tagged.merged.uniq.ptb", "w")
    for line in fin:
        line = line.strip()
        tkn = [re.split("/(?=[^/]+$)", x) for x in line.split()]
        tkn = convert(tkn)
        fout.write(" ".join(tkn) + "\n")
    fin.close()
    fout.close()
