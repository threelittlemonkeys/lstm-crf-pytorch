import sys
import re

fo = open("brown.tagged.merged.uniq")
for line in fo:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        tags = []
        for x in tag.split("+"):
            neg = False
            poss = False
            if x[:3] == "FW-": x = x[3:]
            if x[-3:] == "-NC": x = x[:-3]
            if x[-3:] == "-HL": x = x[:-3]
            if x[-3:] == "-TL": x = x[:-3]
            if len(x) and x[-1] == "*":
                x = x[:-1]
                neg = True
            if re.match("[PW]P\$$", x): tags.append("DET")
            if re.match("PP\$\$$", x): tags.append("PRON")
            if len(x) and x[-1] == "$":
                x = x[:-1]
                poss = True
            if re.match("[^A-Z]+$", x): tags.append(x) # other special characters
            if re.match("A(B[LNX]|[PT])$", x): tags.append("DET")
            if re.match("BE[DGMNRZ]*$", x): tags.append("VERB")
            if re.match("C[CS]$", x): tags.append("CONJ")
            if re.match("[CO]D$", x): tags.append("NUM")
            if re.match("DO[DZ]?$", x): tags.append("VERB")
            if re.match("DT[ISX]?$", x): tags.append("DET")
            if re.match("EX$", x): tags.append("ADV")
            if re.match("HV[DGNZ]?$", x): tags.append("VERB")
            if re.match("IN$", x): tags.append("ADP")
            if re.match("JJ[RST]?$", x): tags.append("ADJ")
            if re.match("MD$", x): tags.append("AUX")
            if re.match("NIL$", x): tags.append("UNK")
            if re.match("N([NPR]S?)$", x): tags.append("NOUN")
            if re.match("P(N|P[LOS]S?)$", x): tags.append("PRON")
            if re.match("QLP?$", x): tags.append("ADV")
            if re.match("RB[RT]?$", x): tags.append("ADV")
            if re.match("RN$", x): tags.append("ADV")
            if re.match("RP$", x): tags.append("PART")
            if re.match("TO$", x): tags.append("PART")
            if re.match("UH$", x): tags.append("INTJ")
            if re.match("VB[DGNPZ]?$", x): tags.append("VERB")
            if re.match("WDT$", x): tags.append("DET")
            if re.match("WP[OS]?$", x): tags.append("PRON")
            if re.match("W(QL|RB)$", x): tags.append("ADV")
            if neg:
                tags.append("PART")
            if poss:
                tags.append("POSS")
        out.append(word + "/" + "+".join(tags))
    print(" ".join(out))
fo.close()

'''
===================
Brown Corpus Tagset
===================

.    sentence (. ; ? *)
(    left parenthesis
)    right parenthesis
*    not, n't
--   dash
,    comma
:    colon
ABL  pre-qualifier (quite, rather)
ABN  pre-quantifier (all, half, many)
ABX  pre-quantifier (both)
AP   post-determiner (many, several, next)
AT   article (a, the, no)
BE   be
BED  were
BEDZ was
BEG  being
BEM  am
BEN  been
BER  are, art
BEZ  is
CC   coordinating conjunction (and, or)
CD   cardinal numeral (one, two, 3, etc.)
CS   subordinating conjunction (if, although)
DO   do
DOD  did
DOZ  does
DT   determiner/quantifier, sg (this, that)
DTI  determiner/quantifier, sg or pl (some, any)
DTS  determiner, pl (these, those)
DTX  determiner/double conjunction (either)
EX   existential there
FW   foreign word (hyphenated before regular tag)
HL   headline (hyphenated after regular tag)
HV   have
HVD  had (past tense)
HVG  having
HVN  had (past participle)
HVZ  has
IN   preposition
JJ   adjective
JJR  comparative adjective
JJS  semantically superlative adjective (chief, top)
JJT  morphologically superlative adjective (biggest)
MD   modal auxiliary (can, should, will)
NC   cited word (hyphenated after regular tag)
NIL  no category assigned
NN   noun, sg or mass
NNS  noun, pl
NP   proper noun, sg or part of name phrase
NPS  proper noun, pl
NR   adverbial noun, sg (home, today, west)
NRS  adverbial noun, pl (home, today, west)
OD   ordinal numeral (first, 2nd)
PN   nominal pronoun (everybody, nothing)
PP$  possessive pronoun (my, our)
PP$$ possessive pronoun, nominal (mine, ours)
PPL  reflexive pronoun, sg (myself)
PPLS reflexive pronoun, pl (ourselves)
PPO  objective pronoun (me, him, it, them)
PPS  nominative pronoun, 3rd sg (he, she, it, one)
PPSS nominative pronoun, other (I, we, they, you)
QL   qualifier (very, fairly)
QLP  post-qualifier (enough, indeed)
RB   adverb
RBR  comparative adverb
RBT  superlative adverb
RN   nominal adverb (here, then, indoors)
RP   adverbial particle (about, off, up)
TL   title (hyphenated after regular tag)
TO   infinitive marker to
UH   interjection, exclamation
VB   verb, base form
VBD  verb, past tense
VBG  verb, present participle/gerund
VBN  verb, past participle
VBP  verb, non 3rd sg present
VBZ  verb, 3rd sg present
WDT  wh-determiner (what, which)
WP$  wh-pronoun, poss (whose)
WPO  wh-pronoun, obj (whom, which, that)
WPS  wh-pronoun, nom (who, which, that)
WQL  wh-qualifier (how)
WRB  wh-adverb (how, where, when)

====================
Universal POS tagset
====================

ADJ  ADJ   adjective
ADP  ADP   adposition
ADV  ADV   adverb
AUX  AUX   auxiliary
CONJ CCONJ coordinating conjunction
DET  DET   determiner
INTJ INTJ  interjection
NOUN NOUN  noun
NUM  NUM   numeral
PART PART  particle
PRON PRON  pronoun
CONJ PROPN proper noun
PUNC PUNCT punctuation
CONJ SCONJ subordinating conjunction
SYM  SYM   symbol
VERB VERB  verb
UNK  X     other
'''
