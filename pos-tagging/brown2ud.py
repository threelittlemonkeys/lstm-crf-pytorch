import sys
import re

fo = open("brown.tagged.merged.uniq")
for line in fo:
    line = line.strip()
    tkn = line.split()
    out = []
    for word, tag in [re.split("/(?=[^/]+$)", x) for x in tkn]:
        if tag == "AP": tag = "JJ"
        if tag == "BE": tag = "VB"
        if tag == "BED": tag = "VBD"
        if tag == "BEG": tag = "VBG"
        if tag == "BEM": tag = "VBP"
        if tag == "BEN": tag = "VBN"
        if tag == "BER": tag = "VBP"
        if tag == "BEZ": tag = "VBZ"
        if tag == "HV": tag = "VB"
        if tag == "HVD": tag = "VBD"
        if tag == "HVG": tag = "VBG"
        if tag == "HVN": tag = "VBN"
        out.append(word + "/" + tag)
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
ABN  pre-quantifier (half, all)
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
CD   cardinal numeral (one, two, 2, etc.)
CS   subordinating conjunction (if, although)
DO   do
DOD  did
DOZ  does
DT   singular determiner/quantifier (this, that)
DTI  singular or plural determiner/quantifier (some, any)
DTS  plural determiner (these, those)
DTX  determiner/double conjunction (either)
EX   existential there
FW   foreign word (hyphenated before regular tag)
HV   have
HVD  had (past tense)
HVG  having
HVN  had (past participle)
IN   preposition
JJ   adjective
JJR  comparative adjective
JJS  semantically superlative adjective (chief, top)
JJT  morphologically superlative adjective (biggest)
MD   modal auxiliary (can, should, will)
NC   cited word (hyphenated after regular tag)
NN   singular or mass noun
NN$  possessive singular noun
NNS  plural noun
NNS$ possessive plural noun
NP   proper noun or part of name phrase
NP$  possessive proper noun
NPS  plural proper noun
NPS$ possessive plural proper noun
NR   adverbial noun (home, today, west)
OD   ordinal numeral (first, 2nd)
PN   nominal pronoun (everybody, nothing)
PN$  possessive nominal pronoun
PP$  possessive personal pronoun (my, our)
PP$$ second (nominal) possessive pronoun (mine, ours)
PPL  reflexive personal pronoun, singular (myself)
PPLS reflexive personal pronoun, plural (ourselves)
PPO  objective personal pronoun (me, him, it, them)
PPS  nominative personal pronoun, 3rd person singular (he, she, it, one)
PPSS nominative personal pronoun, other (I, we, they, you)
PRP  personal pronoun
PRP$ possessive pronoun
QL   qualifier (very, fairly)
QLP  post-qualifier (enough, indeed)
RB   adverb
RBR  comparative adverb
RBT  superlative adverb
RN   nominal adverb (here, then, indoors)
RP   adverbial particle (about, off, up)
TO   infinitive marker to
UH   interjection, exclamation
VB   verb, base form
VBD  verb, past tense
VBG  verb, present participle/gerund
VBN  verb, past participle
VBP  verb, non 3rd person singular present
VBZ  verb, 3rd person singular present
WDT  wh-determiner (what, which)
WP$  possessive wh-pronoun (whose)
WPO  objective wh-pronoun (whom, which, that)
WPS  nominative wh-pronoun (who, which, that)
WQL  wh-qualifier (how)
WRB  wh-adverb (how, where, when)

====================
Universal POS tagset
====================

ADJ   adjective
ADP   adposition
ADV   adverb
AUX   auxiliary
CCONJ coordinating conjunction
DET   determiner
INTJ  interjection
NOUN  noun
NUM   numeral
PART  particle
PRON  pronoun
PROPN proper noun
PUNCT punctuation
SCONJ subordinating conjunction
SYM   symbol
VERB  verb
X     other
'''
