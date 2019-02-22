from predict import *
from collections import defaultdict

def evaluate(result):
    a = defaultdict(float) # average
    s = defaultdict(int) # entire set
    p = defaultdict(int) # positive
    t = defaultdict(int) # true positive
    for _, y0, y1 in result: # actual value, predicted outcome
        for y0, y1 in zip(y0, y1):
            s[y0] += 1
            p[y1] += 1
            if y0 == y1:
                t[y0] += 1
    for y in sorted(s.keys()):
        pr = t[y] / p[y] if p[y] else 0
        rc = t[y] / s[y]
        a["macro_pr"] += pr
        a["macro_rc"] += rc
        print("\nlabel = %s" % y)
        print("precision = %f (%d/%d)" % (pr, t[y], p[y]))
        print("recall = %f (%d/%d)" % (rc, t[y], s[y]))
        print("f1 = %f" % f1(pr, rc))
    a["macro_pr"] /= len(s)
    a["macro_rc"] /= len(s)
    a["micro_pr"] = sum(t.values()) / sum(p.values())
    a["micro_rc"] = sum(t.values()) / sum(s.values())
    print()
    print("macro precision = %f" % a["macro_pr"])
    print("macro recall = %f" % a["macro_rc"])
    print("macro f1 = %f" % f1(a["macro_pr"], a["macro_rc"]))
    print()
    print("micro precision = %f" % a["micro_pr"])
    print("micro recall = %f" % a["micro_rc"])
    print("micro f1 = %f" % f1(a["micro_pr"], a["micro_rc"]))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        evaluate(predict(True))
