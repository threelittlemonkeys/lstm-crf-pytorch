from predict import *
from collections import defaultdict

def evaluate(result):
    a = defaultdict(float) # average
    tp = defaultdict(int) # true positives
    tpfn = defaultdict(int) # true positives + false negatives
    tpfp = defaultdict(int) # true positives + false positives
    for _, y0, y1 in result: # actual value, prediction
        for y0, y1 in zip(y0, y1):
            tp[y0] += y0 == y1
            tpfn[y0] += 1
            tpfp[y1] += 1
    for y in sorted(tpfn.keys()):
        pr = tp[y] / tpfp[y] if tpfp[y] else 0
        rc = tp[y] / tpfn[y] if tpfn[y] else 0
        a["macro_pr"] += pr
        a["macro_rc"] += rc
        print()
        print("label = %s" % y)
        print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
        print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
        print("f1 = %f" % f1(pr, rc))
    a["macro_pr"] /= len(tpfn)
    a["macro_rc"] /= len(tpfn)
    a["micro_pr"] = sum(tp.values()) / sum(tpfp.values())
    a["micro_rc"] = sum(tp.values()) / sum(tpfn.values())
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
