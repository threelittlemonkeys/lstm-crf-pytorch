from predict import *

def evaluate(result, summary = False):
    avg = defaultdict(float) # average
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
        avg["macro_pr"] += pr
        avg["macro_rc"] += rc
        if not summary:
            print("label = %s" % y)
            print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
            print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
            print("f1 = %f" % f1(pr, rc))
            print()
    avg["macro_pr"] /= len(tpfn)
    avg["macro_rc"] /= len(tpfn)
    avg["micro_f1"] = sum(tp.values()) / sum(tpfp.values())
    print("macro precision = %f" % avg["macro_pr"])
    print("macro recall = %f" % avg["macro_rc"])
    print("macro f1 = %f" % f1(avg["macro_pr"], avg["macro_rc"]))
    print("micro f1 = %f" % avg["micro_f1"])

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    evaluate(predict(sys.argv[5], *load_model()))
