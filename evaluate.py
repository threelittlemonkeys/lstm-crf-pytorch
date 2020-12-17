from predict import *

def evaluate(result, summary = False):
    result = list(result)

    avg = defaultdict(float) # average
    tp = defaultdict(int) # true positives
    tpfn = defaultdict(int) # true positives + false negatives
    tpfp = defaultdict(int) # true positives + false positives
    for _, y0, y1 in result: # actual value, prediction
        if HRE:
            tp[y0] += (y0 == y1)
            tpfn[y0] += 1
            tpfp[y1] += 1
            continue
        for y0, y1 in zip(y0, y1):
            tp[y0] += (y0 == y1)
            tpfn[y0] += 1
            tpfp[y1] += 1
    print()
    for y in sorted(tpfn.keys()):
        pr = (tp[y] / tpfp[y]) if tpfp[y] else 0
        rc = (tp[y] / tpfn[y]) if tpfn[y] else 0
        avg["macro_pr"] += pr
        avg["macro_rc"] += rc
        if not summary:
            print("label = %s" % y)
            print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
            print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
            print("f1 = %f\n" % f1(pr, rc))
    avg["macro_pr"] /= len(tpfn)
    avg["macro_rc"] /= len(tpfn)
    avg["micro_f1"] = sum(tp.values()) / sum(tpfn.values())
    print("macro precision = %f" % avg["macro_pr"])
    print("macro recall = %f" % avg["macro_rc"])
    print("macro f1 = %f" % f1(avg["macro_pr"], avg["macro_rc"]))
    print("micro f1 = %f" % avg["micro_f1"])

    if TASK == "word-segmentation":
        print()
        evaluate_word_segmentation(result)

def evaluate_word_segmentation(result):
    tp, tpfn, tpfp = 0, 0, 0
    isbs = lambda x: x in ("B", "S")
    for _, Y0, Y1 in result:
        i = 0
        tpfn += len(list(filter(isbs, Y0)))
        tpfp += len(list(filter(isbs, Y1)))
        for j, (y0, y1) in enumerate(zip(Y0 + ["B"], Y1 + ["B"])):
            if j and isbs(y0) and isbs(y1):
                tp += (Y0[i:j] == Y1[i:j])
                i = j
    print("TASK = %s" % TASK)
    print("precision = %f (%d/%d)" % (tp / tpfp, tp, tpfp))
    print("recall = %f (%d/%d)" % (tp / tpfn, tp, tpfn))
    print("f1 = %f" % f1(tp / tpfp, tp / tpfn))

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    evaluate(predict(*load_model(sys.argv[1:5]), sys.argv[5]))
