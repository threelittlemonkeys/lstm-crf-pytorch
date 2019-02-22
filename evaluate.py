from predict import *
from collections import defaultdict

def evaluate(result):
    a = [0, 0] # average precision, recall
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
        prec = t[y] / p[y] if p[y] else 0
        rec = t[y] / s[y]
        a[0] += prec
        a[1] += rec
        print("\nlabel = %s" % y)
        print("precision = %f (%d/%d)" % (prec, t[y], p[y]))
        print("recall = %f (%d/%d)" % (rec, t[y], s[y]))
        print("f1 = %f" % f1(prec, rec))
    a = [x / len(s) for x in a]
    print("\nprecision = %f" % a[0])
    print("recall = %f" % a[1])
    print("f1 = %f" % f1(*a))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage: %s model word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    with torch.no_grad():
        evaluate(predict(True))
