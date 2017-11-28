import sys
from utils import gpu2cpu

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s model" % sys.argv[0])
    gpu2cpu(sys.argv[1])
