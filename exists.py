import glob
import parser
from NormalizingFlow import get_fname, get_lname
from os import path
from sys import exit

def exists(args):
    lname = get_lname(args, plot="all")
    return path.exists(lname)

if __name__ == "__main__":
    argparser = parser.create_parser()
    args = argparser.parse_args()
    args.noisyT21 = True
    args.diffCombineSigma = True
    # parser.prettyprint(args)

    fname = get_fname(args)
    if path.exists(fname):
        print("\n\nmodel found")
        print(fname)
    else:
        print("\n\nmodel missing")

    lname = get_lname(args, plot="A")
    if path.exists(lname):
        print("\n\nlikelihood found")
        print(lname)
    else:
        print("\n\nlikelihood missing")

    # cname = get_lname(args, plot="corner") + ".pdf"
    # if path.exists(cname):
    #     print("\n\ncorner plot found")
    #     print(cname)
    # else:
    #     print("\n\nno corner plot")

    print("\n\nalso found similar models:")
    args.noiseSeed = "*"
    args.append = "*"
    fname = get_fname(args)
    for f in glob.glob(fname):
        print(f)

    print("\n\nand similar likelihoods:")
    args.noiseSeed = "*"
    args.append = "*"
    args.appendLik = "*"
    lname = get_lname(args, plot="all")
    for f in glob.glob(lname):
        print(f)

    print("\n\nand galcuts")
    args.appendLik = ""
    args.galcut = "*"
    lname = get_lname(args, plot="A")
    print(lname)
    for f in glob.glob(lname):
        print(f)
