import parser
from NormalizingFlow import get_fname, get_lname
from os import path
from sys import exit

argparser = parser.create_parser()
args = argparser.parse_args()
args.noisyT21=True
args.diffCombineSigma=True
parser.prettyprint(args)

fname = get_fname(args)
if path.exists(fname):
    print("\n\nmodel found")
    print(fname)
else:
    print("\n\nno model, need to train")

lname = get_lname(args,plot='all')
if path.exists(lname):
    print("\n\nlikelihood found")
    print(lname)
else:
    print("\n\nno likelihood")

