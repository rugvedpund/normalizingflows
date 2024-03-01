import parser
from NormalizingFlow import get_fname
from os import path
from sys import exit

argparser = parser.create_parser()
args = argparser.parse_args()

parser.prettyprint(args)

fname = get_fname(args)
if path.exists(fname):
    print("\n\nmodel found")
    print(fname)
    exit(0)
else:
    print("\n\nno model, need to train")
    exit(1)
