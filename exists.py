import parser
from NormalizingFlow import get_fname
from os import path
from sys import exit

parser = parser.create_parser()
args = parser.parse_args()

args.print()

fname = get_fname(args)
if path.exists(fname):
    print("\n\nmodel found")
    print(fname)
    exit(0)
else:
    print("\n\nno model, need to train")
    exit(1)
