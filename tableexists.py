import exists
import NormalizingFlow as nf
import pandas as pd
import itertools

# df = pd.DataFrame(columns=["model","sigma","achromatic","chromatic","gF1","gF5","fF1","fF5"],index=[1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12,1e24])


def println(*args):
    print(*args, "\n", "--" * 40)


args = nf.Args()

sigmas = [6.0, 4.0, 2.0]
chros = [False, True]
dasnrpps = [1e8, 1e9, 1e10, 1e11, 1e12]
cdsnrpps = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
seeds = [0, 1, 2]
cSs = ["", "4", "4 6"]
appLiks = ["_cube", "_smallcube", "_xsmallcube"]
gFs = ["0.0", "0.01", "0.05"]
fFs = ["0.0", "0.01", "0.05"]

println("Dark Ages")
println("Noiseless")

for seed in seeds:
    args.noiseSeed = seed
    println("seed", seed)
    println(f"{'chromatic':^10} {'sigmas':^10} {'exists?':^10}")
    args.SNRpp = 1e24
    for chromatic, sigma in itertools.product(chros, sigmas):
        args.chromatic = chromatic
        args.sigma = sigma
        print(f"{repr(args.chromatic):^10} {args.sigma:^10} {exists.exists(args):^10}")
    println()
    for chromatic, cs in itertools.product(chros, cSs):
        args.chromatic = chromatic
        args.combineSigma = cs
        print(
            f"{repr(args.chromatic):^10} {'2 '+args.combineSigma:^10} {exists.exists(args):^10}"
        )
    println()


# for appLik in appLiks:
#     print("\n\n", appLik,"-------------------")
#     for (fg,freq), chromatic, snrpp, seed, cS in itertools.product(fgs,chros,snrpps,seeds,cSs):
#         args.fgFITS,args.freqs = fg,freq
#         args.chromatic = chromatic
#         args.SNRpp = snrpp
#         args.noiseSeed = seed
#         args.combineSigma = cS
#         args.appendLik = appLik
#         print(f"{args.fgFITS} {args.freqs} {args.chromatic} {args.SNRpp:.0e} {args.noiseSeed} {'2 '+args.combineSigma} {exists.exists(args)}")
